import datarobot as dr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional


class Fire:
    def __init__(self, project_id: str):
        """
        Parameters:
        -----------
        project_id: str, id of DR project,
        """
        self.project_id = project_id
        self.project = dr.Project.get(project_id)
        self.all_impact = pd.DataFrame()

    def feature_importance_rank_ensembling(
        self,
        n_models: int = 5,
        metric: Optional[str] = None,
        by_partition: str = "validation",
        feature_list_name: Optional[str] = None,
        ratio: float = 0.95,
        model_search_params: Optional[dict] = None,
        use_ranks: bool = True,
    ):
        """
        Function that implements the logic of Feature Selection using Feature Importance Rank Ensembling and restarts DR autopilot

        Parameters:
        -----------
        n_models: int, get top N best models on the leaderboard to compute feature impact on. Default 5
        metric: str, DR metric to check performance against. Default None. If Default, it will use DR project defined metric
        by_partition: str, whether to use 'validation' or 'crossValidation' partition to get the best model on. Default 'validation'
        feature_list_name: str, name of the feature list to start iterating from. Default None
        ratio: float, ratio of total feature impact that new feature list will contain. Default 0.95
        model_search_params: dict, dictionary of parameters to search the best model. See official DR python api docs. Default None
        use_ranks: Boolean, True to use median rank aggregation or False to use total impact unnormalized. Default True
        """
        
        if by_partition == 'backtesting' and self.project.is_datetime_partitioned:
            backtest_job = None
            for model in self.project.get_models():
                try:
                    backtest_job = model.score_backtests()
                except:
                    pass
            
            if isinstance(backtest_job, dr.Job):
                #wait until the last job is completed, simplification of waiting fol all jobs over the loop
                backtest_job.wait_for_completion()
        
        models = self.get_best_models(
            metric=metric,
            by_partition=by_partition,
            start_featurelist_name=feature_list_name,
            model_search_params=model_search_params,
        )

        models = models.values[:n_models]

        print("Request Feature Impact calculations")
        # first kick off all FI requests, let DR deal with parallelizing
        for model in models:
            try:
                model.request_feature_impact(
                    # row_count = 1e5,
                    # metadata = True,
                )
            except:
                pass

        for model in models:
            # This can take some time to compute feature impact
            feature_impact = pd.DataFrame(
                model.get_or_request_feature_impact(
                    max_wait=900,
                )
            )  # 15min

            # Track model name and ID for bookkeeping purposes
            feature_impact["model_type"] = model.model_type
            feature_impact["model_id"] = model.id
            # By sorting and re-indexing, the new index becomes our 'ranking'
            feature_impact = feature_impact.sort_values(
                by="impactUnnormalized", ascending=False
            ).reset_index(drop=True)
            feature_impact["rank"] = feature_impact.index.values

            # Add to our master list of all models' feature ranks
            self.all_impact = pd.concat(
                [self.all_impact, feature_impact], ignore_index=True
            )

        # We need to get a threshold number of features to select based on cumulative sum of impact
        all_impact_agg = (
            self.all_impact.groupby("featureName")[
                ["impactNormalized", "impactUnnormalized"]
            ]
            .sum()
            .sort_values("impactUnnormalized", ascending=False)
            .reset_index()
        )

        # calculate cumulative feature impact and take first features that possess <ratio> of total impact
        all_impact_agg["impactCumulative"] = all_impact_agg[
            "impactUnnormalized"
        ].cumsum()
        total_impact = all_impact_agg["impactCumulative"].max() * ratio
        tmp_fl = list(
            set(
                all_impact_agg[all_impact_agg.impactCumulative <= total_impact][
                    "featureName"
                ].values.tolist()
            )
        )

        # that will be a number of feature to use
        n_feats = len(tmp_fl)

        if use_ranks:
            # get top features based on median rank
            top_ranked_feats = list(
                self.all_impact.groupby("featureName")['rank']
                .median()
                .sort_values(ascending=True)
                .head(n_feats)
                .index.values
            )
        else:
            # otherwise get features based just on total unnormalized feature impact
            top_ranked_feats = list(all_impact_agg.featureName.values[:n_feats])

        # Create a new featurelist
        featurelist = self.project.create_modeling_featurelist(
            f"Reduced FL by Median Rank, top{n_feats}", top_ranked_feats
        )
        featurelist_id = featurelist.id

        # Start autopilot
        print("Starting AutoPilot on a reduced feature list")
        self.project.start_autopilot(
            mode=dr.enums.AUTOPILOT_MODE.COMPREHENSIVE,
            featurelist_id=featurelist_id,
            prepare_model_for_deployment=False,
            blend_best_models=False,
        )
        self.project.wait_for_autopilot(verbosity=0)
        print("... AutoPilot is completed.")

    
    def get_best_models(
        self,
        metric: Optional[str] = None,
        by_partition: str = "validation",
        start_featurelist_name: Optional[str] = None,
        model_search_params: Optional[dict] = None,
    ) -> pd.Series:
        """
        Gets pd.Series of DR model objects sorted by performance. Excludes blenders, frozend and on DR Reduced FL

        Parameters:
        -----------
        metric: str, metric to use for sorting models on lb, if None, default project metric will be used. Default None
        by_partition: boolean, whether to use 'validation' or 'crossValidation' partitioning. Default 'validation'
        start_featurelist_name: str, initial featurelist name to get models on. Default None
        model_search_params: dict to pass model search params. Default None

        Returns:
        -----------
        pd.Series of dr.Model objects, not blender, not frozen and not on DR Reduced Feature List
        """

        all_metrics = self.project.get_metrics(feature_name=self.project.target)
        all_metrics_list = all_metrics['available_metrics']
        all_metric_details = all_metrics['metric_details']

        if not metric:
            metric = self.project.metric
        else:
            if metric not in all_metrics_list:
                raise f"Metric you selected is unavailable. Please pick available metric: {','.join(all_metrics_list)}"

        asc_flag = [mdict.get('ascending', True) for mdict in all_metric_details if mdict.get('metric_name') == metric][0]

        if self.project.is_datetime_partitioned:
            
            assert by_partition in [
                "validation",
                "backtesting",
                "holdout",
            ], "Please specify correct partitioning, in datetime partitioned projects supported options are: 'validation', 'backtesting', 'holdout' "
            
            models_df = pd.DataFrame(
                [
                    [
                        model.metrics[metric]["validation"],
                        model.metrics[metric]["backtesting"],
                        model.model_category,
                        model.is_frozen,
                        model.featurelist_name,
                        model,
                    ]
                    for model in self.project.get_datetime_models()
                ],
                columns=[
                    "validation",
                    "backtesting",
                    "category",
                    "is_frozen",
                    "featurelist_name",
                    "model",
                ],
            ).sort_values([by_partition], ascending=asc_flag, na_position="last")

        else:
            assert by_partition in [
                "validation",
                "crossValidation",
                "holdout",
            ], "Please specify correct partitioning, supported options are: 'validation', 'crossValidation', 'holdout' "
            models_df = pd.DataFrame(
                [
                    [
                        model.metrics[metric]["crossValidation"],
                        model.metrics[metric]["validation"],
                        model.model_category,
                        model.is_frozen,
                        model.featurelist_name,
                        model,
                    ]
                    for model in self.project.get_models(
                        with_metric=metric, search_params=model_search_params
                    )
                ],
                columns=[
                    "crossValidation",
                    "validation",
                    "category",
                    "is_frozen",
                    "featurelist_name",
                    "model",
                ],
            ).sort_values([by_partition], ascending=asc_flag, na_position="last")

        if start_featurelist_name:
            return models_df.loc[
                (
                    (models_df.category == "model")
                    & (models_df.is_frozen == False)
                    & (models_df.featurelist_name == start_featurelist_name)
                ),
                "model",
            ]
        else:
            return models_df.loc[
                (
                    (models_df.category == "model")
                    & (models_df.is_frozen == False)
                    & (
                        models_df.featurelist_name.str.contains("DR Reduced Features M") == False
                    )
                ),
                "model",
            ]

    def main_feature_selection(
        self,
        start_featurelist_name: Optional[str] = None,
        lives: int = 3,
        top_n_models: int = 5,
        partition: str = "validation",
        main_scoring_metric: Optional[str] = None,
        initial_impact_reduction_ratio: float = 0.95,
        best_model_search_params: Optional[dict] = None,
        use_ranks: bool = True,
    ):
        """
        Main function. Meant to get the optimal shortest feature list by repeating feature selection process until stop criteria is met.
        Currently supports Binary, Regression, Multiclass, Datetime partitioned(OTV) and AutoTS DataRobot projects.

        Example usage:
        >> import datarobot as dr
        >> dr.Client(config_path='PATH_TO_DR_CONFIG/drconfig.yaml')
        TIP: set best_model_search_params = {'sample_pct__lte': 65} to avoid using models trained on a higher sample size
        than 3rd stage of autopilot, which is typically ~64% of the data

        >> main_feature_reduction(
                                  start_featurelist_name=None,
                                  lives=3,
                                  top_n_models=5,
                                  partition='validation',
                                  main_scoring_metric=None,
                                  initial_impact_reduction_ratio=0.95,
                                  best_model_search_params=None,
                                  use_ranks=True)

        Parameters:
        -----------
        start_featurelist_name: str, name of feature list to start iterating from. Default None
        lives: int, stopping criteria, if no best model produced after lives iterations, stop feature reduction. Default 3
        top_n_models: int, only for 'Rank Aggregation method', get top N best models on the leaderboard. Default 5
        partition: str, whether to use 'validation','crossValidation' or 'backtesting' partition to get the best model on. Default 'validation'
        main_scoring_metric: str, DR metric to check performance against, If None DR project metric will be used
        initial_impact_reduction_ratio: float, ratio of total feature impact that new feature list will contain. Default 0.95
        best_model_search_params: dict, dictonary of parameters to search the best model. See official DR python api docs. Default None
        use_ranks: Boolean, True to use median rank aggregation or False to use total impact unnormalized. Default True

        Returns:
        ----------
        dr.Model object of the best model on the leaderboard
        """
        self.project = dr.Project.get(self.project_id)

        ratio = initial_impact_reduction_ratio
        assert ratio < 1, "Please specify initial_impact_reduction_ratio < 1"
        assert lives > 0, "Please provide at least one life"

        model_search_params = best_model_search_params

        # find the current best model
        best_model = self.get_best_models(
            metric=main_scoring_metric,
            by_partition=partition,
            model_search_params=model_search_params,
        ).values[0]

        runs = 0
        # main function loop
        while lives > 0:
            if runs > 0:
                start_featurelist_name = None
            try:
                # run FIRE
                self.feature_importance_rank_ensembling(
                    n_models=top_n_models,
                    metric=main_scoring_metric,
                    by_partition=partition,
                    feature_list_name=start_featurelist_name,
                    ratio=ratio,
                    model_search_params=best_model_search_params,
                    use_ranks=use_ranks,
                )
            except dr.errors.ClientError as e:
                # decay the ratio
                ratio *= ratio
                print(
                    e, f"\nWill try again with a ratio decay ...  New ratio={ratio:.3f}"
                )
                if ratio <= 0.01:
                    break
                else:
                    continue

            ##############################
            ##### GET NEW BEST MODEL #####
            ##############################
            # find the best model now that we've reduced our feature list via FIRE
            new_best_model = self.get_best_models(
                metric=main_scoring_metric,
                by_partition=partition,
                model_search_params=model_search_params,
            ).values[0]

            #################################
            ##### PROCESS STOP CRITERIA #####
            #################################
            if best_model.id == new_best_model.id:
                # if no better model is produced with a recent run, burn 1 life
                lives -= 1

                # if no lives left -> stop
                if lives <= 0:
                    print(
                        "New model is worse. No lives left.\nAUTOMATIC FEATURE SELECTION PROCESS HAS BEEN STOPPED"
                    )
                    return best_model

                # decay the ratio
                ratio *= ratio
                print(
                    f"New model is worse. One life is burnt. "
                    f"Repeat again with decaying the cumulative impact ratio. New ratio={ratio:.3f}"
                )

            best_model = new_best_model
            runs += 1
            print("Run ", runs, " completed")

        return best_model

    def plot_feature_impacts_normalized(self, n_features: int = 25):
        """
        Plots box plot of normalized feature impacts over all models

        Parameters:
        -----------
        n_features: int, number of most impactful features to show in the plot
        """
        ordered_fi = (
            self.all_impact.groupby(["featureName"])["impactNormalized"]
            .agg(["mean"])
            .sort_values("mean", ascending=False)
            .index[:n_features]
        )

        _, ax = plt.subplots(figsize=(12, 10))
        sns.boxplot(
            y="featureName",
            x="impactNormalized",
            data=self.all_impact[self.all_impact.featureName.isin(ordered_fi)],
            order=ordered_fi,
            ax=ax,
            orient="h",
        )
        _ = ax.set_ylabel("Feature Name")
        _ = ax.set_xlabel("Normalized Impact")
        plt.show()
