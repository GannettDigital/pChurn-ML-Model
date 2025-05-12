######################################################################
# AUTHOR:     Taylor Larkin <taylor.larkin@datarobot.com>
# DATE:       2022-April-20
# DISCLAIMER: Last tested with DataRobot API version 2.28.0 and python version 3.7.10
######################################################################

import itertools
import time
from typing import List

# Imports
from xmlrpc.client import Boolean

import datarobot as dr
import numpy as np
import pandas as pd
from dask import compute, delayed
from datarobot import QUEUE_STATUS
from datarobot.models.modeljob import wait_for_async_model_creation
from datarobot.models.feature_association_matrix import FeatureAssociationMatrix
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

### Potential Additions ##
# 1) Add ability to supply custom metric (will add when external target data isn't required)
# 2) Add options to keep all models rather than delete leaderboard
# 3) Create "beyond_autopilot" class

### Main Function ###
def beyond_autopilot(
    project_id: str,
    worker_count: int = -1,
    sorting_metric: str = None,
    max_n_models_to_keep: int = 5,
    run_similar_models_for_top_n_models: bool = False,
    accumulation_ratio: float = None,
    try_fam_featurelist: bool = False,
    training_duration_grid: List[
        dr.helpers.partitioning_methods.construct_duration_string
    ] = None,
    use_project_settings: bool = True,
    advanced_tuning_grid: List[List[dict]] = None,
    blend_methods: List[str] = None,
    max_size_of_blender: int = 3,
    remove_redundancy_from_best_model: bool = False,
    mark_project_name: bool = False,
    wait_for_jobs_to_process_timeout: int = 60,
) -> dr.Model:

    """
    Helper to do further model iteration after autopilot is finished. Can be used on any leaderboard.
    The basic flow is:

    1) Get the best models in project at the max training percentage (or unsampled training duration)
        - Runs cross-validation / backtesting when applicable after each step

    2) Run more blueprints from the repository based on the top models
        - Looks at the top best models, extracts the first two words, then finds models in the repo similar to those in name

    3) Perform feature selection (if requested)
        - Runs feature selection across multiple feature lists, subsets of feature lists, and individual models
        - Optionally can also try to create a new feature association matrix (FAM) based feature list from each new feature list created

    4) Tune training duration (if requested)
        - Runs different training durations for time-based partitionined projects

    5) Tune hyperparameters (if requested)
        - Hyperparameters will be tried and ran when the proper "parameter_name" and "value" arguments are found

    6) Create blenders (if requested)
        - Runs every combination of <max_n_models_to_keep> starting at a blender size of 2 and ending at <max_size_of_blender>
        - Deletes worst performing blender models to prevent leaderboard clutter

    7) Return most accurate model (with redundant features removed if requested)
        - Continues to rebuild the model until no redundant feature are found


    Parameters:
    project_id: string of DataRobot project
    worker_count: number of worker you want to use
    sorting_metric: name of metric to sort the leaderboard on
    max_n_models_to_keep: number of models to keep with each iteration
    run_similar_models_for_top_n_models: whether or not to run similarly named models. Based on looking at the first two words from the top "max_n_models_to_keep" models. If not desired, set to False.
    accumulation_raio: accumulation ratio cut-off used during FIRE (e.g., 0.95 == keep features that provide 95% of the accumulated impact). If no selection desired, set to None.
    try_fam_featurelist: whether or not to try using the feature association matrix (FAM) (if available) to extract the cluster labels as a new feature list. If not desired, set to False.
    training_duration_grid: list of "construct_duration_string"s to use for each model. If no tuning desired, set to None.
    use_project_settings: whether or not to use custom backtesting partition for evaluation (only applicable to datetime projects)
    advanced_tuning_grid: nested list of hyperparameters combinations to try. Must contain "parameter_name" and "value" as keys in dictionary. If no tuning desired, set to None.
    blend_methods: list of blending methods to use from datarobot.enums.BLENDER_METHOD. If no blending desired, set to None.
    max_size_of_blender: maximum number of component models in a blender (only used if blender_methods isn't None)
    remove_redundancy_from_best_model: whether or not to rebuild most accurate model with no redundant features
    mark_project_name: appends the phrase '[beyond autopilot]' to the project name denoting the beyond autopilot function has been executed
    wait_for_jobs_to_process_timeout: seconds to wait until pinging DataRobot to see if computed jobs are done

    """

    print("*** Initializing beyond autopilot process ***\n")

    # Create and run project
    project = dr.Project.get(project_id=project_id)

    # Set workers
    project.set_worker_count(worker_count)

    # Determine best partitioning to sort by automatically
    partition_to_sort_by = _determine_partitioning_to_sort_by(project_id=project.id)

    # Do some validations
    _validate_metric(project_id=project.id, metric=sorting_metric)
    _validate_accumulation_ratio(accumulation_ratio=accumulation_ratio)
    _validate_blend_methods(blend_methods=blend_methods)

    # Find best model before any iteration
    best_model_before_iteration = _get_best_models(
        project_id=project.id,
        partition=partition_to_sort_by,
        metric=sorting_metric,
        max_n_models_to_keep=max_n_models_to_keep,
        include_blenders=True,
        select_models_for_feature_selection=False,
        wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
    )[0]

    print("\n*** Starring current best model ***\n")
    best_model_before_iteration.star_model()

    # Run more repo models
    if run_similar_models_for_top_n_models:

        print("\n*** Running more blueprints ***\n")

        models_to_use = _get_best_models(
            project_id=project.id,
            partition=partition_to_sort_by,
            metric=sorting_metric,
            max_n_models_to_keep=max_n_models_to_keep,
            include_blenders=False,
            select_models_for_feature_selection=True,
            wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
        )

        if models_to_use:

            _run_similar_models(
                project_id=project.id,
                models_to_use=models_to_use,
                partition=partition_to_sort_by,
                metric=sorting_metric,
                max_n_models_to_keep=max_n_models_to_keep,
                use_project_settings=use_project_settings,
                wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
            )

        else:

            print(
                "\n*** No models deemed useful for trying to run more blueprints against, moving to next step ***\n"
            )

    # Creating reduced models
    if accumulation_ratio is not None:

        print("\n*** Feature selection ***\n")

        # Get list of models that made it to the last round
        models_to_use = _get_best_models(
            project_id=project.id,
            partition=partition_to_sort_by,
            metric=sorting_metric,
            max_n_models_to_keep=max_n_models_to_keep,
            include_blenders=False,
            select_models_for_feature_selection=True,
            wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
        )

        if models_to_use:

            _create_reduced_models(
                project_id=project.id,
                model_ids=[x.id for x in models_to_use],
                accumulation_ratio=accumulation_ratio,
                try_fam_featurelist=try_fam_featurelist,
                partition=partition_to_sort_by,
                metric=sorting_metric,
                max_n_models_to_keep=max_n_models_to_keep,
                use_project_settings=use_project_settings,
                wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
            )

        else:

            print(
                "\n*** No models deemed useful for feature selection, moving to next step ***\n"
            )

    # Now do training duration
    if project.is_datetime_partitioned:

        if training_duration_grid is not None:

            print("\n*** Training duration tuning ***\n")

            models_to_use = _get_best_models(
                project_id=project.id,
                partition=partition_to_sort_by,
                metric=sorting_metric,
                max_n_models_to_keep=max_n_models_to_keep,
                include_blenders=True,
                select_models_for_feature_selection=False,
                wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
            )

            _tuning_training_duration(
                project_id=project.id,
                models_to_tune=models_to_use,
                duration_grid=training_duration_grid,
                partition=partition_to_sort_by,
                metric=sorting_metric,
                max_n_models_to_keep=max_n_models_to_keep,
                wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
            )

    # Now tune hyperparameters
    if advanced_tuning_grid is not None:

        print("\n*** Hyperparameters tuning ***\n")

        models_to_use = _get_best_models(
            project_id=project.id,
            partition=partition_to_sort_by,
            metric=sorting_metric,
            max_n_models_to_keep=max_n_models_to_keep,
            include_blenders=False,
            select_models_for_feature_selection=False,
            wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
        )

        _tuning_hyperparameters(
            project_id=project.id,
            models_to_tune=models_to_use,
            advanced_tuning_grid=advanced_tuning_grid,
            partition=partition_to_sort_by,
            metric=sorting_metric,
            max_n_models_to_keep=max_n_models_to_keep,
            wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
        )

    # Now do blending
    if blend_methods is not None:

        print("\n*** Model blending ***\n")

        # Get models for blending
        models_to_use = _get_best_models(
            project_id=project.id,
            partition=partition_to_sort_by,
            metric=sorting_metric,
            max_n_models_to_keep=max_n_models_to_keep,
            include_blenders=False,
            select_models_for_feature_selection=False,
            wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
        )

        _create_blenders(
            project_id=project.id,
            models=models_to_use,
            blend_methods=blend_methods,
            max_size_of_blender=max_size_of_blender,
            partition=partition_to_sort_by,
            metric=sorting_metric,
            max_n_models_to_keep=max_n_models_to_keep,
            wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
        )

    print("\n*** Selecting best model ***\n")

    # Getting new list of models
    best_model_after_iteration = _get_best_models(
        project_id=project.id,
        partition=partition_to_sort_by,
        metric=sorting_metric,
        max_n_models_to_keep=max_n_models_to_keep,
        include_blenders=True,
        select_models_for_feature_selection=False,
        wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
    )[0]

    # Check if new model is better
    if best_model_after_iteration.id != best_model_before_iteration.id:

        print(
            "\n*** Beyond autopilot iteration produced a better model... updating best model ***\n"
        )

        # Making sure it's starred
        best_model_after_iteration.star_model()

    else:

        print(
            "\n*** Beyond autopilot iteration did not produce a better model... keeping original best model ***\n"
        )

    # Making this our best model going forward
    best_model = best_model_after_iteration

    # If requested
    if remove_redundancy_from_best_model:

        # Check if a nonredundant version can to be built
        if (not project.unsupervised_mode) and (
            not project.advanced_options.shap_only_mode
        ):

            print("\n*** Redundancy Check ***\n")

            best_model = _redundancy_check(project_id=project.id, model=best_model)

            # Ensuring non-redundant model is also starred
            best_model.star_model()

        else:

            print(
                "\n*** Redundancy removal cannot be applied to an unsupervised or SHAP-based project, skipping this step ***\n"
            )

    print("\n*** Finalizing best model ***\n")

    # If requested
    if mark_project_name:

        _mark_project_name(project_id=project_id)

    # Print info
    _print_performance_info(
        project_id=project.id,
        model=best_model,
        metric=sorting_metric,
        partition=partition_to_sort_by,
    )

    print(f"Model iteration finished! Check leaderboard here: {project.get_uri()}")
    # Up to 3 models will be starred automatically:
    # (1) best model before iteration
    # (2) best model after iteration
    # (3) non-redundant version of whichever is better between (1) and (2) (if redundant features are detected)
    return best_model


### Helpers ###
def error_retry_decision(x) -> Boolean:

    """
    Helper for determining when to retry the block of code

    """

    if hasattr(x, "status_code"):

        if x.status_code == 502:

            print("Server error! Trying again...")
            return True

        if x.status_code == 422:

            if "message" in x.json.keys():

                if "Unable to add jobs to the queue" in x.json["message"]:

                    print("Unable to add jobs to the queue! Trying again...")
                    return True

        if x.status_code == 404:

            if "message" in x.json.keys():

                if (
                    "No matching FAM record was found for this feature list"
                    in x.json["message"]
                ):

                    print("Waiting for FAM computation...")
                    return True

    else:

        return False


def _test_if_project_has_fam(project_id: str) -> Boolean:

    """
    Helper for determining if a project has a feature association matrix

    """

    # Try and see if we can pull any FAM featurelists
    try:

        dr.models.FeatureAssociationFeaturelists.get(project_id)

        return True

    except Exception:

        return False


def _determine_partitioning_to_sort_by(project_id: str) -> str:

    """
    Returns either validation or crossValidation

    project_id: DataRobot project id

    """

    # Initialize
    project = dr.Project.get(project_id)

    # Enter default
    partition = "validation"

    # First determine if it's a datetime partitioned project
    if project.is_datetime_partitioned:

        # See if it has more than 1 backtest
        if len(dr.DatetimePartitioning.get(project_id).backtests) > 1:

            partition = "backtesting"

    # See if cross-validation exists
    if project.partition["validation_type"] == "CV":

        partition = "crossValidation"

    return partition


def _validate_metric(project_id: str, metric: str):

    """
    Checks and returns the metric if valid

    project_id: DataRobot project id
    metric: user-supplied metric

    """

    if metric is not None:

        # Initialize
        project = dr.Project.get(project_id)

        # Pull list of possible metrics
        if project.unsupervised_mode:

            metrics = ["Synthetic AUC", "Synthetic LogLoss"]

        else:

            metrics = project.get_metrics(feature_name=project.target)
            metrics = metrics["available_metrics"]

        # If project has weights
        if project.advanced_options.weights is not None:

            metrics = ["Weighted " + x for x in metrics]

        # Error if not in metrics
        assert metric in metrics, "Metric specified is not an available metric."


def _validate_accumulation_ratio(accumulation_ratio: float):

    """
    Checks and returns the metric if valid

    accumulation_ratio: accumulation ratio cut-off used during FIRE

    """

    if accumulation_ratio is not None:

        # Error if not in metrics
        assert (accumulation_ratio >= 0) and (
            accumulation_ratio <= 1
        ), "Accumulation ratio is not >= 0 and <= 1."


def _validate_blend_methods(blend_methods: List[str]):

    """
    Checks and returns the blend method list if valid

    blend_methods: user-supplied list of blender methods

    """

    if blend_methods:

        # List of possible methods
        methods = [
            "PLS",
            "GLM",
            "ENET",
            "AVG",
            "MED",
            "RF",
            "KERAS",
            "LGBM",
            "FORECAST_DISTANCE_AVG",
            "FORECAST_DISTANCE_ENET",
            "MAX",
            "MIN",
        ]

        for method in methods:

            # Error if not in methods
            assert method in methods, f"Blend method {method} is not valid."


def _select_models_for_feature_selection(
    project_id: str, models: List[dr.Model]
) -> List[dr.Model]:

    """
    Removes model types that aren't useful for FIRE feature selection

    project_id: DataRobot project id
    models: list of DataRobot models

    """

    # Initialize
    project = dr.Project.get(project_id)

    # If unsupervised project, remove blenders
    if project.unsupervised_mode:

        models = [x for x in models if x.model_category != "blend"]

    # Filter out some model types based on name
    models_to_remove = [
        "Baseline Predictions",
        "Mean Response",
        "Majority Class",
        "Auto-Tuned",
    ]
    for i in models_to_remove:

        tmp_model = [x for x in models if i in x.model_type]

        if tmp_model:

            for j in tmp_model:

                models.remove(j)

    # Filter out some model types based on feature list
    tmp_model = [
        x
        for x in [x for x in models if x.featurelist_name]
        if x.featurelist_name.startswith("Baseline")
        or x.featurelist_name.startswith("Date")
    ]

    if tmp_model:

        for j in tmp_model:

            models.remove(j)

    # If we've removed all the models, return None
    if (models == [None] * len(models)) or (len(models) < 2):

        models = None

    return models


def _run_cv(
    project_id: str, models: List[dr.Model], wait_for_jobs_to_process_timeout: int = 60
):

    """
    Runs cross-validation or backtesting on requested models

    project_id: DataRobot project id
    models: list of DataRobot models
    wait_for_jobs_to_process_timeout: seconds to wait until pinging DataRobot to see if computed jobs are done

    """

    # Initialize
    project = dr.Project.get(project_id)
    jobs = []

    # Perform cross-validation / backtesting, if requested
    if project.is_datetime_partitioned:

        message = "Backtesting"
        for model in models:
            try:

                jobs.append(
                    dr.DatetimeModel.get(project_id, model.id).score_backtests()
                )

            except:

                pass

    else:

        message = "Cross-validating"
        for model in models:

            try:

                jobs.append(dr.Model.get(project_id, model.id).cross_validate())

            except:

                pass

    if jobs:

        print(f"{message}...")

        # Wait
        _wait_for_jobs_to_process(project_id, timeout=wait_for_jobs_to_process_timeout)

        print(f"{message} done!")


def _sort_leaderboard(
    project_id: str,
    models: List[dr.Model],
    partition: str = "validation",
    metric: str = None,
) -> List[dr.Model]:

    """
    Sorts the leaderboard by a requested metric and partition

    project_id: DataRobot project id
    models: list of DataRobot models
    partition: string representing which partition to use, by default it uses the "validation"
    metric: metric to sort by

    """
    # Initialize
    project = dr.Project.get(project_id)

    # If metric isn't specified, set to project metric
    if metric is None:

        metric = project.metric

    # If unsupervised project, manually create metrics dict
    if project.unsupervised_mode:

        metrics = {
            "metric_details": [
                {"ascending": False, "metric_name": "Synthetic AUC"},
                {"ascending": True, "metric_name": "Synthetic LogLoss"},
            ],
            "available_metrics": ["Synthetic AUC", "Synthetic LogLoss"],
        }

    else:

        # Pull list of possible metrics
        metrics = project.get_metrics(feature_name=project.target)

    # Capture direction
    ascending = [
        x for x in metrics["metric_details"] if x["metric_name"].startswith(metric)
    ][0]["ascending"]

    # Ensuring we only have models where stuff has a value
    models_with_score = [
        model for model in models if model.metrics[metric][partition] is not None
    ]

    return sorted(
        models_with_score,
        key=lambda model: model.metrics[metric][partition],
        reverse=(not ascending),
    )


def get_max_n_models_to_keep(
    project_id: str,
    models: List[dr.Model],
    partition: str = None,
    metric: str = None,
    max_n_models_to_keep: int = 9,
    wait_for_jobs_to_process_timeout: int = 60,
) -> List[dr.Model]:

    """
    Gets and sorts appropriately a specified number of models from the leaderboard

    project_id: DataRobot project id
    models: list of DataRobot models
    partition: string representing which partition to use, by default it uses the "validation"
    metric: metric to sort by
    max_n_models_to_keep: number of models to keep
    wait_for_jobs_to_process_timeout: seconds to wait until pinging DataRobot to see if computed jobs are done

    """

    # Initialize
    project = dr.Project.get(project_id)

    # Sort by requested metric
    models = _sort_leaderboard(
        project_id=project.id, models=models, partition="validation", metric=metric
    )
    n_models = len(models)

    # Here, prioritize cv / backtested models first
    if partition in ["crossValidation", "backtesting"]:

        # Prioritize cv models
        cv_models = [
            x for x in models if x.metrics[project.metric][partition] is not None
        ]
        no_cv_models = [
            x for x in models if x.metrics[project.metric][partition] is None
        ]
        n_cv_models = len(cv_models)

        # If partition is cross-validaiton
        # Note that if a model object (rather than a datetime), backtesting = cross-validation
        if n_cv_models < max_n_models_to_keep:

            # How many models do we need to compute cv for?
            n_need_cv = max_n_models_to_keep - n_cv_models

            # Select the next best models after those which cv has been ran
            n_no_cv_models = len(no_cv_models)
            models_which_need_cv = no_cv_models[0 : min(n_no_cv_models, n_need_cv)]

            # Run and wait
            _run_cv(
                project_id=project.id,
                models=models_which_need_cv,
                wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
            )

            # Repull models
            model_ids = [x.id for x in cv_models + models_which_need_cv]
            models = [x for x in project.get_models() if x.id in model_ids]

    # Repull models
    top_x_models = _sort_leaderboard(
        project_id=project.id, models=models, partition=partition, metric=metric
    )[0 : min(n_models, max_n_models_to_keep)]

    return top_x_models


@retry(
    wait=wait_fixed(600),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(lambda x: error_retry_decision(x)),
)
def _get_best_models(
    project_id: str,
    partition: str = None,
    metric: str = None,
    max_n_models_to_keep: int = 9,
    include_blenders: bool = True,
    select_models_for_feature_selection: bool = True,
    wait_for_jobs_to_process_timeout: int = 60,
) -> List[dr.Model]:

    """
    Gets list of models at the max training pct or with a time window sample

    project_id: DataRobot project id
    partition: string representing which partition to use, by default it uses the "validation"
    metric: metric to sort by
    max_n_models_to_keep: number of models to keep
    include_blenders: whether or not to include blenders in model retrieval
    select_models_for_feature_selection: whether or not to remove models that wouldn't be useful for feature selection (e.g., auto-tune n-grams modelers)
    wait_for_jobs_to_process_timeout: seconds to wait until pinging DataRobot to see if computed jobs are done

    """

    # Initialize
    project = dr.Project.get(project_id)

    print("Collecting best models...")

    # Ensuring no outstanding jobs are going
    _wait_for_jobs_to_process(project.id, timeout=wait_for_jobs_to_process_timeout)

    # Start with models
    models = project.get_models()

    #     # Start with models
    #     all_models = project.get_models()

    #     # Get final stage models
    #     if project.is_datetime_partitioned:

    #         models = [
    #             x
    #             for x in all_models
    #             if dr.DatetimeModel.get(project.id, x.id).time_window_sample_pct is None
    #         ]

    #     else:

    #         models = [
    #             x for x in all_models if x.training_row_count == project.max_train_rows
    #         ]

    # Whether or not to pull blenders
    if not include_blenders:

        models = [x for x in models if x.model_category != "blend"]

    # Subset to only models we're interested in using for feature selection
    if select_models_for_feature_selection:

        models = _select_models_for_feature_selection(
            project_id=project.id, models=models
        )

        # If we removed them all, return an empty list
        if not models:

            return []

    # Sort models (in case of comprehensive mode with no CV, need to select best models)
    sorted_models = get_max_n_models_to_keep(
        project_id=project_id,
        models=models,
        partition=partition,
        metric=metric,
        max_n_models_to_keep=max_n_models_to_keep,
        wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
    )

    return sorted_models


def _wait_for_jobs_to_process(project_id: str, timeout: int = 60):

    """
    Wait for all jobs (modeling, prediction, etc.) to finish before continuing

    project_id: DataRobot project id
    timeout: seconds to wait before checking jobs in queue

    """

    # Initialize
    project = dr.Project.get(project_id)

    # Get jobs
    all_jobs = len(project.get_all_jobs())

    while all_jobs > 0:

        time.sleep(timeout)

        jobs_list = project.get_all_jobs()
        all_jobs = len(jobs_list)
        jobs_by_type = {}
        for job in jobs_list:
            if job.job_type not in jobs_by_type:
                jobs_by_type[job.job_type] = [0, 0]
            if job.status == QUEUE_STATUS.QUEUE:
                jobs_by_type[job.job_type][0] += 1
            else:
                jobs_by_type[job.job_type][1] += 1
        for type in jobs_by_type:
            (num_queued, num_inprogress) = jobs_by_type[type]
            print(
                "{} jobs: {} queued, {} inprogress".format(
                    type, num_queued, num_inprogress
                )
            )


@retry(
    wait=wait_fixed(600),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(lambda x: error_retry_decision(x)),
)
def _run_similar_models(
    project_id: str,
    models_to_use: List[dr.Model],
    partition: str = None,
    metric: str = None,
    max_n_models_to_keep: int = 9,
    use_project_settings: bool = True,
    wait_for_jobs_to_process_timeout: int = 60,
):

    """
    Runs blueprints from the repository that have the same first two words as the models in models_to_use

    project_id: DataRobot project id
    models_to_use: list of DataRobot models to get similar blueprints for
    partition: string representing which partition to use, by default it uses the "validation"
    metric: metric to sort by
    max_n_models_to_keep: number of models to keep
    use_project_settings: whether or not to use custom backtesting partition for evaluation (only applicable to datetime projects)
    wait_for_jobs_to_process_timeout: seconds to wait until pinging DataRobot to see if computed jobs are done

    """

    # Initialize
    project = dr.Project.get(project_id)
    model_job_ids = []

    # Identify reduced set of blueprints to run
    bps = project.get_blueprints()
    print(f"Total # of blueprints in project: {len(bps)}")

    # Extract first 2 words of each model
    model_phrases = list(
        set([" ".join(x.model_type.split()[:2]) for x in models_to_use])
    )

    # Filtered blueprints
    similar_bps = []
    for i in model_phrases:

        # Save only ones that contain the phrase
        similar_bps = similar_bps + [x for x in bps if i in x.model_type]

    # Check the remaining
    print(f"# of similar blueprints to the top n: {len(similar_bps)}")
    fl_ids = list(set([x.featurelist_id for x in models_to_use]))
    model_job_ids = _run_blueprints(
        project_id=project.id,
        blueprint_ids=[x.id for x in similar_bps],
        featurelist_ids=fl_ids,
        use_project_settings=use_project_settings,
    )

    # Clean-up
    if model_job_ids:

        _run_cv_and_delete_models(
            project_id=project.id,
            model_job_ids=model_job_ids,
            partition=partition,
            metric=metric,
            max_n_models_to_keep=max_n_models_to_keep,
            wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
            check_for_duplicates=True,
        )


@retry(
    wait=wait_fixed(30),
    stop=stop_after_attempt(5),
)
def _aggregate_feature_impacts(
    project_id: str,
    models: List[dr.Model],
) -> pd.DataFrame:

    """
    Helper function to gather all the feature impact and take the median of the normalized score per feature

    project_id: DataRobot project id
    models: list of DataRobot models
    wait_for_jobs_to_process_timeout: seconds to wait until pinging DataRobot to see if computed jobs are done

    """

    # Initialize
    project = dr.Project.get(project_id)
    jobs = []

    # For SHAP models
    if project.advanced_options.shap_only_mode:

        # Compute shap impacts
        for model in models:

            jobs.append(dr.ShapImpact.create(project_id=project.id, model_id=model.id))

            # Sleep to prevent too many API requests at once
            time.sleep(1)

        # Wait
        [x.wait_for_completion(max_wait=60 * 60 * 24) for x in jobs]

        # Collecting impacts
        fi_all = pd.concat(
            [
                pd.DataFrame.from_records(
                    dr.ShapImpact.get(project_id=project.id, model_id=x.id).shap_impacts
                )
                for x in models
            ]
        )

    else:

        # Compute permutation importance
        for model in models:

            try:

                # Try requesting
                jobs.append(model.request_feature_impact())

                # Sleep to prevent too many API requests at once
                time.sleep(1)

            # If job was already ran, collect the job id
            except Exception as error:
                
                pass
            
        # Wait for them to complete
        _wait_for_jobs_to_process(project.id)

        # Collecting impacts
        fi_all = pd.concat(
            [pd.DataFrame.from_records(m.get_feature_impact()) for m in models]
        )

        # Rename (SHAP is already named this way)
        fi_all.rename(
            columns={
                "impactNormalized": "impact_normalized",
                "featureName": "feature_name",
                "redundantWith": "redundant_with",
            },
            inplace=True,
        )

    # If identified as redundant, set to 0
    if "redundant_with" in fi_all.columns:

        fi_all["impact_normalized"] = np.where(
            ~fi_all["redundant_with"].isnull(), 0, fi_all["impact_normalized"]
        )

    # Aggregate the normalized score using the mean (it's possble for median to result in all 0s)
    all_impact_agg = (
        fi_all.groupby("feature_name")[["impact_normalized"]]
        .mean()
        .sort_values("impact_normalized", ascending=False)
        .reset_index()
    )

    return all_impact_agg


def _get_top_features_from_feature_impact(
    impact: pd.DataFrame,
    accumulation_ratio: float = 0.95,
) -> List[str]:
    """
    Returns a list of reduced features to use from feature impact (assumes certain column names in impact and is sorted correctly)

    impact: a pandas dataframe containing the feature impact results
    accumulation_raio: accumulation ratio cut-off (e.g., 0.95 == keep features that provide 95% of the accumulated impact)

    """

    # Initialize
    accumulated_impact = 0.0
    num_features_acc = 0

    # Cut out importances 0 or less
    # Because we zero-ed out redundant features, it will remove them as well
    nonzero_impact = impact.loc[impact["impact_normalized"] > 0].reset_index(drop=True)

    # If only one feature or all are 0, return the first feature
    if nonzero_impact.shape[0] <= 1:

        return [impact["feature_name"].iloc[0]]

    else:

        # Divde by sum
        normalized_impact = nonzero_impact["impact_normalized"].values
        normalized_impact /= np.sum(normalized_impact)

        # Collect the minimum number of features to meet accumulated impact
        while accumulated_impact <= accumulation_ratio and num_features_acc < len(
            normalized_impact
        ):
            accumulated_impact += normalized_impact[num_features_acc]
            num_features_acc += 1

        # Number of features used should be minimum of the following numbers
        feat_cnt = round(
            min(100, max(25, 0.5 * len(normalized_impact)), num_features_acc)
        )

        return nonzero_impact.iloc[0:feat_cnt]["feature_name"].tolist()


@retry(
    wait=wait_fixed(10),
    retry=retry_if_exception(lambda x: error_retry_decision(x)),
)

def _get_feature_association_matrix_cluster_features(
    project_id: str, featurelist_id: str
):

    """
    Helper function that shows how you can get the cluster names from the feature association matrix (FAM) for any featurelist.
    Note that there isn't a formal client function to do this, so calling the API manually.

    Paramters:
    project_id: DataRobot project id
    featurelist_id: DataRobot feature list id

    Returns:
    List of feature names used to define clusters from FAM

    """

    # # Set the client so we can call the API endpoint
    # client = dr.Client()

    # Get feature list name
    featurelist_name = dr.Featurelist.get(project_id, featurelist_id).name

    # Make the API end point request
    try:

        # client.post(
        #     client.endpoint + f"/projects/{project_id}/featureAssociationMatrix/",
        #     data={"featurelistId": featurelist_id},
        # )

        status = FeatureAssociationMatrix.create(project_id, featurelist_id)
        status.wait_for_completion()

    except:

        print(f"Unable to compute FAM on feature list: {featurelist_name}!")
        return []

    # Fetch the data
    # Because it can take a few seconds, we're going to poll until it succeeds (max 100 times)
    fam_data = dr.FeatureAssociationMatrix.get(
        project_id=project_id, featurelist_id=featurelist_id
    )

    # Get cluster names as new features
    cluster_names = [
        x.get("cluster_name").replace(" cluster", "")
        for x in fam_data.features
        if x.get("cluster_name") is not None
    ]

    # Return unique list
    return list(set(cluster_names))


def _get_non_numeric_categorical_features(
    project_id: str, featurelist_id: str
) -> List[str]:

    """
    Returns a list of non-numeric, non-categorical features

    project_id: DataRobot project id
    featurelist_id: DataRobot feature list id List of features to check

    """

    # Initialize
    non_num_cat_features = []

    # Get featurelist info
    fl = [
        x
        for x in dr.Project.get(project_id).get_featurelists()
        if x.id == featurelist_id
    ][0]

    # Collect features
    for f in fl.features:

        # Gather info about feature
        feature_info = dr.Feature.get(project_id, f)

        # Check if it's numeric or categorical
        if feature_info.feature_type not in ["Numeric", "Categorical"]:

            non_num_cat_features = non_num_cat_features + [f]

    return non_num_cat_features


def _create_datarobot_featurelist_id(
    project_id: str, featurelist_name: str, feature_names: List[str]
) -> str:

    """
    Creates new feature lists or return already created featurelist

    project_id: DataRobot project id
    featurelist_name: name to give new feature list
    feature_names: List of features to use in feature list

    """

    # Initialize
    project = dr.Project.get(project_id)

    # Check if featurelist already exists, if not, create it.
    if project.use_time_series:

        existing_featurelists = project.get_modeling_featurelists()

    else:

        existing_featurelists = project.get_featurelists()

    featurelist = [x for x in existing_featurelists if x.name == featurelist_name]

    # Sanitize name
    featurelist_name = featurelist_name.replace("+", "plus")

    # If exists, define it.
    if featurelist:

        featurelist = featurelist[0]
        print(f"Feature list named '{featurelist.name}' extracted!")

    else:

        if project.use_time_series:

            featurelist = project.create_modeling_featurelist(
                featurelist_name,
                features=feature_names,
            )

        else:
                
            featurelist = project.create_featurelist(
                featurelist_name,
                features=feature_names,
            )
            
        print(
            f"Feature list name '{featurelist.name}' created with {len(featurelist.features)} features!"
        )

    return featurelist.id


def _create_fire_featurelist_id(
    project_id: str,
    model_ids: List[str],
    fire_featurelist_name: str,
    accumulation_ratio: float = 0.95,
    try_fam_featurelist: bool = False,
) -> List[str]:

    """
    Performs feature reduction via a variant of FIRE [https://www.datarobot.com/blog/using-feature-importance-rank-ensembling-fire-for-advanced-feature-selection/] / FAM and returns the new feautre list ids

    project_id: DataRobot project id
    model_ids: list of DataRobot model ids to use
    fire_featurelist_name: name to give new FIRE reduced feature list
    accumulation_raio: accumulation ratio cut-off (e.g., 0.95 == keep features that provide 95% of the accumulated impact)
    try_fam_featurelist: whether or not to run the computed feature list on the feature association matrix (FAM) and extract the cluster labels as a new feature list

    """

    # Initialize
    project = dr.Project.get(project_id)
    featurelist_ids = []

    # Get models
    models = [dr.Model.get(project.id, x) for x in model_ids]

    # Collecting impacts
    all_impact_agg = _aggregate_feature_impacts(project_id=project.id, models=models)

    # Apply heuristics to select features
    selected_features = _get_top_features_from_feature_impact(
        impact=all_impact_agg, accumulation_ratio=accumulation_ratio
    )

    # Create feature list
    featurelist_id = _create_datarobot_featurelist_id(
        project_id=project.id,
        featurelist_name=fire_featurelist_name,
        feature_names=selected_features,
    )
    featurelist_ids = featurelist_ids + [featurelist_id]

    # Creating a FAM reduced list based on given feature list
    if try_fam_featurelist and _test_if_project_has_fam(project_id=project.id):

        fam_features = _get_feature_association_matrix_cluster_features(
            project_id=project.id,
            featurelist_id=featurelist_id,
        )

        # Ensure something is returned
        if fam_features:

            print(
                f"Extracted cluster names in FAM from featurelist_id: {featurelist_id}"
            )

            # Add non-numeric / categoricals features back in (if any)
            non_num_cat_features = _get_non_numeric_categorical_features(
                project_id=project_id, featurelist_id=featurelist_id
            )
            fam_features = list(set(fam_features + non_num_cat_features))

            featurelist_fam_id = _create_datarobot_featurelist_id(
                project_id=project.id,
                featurelist_name=f"{fire_featurelist_name} - FAM",
                feature_names=fam_features,
            )

            featurelist_ids = featurelist_ids + [featurelist_fam_id]

        else:

            print(
                f"No cluster names found in FAM from featurelist_id: {featurelist_id}"
            )

    return featurelist_ids


@retry(
    wait=wait_fixed(600),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(lambda x: error_retry_decision(x)),
)
def _run_blueprints(
    project_id: str,
    blueprint_ids: List[str],
    featurelist_ids: List[str] = [None],
    training_duration: dr.helpers.partitioning_methods.construct_duration_string = None,
    use_project_settings: bool = True,
) -> List[str]:

    """
    Runs blueprints against the specified feature lists and returns successful job ids

    project_id: DataRobot project id
    blueprint_ids: list of blueprint ids you want to run against
    featurelist_ids: DataRobot feature list ids you want to run the blueprints on
    training_duration: training duration to use for a given blueprint (only applicable to datetime projects)
    use_project_settings: whether or not to use custom backtesting partition for evaluation (only applicable to datetime projects)

    """

    # Initialize
    project = dr.Project.get(project_id)
    job_ids = []

    # Run each blueprint on the feature list
    if project.is_datetime_partitioned:

        for bp_id in blueprint_ids:

            for fl_id in featurelist_ids:

                try:

                    if use_project_settings:

                        job_ids.append(
                            str(
                                project.train_datetime(
                                    blueprint_id=bp_id,
                                    featurelist_id=fl_id,
                                    use_project_settings=use_project_settings,
                                ).id
                            )
                        )

                    else:

                        job_ids.append(
                            str(
                                project.train_datetime(
                                    blueprint_id=bp_id,
                                    featurelist_id=fl_id,
                                    training_duration=training_duration,
                                ).id
                            )
                        )

                    # Sleep to prevent too many API requests at once
                    time.sleep(1)
                    
                # If job was already ran, collect the job id
                except Exception as error:

                    # Save error info
                    error_data = error.json

                    # Ensure this thing is in the data
                    if "errorName" in error_data:

                        # Get the ID
                        if error.json["errorName"] == "JobAlreadyAdded":

                            if error.json["previousJob"] is not None:
                                
                                job_ids.append(error.json["previousJob"]["id"])

                    pass

    else:

        for bp_id in blueprint_ids:

            for fl_id in featurelist_ids:

                try:

                    job_ids.append(
                        str(project.train(trainable=bp_id, featurelist_id=fl_id))
                    )

                # If job was already ran, collect the job id
                except Exception as error:

                    # Save error info
                    error_data = error.json

                    # Ensure this thing is in the data
                    if "errorName" in error_data:

                        # Get the ID
                        if error.json["errorName"] == "JobAlreadyAdded":
                            
                            if error.json["previousJob"] is not None:

                                job_ids.append(error.json["previousJob"]["id"])

                    pass

    return job_ids


def _run_cv_and_delete_models(
    project_id: str,
    model_job_ids: List[str],
    partition: str = None,
    metric: str = None,
    max_n_models_to_keep: int = 9,
    wait_for_jobs_to_process_timeout: int = 60,
    check_for_duplicates: bool = False,
):

    """
    Collects model objects from the model jobs, requests CV (if possible), and then deletes extra models

    project_id: DataRobot project id
    model_job_ids: list of DataRobot model job ids to use
    partition: string representing which partition to use, by default it uses the "validation"
    metric: metric to sort by
    max_n_models_to_keep: number of models to keep
    wait_for_jobs_to_process_timeout: seconds to wait until pinging DataRobot to see if computed jobs are done
    check_for_duplicates: flag for whether or not to check for duplicate models (can happen is featurelist name is diff)

    """

    # Initialize
    project = dr.Project.get(project_id)

    # Ensure the jobs are unique
    model_job_ids = list(set(model_job_ids))

    # N jobs
    n_model_job_ids = len(model_job_ids)
    if n_model_job_ids == 1:

        print(f"Running {n_model_job_ids} model...")

    else:

        print(f"Running {n_model_job_ids} models...")

    # Find models which need cv / backtesting to be ran on
    models_which_need_cv = []
    for job_id in model_job_ids:

        try:

            models_which_need_cv.append(
                wait_for_async_model_creation(
                    project_id=project.id, model_job_id=job_id, max_wait=60 * 60 * 24
                )
            )

        except:

            pass

    _run_cv(
        project_id=project.id,
        models=models_which_need_cv,
        wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
    )

    print("Models are finished!")

    # Clean up models (need to repull leadeboard for CV metrics)
    model_ids_to_check = [x.id for x in models_which_need_cv]
    _delete_models(
        project_id=project.id,
        models=[x for x in project.get_models() if x.id in model_ids_to_check],
        partition=partition,
        metric=metric,
        max_n_models_to_keep=max_n_models_to_keep,
        check_for_duplicates=True,
    )


@retry(
    wait=wait_fixed(600),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(lambda x: error_retry_decision(x)),
)
def _create_reduced_models(
    project_id: str,
    model_ids: List[str],
    accumulation_ratio: float = 0.95,
    try_fam_featurelist: bool = False,
    partition: str = None,
    metric: str = None,
    max_n_models_to_keep: int = 9,
    use_project_settings: bool = True,
    wait_for_jobs_to_process_timeout: int = 60,
):

    """
    Runs reduced models and deletes some if too many. Consists of 3 steps:

    1) Run FIRE on all model ids supplied
    2) Run FIRE on subsets of features lists (if more than 1 feature list detected)
    3) Run FIRE on each model individually (running on 1 model will approximate "DR Reduced" feature list logic)

    This function can also try to create FAM reduced feature lists from the subsequent FIRE reduced lists.

    project_id: DataRobot project id
    model_ids: list of DataRobot model ids to use
    accumulation_raio: accumulation ratio cut-off (e.g., 0.95 == keep features that provide 95% of the accumulated impact)
    try_fam_featurelist: whether or not to run the computed feature list on the feature association matrix (FAM) and extract the cluster labels as a new feature list
    partition: string representing which partition to use, by default it uses the "validation"
    metric: metric to sort by
    max_n_models_to_keep: number of models to keep
    use_project_settings: whether or not to use custom backtesting partition for evaluation (only applicable to datetime projects)
    wait_for_jobs_to_process_timeout: seconds to wait until pinging DataRobot to see if computed jobs are done

    """

    # Initialize
    project = dr.Project.get(project_id)
    model_job_ids = []

    # Get models
    if project.is_datetime_partitioned:

        models = [dr.DatetimeModel.get(project.id, x) for x in model_ids]

    else:

        models = [dr.Model.get(project.id, x) for x in model_ids]

    # Get featurelist ids (need to remove None from blenders)
    featurelist_ids = list(set([x.featurelist_id for x in models]))
    featurelist_ids = [x for x in featurelist_ids if x]

    print("Creating new feature lists...")

    # 1: See if we can make a feature list from multiple
    if len(featurelist_ids) > 1:

        global_featurelist_ids = _create_fire_featurelist_id(
            project_id=project.id,
            model_ids=[x.id for x in models],
            fire_featurelist_name="Reduced from Multiple Feature Lists",
            accumulation_ratio=accumulation_ratio,
            try_fam_featurelist=try_fam_featurelist,
        )

        # Running each model on global feature list
        for model in models:

            job_ids = _run_blueprints(
                project_id=project.id,
                blueprint_ids=[model.blueprint_id],
                featurelist_ids=global_featurelist_ids,
                training_duration=model.training_duration,
                use_project_settings=use_project_settings,
            )
            model_job_ids = model_job_ids + job_ids

    # 2: Now see if we can compute FIRE on subsets of feature lists
    for fl_id in featurelist_ids:

        # Get relevant models
        models_for_feature_selection = [x for x in models if x.featurelist_id == fl_id]

        # If a featurelist has 2 or more models run FIRE on that subset
        if len(models_for_feature_selection) > 1:

            fire_featurelist_ids = _create_fire_featurelist_id(
                project_id=project.id,
                model_ids=[x.id for x in models_for_feature_selection],
                fire_featurelist_name=f"Reduced from {models_for_feature_selection[0].featurelist_name}",
                accumulation_ratio=accumulation_ratio,
                try_fam_featurelist=try_fam_featurelist,
            )

            # Running one model at a time to ensure the training duration is preserved per model
            for model in models_for_feature_selection:

                job_ids = _run_blueprints(
                    project_id=project.id,
                    blueprint_ids=[model.blueprint_id],
                    featurelist_ids=fire_featurelist_ids,
                    training_duration=model.training_duration,
                    use_project_settings=use_project_settings,
                )
                model_job_ids = model_job_ids + job_ids

    # 3: Finally, create reduced versions individually
    for model in models:

        individual_featurelist_id = _create_fire_featurelist_id(
            project_id=project.id,
            model_ids=[model.id],
            fire_featurelist_name=f"Reduced from M{model.model_number}",
            accumulation_ratio=accumulation_ratio,
            try_fam_featurelist=try_fam_featurelist,
        )

        job_ids = _run_blueprints(
            project_id=project.id,
            blueprint_ids=[model.blueprint_id],
            featurelist_ids=individual_featurelist_id,
            training_duration=model.training_duration,
            use_project_settings=use_project_settings,
        )
        model_job_ids = model_job_ids + job_ids

    # Clean-up
    if model_job_ids:

        # In case all are dups
        if model_job_ids:
            
            _run_cv_and_delete_models(
                project_id=project.id,
                model_job_ids=model_job_ids,
                partition=partition,
                metric=metric,
                max_n_models_to_keep=max_n_models_to_keep,
                wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
                check_for_duplicates=True,
            )


@retry(
    wait=wait_fixed(600),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(lambda x: error_retry_decision(x)),
)
def _tuning_training_duration(
    project_id: str,
    models_to_tune: List[dr.Model],
    duration_grid: List[dr.helpers.partitioning_methods.construct_duration_string],
    partition: str = None,
    metric: str = None,
    max_n_models_to_keep: int = 9,
    wait_for_jobs_to_process_timeout: int = 60,
):

    """
    Runs supplied models against supplied list of duration strings and deletes some if too many

    project_id: DataRobot project id
    models_to_tune: list of DataRobot models to try different training durations
    duration_grid: list of "construct_duration_string"s to use for each model
    partition: string representing which partition to use, by default it uses the "validation"
    metric: metric to sort by
    max_n_models_to_keep: number of models to keep
    wait_for_jobs_to_process_timeout: seconds to wait until pinging DataRobot to see if computed jobs are done

    """

    # Initialize
    project = dr.Project.get(project_id)
    model_job_ids = []

    # Run each blueprint
    for duration in duration_grid:

        for model in models_to_tune:

            # To retrain blenders appropriate, using model.train_datetime
            try:

                model_job_ids.append(
                    str(
                        dr.DatetimeModel.get(project.id, model.id)
                        .train_datetime(
                            featurelist_id=model.featurelist_id,
                            training_duration=duration,
                        )
                        .id
                    )
                )

                # Sleep to prevent too many API requests at once
                time.sleep(1)

            # If job was already ran, collect the job id
            except Exception as error:

                # Save error info
                error_data = error.json

                # Ensure this thing is in the data
                if "errorName" in error_data:

                    # Get the ID
                    if error.json["errorName"] == "JobAlreadyAdded":
                        
                        if error.json["previousJob"] is not None:

                            model_job_ids.append(error.json["previousJob"]["id"])

                pass

    # Clean-up
    if model_job_ids:

        _run_cv_and_delete_models(
            project_id=project.id,
            model_job_ids=model_job_ids,
            partition=partition,
            metric=metric,
            max_n_models_to_keep=max_n_models_to_keep,
            wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
            check_for_duplicates=True,
        )


@retry(
    wait=wait_fixed(600),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(lambda x: error_retry_decision(x)),
)
def _tuning_hyperparameters(
    project_id: str,
    models_to_tune: List[dr.Model],
    advanced_tuning_grid: List[List[dict]],
    partition: str = None,
    metric: str = None,
    max_n_models_to_keep: int = 9,
    wait_for_jobs_to_process_timeout: int = 60,
) -> List[str]:

    """
    Runs supplied models against supplied list of advanced tuning and deletes some if too many

    project_id: DataRobot project id
    models_to_tune: list of DataRobot models to try different training durations
    advanced_tuning_grid: nested list of hyperparameters to try
    partition: string representing which partition to use, by default it uses the "validation"
    metric: metric to sort by
    max_n_models_to_keep: number of models to keep
    wait_for_jobs_to_process_timeout: seconds to wait until pinging DataRobot to see if computed jobs are done

    """

    # Initialize
    project = dr.Project.get(project_id)
    model_job_ids = []

    # For each hyperparameter combo, try running each model
    for hyperparameter_combo in advanced_tuning_grid:

        for model in models_to_tune:

            try:

                tune = model.start_advanced_tuning_session()

                # Go through each dict in list
                for i in range(len(hyperparameter_combo)):

                    tune.set_parameter(
                        parameter_name=hyperparameter_combo[i]["parameter_name"],
                        value=hyperparameter_combo[i]["value"],
                    )

                model_job_ids.append(str(tune.run().id))

            # If error, just skip it
            except Exception as error:

                pass

    # Clean-up
    if model_job_ids:

        _run_cv_and_delete_models(
            project_id=project.id,
            model_job_ids=model_job_ids,
            partition=partition,
            metric=metric,
            max_n_models_to_keep=max_n_models_to_keep,
            wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
            check_for_duplicates=True,
        )


@retry(
    wait=wait_fixed(600),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(lambda x: error_retry_decision(x)),
)
def _run_blenders(
    project_id: str,
    models_to_blend: List[dr.Model],
    blend_methods: List[str],
    max_size_of_blender: int = 3,
    run_combinations: bool = False,
) -> List[str]:

    """
    Runs blender models for each requested blend type (will try every combo) and return successful job ids

    project_id: DataRobot project id
    models_to_blend: list of DataRobot models to use for blending
    blend_methods: list of blending methods to use from datarobot.enums.BLENDER_METHOD
    max_size_of_blender: maximum number of component models in a blender
    run_combinations: boolean denoting if to whether to create exhaustive combinations of the provided model ids

    """

    # Initialize
    project = dr.Project.get(project_id)
    job_ids = []

    # Get model ids
    model_ids = [x.id for x in models_to_blend]

    # Compute combinations, else just run the supplied set of models.
    if run_combinations:

        combos = []
        for L in range(2, max_size_of_blender + 1):

            for subset in itertools.combinations(model_ids, L):

                combos.append(list(subset))

    else:

        combos = [model_ids]

    # Try every combo for each requested type
    for blend_type in blend_methods:

        for combo in combos:

            try:

                job_ids.append(
                    str(project.blend(model_ids=combo, blender_method=blend_type).id)
                )

                # Sleep to prevent too many API requests at once
                time.sleep(1)

            # If job was already ran, collect the job id
            except Exception as error:

                # Save error info
                error_data = error.json

                # Ensure this thing is in the data
                if "errorName" in error_data:

                    # Get the ID
                    if error.json["errorName"] == "JobAlreadyAdded":
                        
                        if error.json["previousJob"] is not None:

                            job_ids.append(error.json["previousJob"]["id"])

                pass

    return job_ids


def _delete_model(model):

    try:

        model.delete()

    except:

        pass


def _delete_models(
    project_id: str,
    models: List[dr.Model],
    partition: str = "validation",
    metric: str = None,
    max_n_models_to_keep: int = 9,
    check_for_duplicates: bool = False,
):
    """
    Sorts and deletes models from supplied list

    project_id: DataRobot project id
    models: list of DataRobot models
    partition: string representing which partition to use, by default it uses the "validation"
    metric: metric to sort by
    max_n_models_to_keep: number of models to keep
    check_for_duplicates: flag for whether or not to check for duplicate models (can happen is featurelist name is diff)

    """
    
        
    # Find and delete dups, if present
    if check_for_duplicates:
        
        models = _check_for_duplicate_models(project_id=project_id, models=models)

    # Sort
    sorted_models = _sort_leaderboard(
        project_id=project_id, models=models, partition=partition, metric=metric
    )[0 : min(len(models), max_n_models_to_keep)]

    # Find model ids to delete
    keep_ids = [x.id for x in sorted_models]
    delete_models = [x for x in models if x.id not in keep_ids]

    # Sort the deleted models so blenders are deleted first
    delete_models = sorted(delete_models, key=lambda x: x.model_category, reverse=False)

    # Check if we need to delete any models
    if delete_models:

        # Delete
        print(f"Deleting {len(delete_models)} of {len(models)} models...")
        jobs = []
        for model in delete_models:

            jobs.append(delayed(_delete_model)(model))

        jobs = compute(*jobs)


def _check_for_duplicate_models(
    project_id: str,
    models: List[dr.Model],
) -> List[str]:

    """
    Collects model objects from the model jobs, checks if any dups exist, and if so, delete them.

    Parameters
    ----------
    project_id: DataRobot project id
    models: list of DataRobot models

    Returns
    -------
    List of model jobs to pass on

    """
    # Initialize
    project = dr.Project.get(project_id)

    # Start of logic
    n_models = len(models)
    if n_models > 1:

        print(f"Checking for duplicates in {n_models} models...")

        # Initialize
        all_model_stats = pd.DataFrame()
        for model in models:

            # Add model stats
            model_stats = pd.json_normalize(model.metrics, sep="_")

            # Drop those who's values are in a list
            # (this messes up the pandas duplicate function)
            model_stats = model_stats[
                [x for x in model_stats.columns.tolist() if not x.endswith("Scores")]
            ]

            # Make id the index
            model_stats.index = [model.id]

            # Add n_features
            model_stats["n_features"] = len(model.get_features_used())

            # Add blueprint id
            model_stats["blueprint_id"] = model.blueprint_id

            # Append
            all_model_stats = pd.concat([all_model_stats, model_stats])

        # Sort by number of features
        all_model_stats = all_model_stats.sort_values("n_features", ascending=True)
        all_model_stats["duplicate"] = all_model_stats.duplicated(
            subset=[
                x for x in all_model_stats.columns.tolist() if x not in "n_features"
            ],
            keep="first",
        )

        # Save for debug
        all_model_stats.to_csv("dup_check.csv")

        # Split into delete and keep
        delete_model_info = all_model_stats.loc[all_model_stats["duplicate"], :]
        keep_model_info = all_model_stats.loc[~all_model_stats["duplicate"], :]

        # If dups, delete them, else pass the model jobs like nothing happened
        if ~delete_model_info.empty:

            # Specify models to delete
            delete_models = [
                dr.Model.get(project.id, x) for x in delete_model_info.index.tolist()
            ]

            # Check if we need to delete any models
            if delete_models:

                # Delete
                print(f"Deleting {len(delete_models)} duplicate models...")
                jobs = []
                for model in delete_models:

                    jobs.append(delayed(_delete_model)(model))

                jobs = compute(*jobs)

            # Return models to keep
            keep_models = [
                dr.Model.get(project.id, x) for x in keep_model_info.index.tolist()
            ]

            return keep_models

        else:

            print("No duplicate models detected!")
            return models

    else:

        return models


def _create_blenders(
    project_id: str,
    models: List[dr.Model],
    blend_methods: List[str],
    max_size_of_blender: int = 3,
    partition: str = None,
    metric: str = None,
    max_n_models_to_keep: int = 9,
    wait_for_jobs_to_process_timeout: int = 60,
):

    """
    Runs blender models and deletes some if too many

    project_id: DataRobot project id
    models: list of DataRobot models to use for blending
    blend_methods: list of blending methods to use from datarobot.enums.BLENDER_METHOD
    max_size_of_blender: maximum number of component models in a blender
    partition: string representing which partition to use, by default it uses the "validation"
    metric: metric to sort by
    max_n_models_to_keep: number of models to keep
    wait_for_jobs_to_process_timeout: seconds to wait until pinging DataRobot to see if computed jobs are done

    """

    # Initialize
    project = dr.Project.get(project_id)

    # Get models to use for blending
    models_to_blend = [x for x in models if x.model_category != "blend"]

    #  Now try blending
    if (
        (not project.advanced_options.shap_only_mode)
        or (not project.use_time_series and not project.unsupervised_mode)
    ) and (len(models_to_blend) > 1):

        # Execute blenders
        model_job_ids = _run_blenders(
            project_id=project.id,
            models_to_blend=models_to_blend,
            max_size_of_blender=max_size_of_blender,
            blend_methods=blend_methods,
            run_combinations=True,
        )

        # Clean-up
        if model_job_ids:

            _run_cv_and_delete_models(
                project_id=project.id,
                model_job_ids=model_job_ids,
                partition=partition,
                metric=metric,
                max_n_models_to_keep=max_n_models_to_keep,
                wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
                check_for_duplicates=True,
            )

    else:

        print(
            "\n*** Project is either SHAP-based or not enough models to blend, skipping blender step ***\n"
        )


def _feature_impact_try_higher_sample_size(
    project_id: str, model: dr.Model
) -> pd.DataFrame:

    """
    Tries building feature impact with a higher sample size

    project_id: DataRobot project id
    model: DataRobot model

    """

    # Initialize
    project = dr.Project.get(project_id)

    if project.is_datetime_partitioned:

        training_row_count = dr.DatetimeModel.get(project.id, model.id).training_info[
            "prediction_training_row_count"
        ]

    else:

        training_row_count = dr.Model.get(project.id, model.id).training_row_count

    # Compute feature impact
    try:

        fi = pd.DataFrame(
            model.get_or_request_feature_impact(
                row_count=min(100000, training_row_count)
            )
        )

    except:

        fi = pd.DataFrame(model.get_or_request_feature_impact())

    return fi


def _create_nonredundant_model(
    project_id: str,
    model: dr.Model,
    wait_for_jobs_to_process_timeout: int = 60,
    use_project_settings: bool = True,
) -> dr.Model:

    """
    Rebuilds model without redundant features

    Note:
    Contrary to other "create" functions, we pull models from the leaderboard rather than building and waiting for them each time.
    This is because we need to output a specific model, as opposed to just try-excepting, silently failing, and then sorting.
    This also allows us to run this function multiple times without erroring (error coming from requested the same model again).

    project_id: DataRobot project id
    model: A DataRobot model
    wait_for_jobs_to_process_timeout: seconds to wait until pinging DataRobot to see if computed jobs are done

    """

    # Initialize
    project = dr.Project.get(project_id)

    # Compute feature impact (should already be calculated)
    fi = pd.DataFrame(model.get_feature_impact())

    # Identify redundant info
    redundant_features = fi.loc[~fi["redundantWith"].isnull(), "featureName"].tolist()

    # If blender, need to remove redundant info from component models
    if model.model_category == "blend":

        # Get blender model object
        blender_model = dr.BlenderModel.get(project.id, model.id)

        # Get model ids
        model_ids = blender_model.model_ids

    else:

        model_ids = [model.id]

    # Retrain component models or model
    featurelist_ids = []
    for model_id in model_ids:

        if project.use_time_series:

            # Get model
            old_model = dr.DatetimeModel.get(project.id, model_id)
            old_model_features = [
                x
                for x in project.get_modeling_featurelists()
                if x.id == old_model.featurelist_id
            ][0].features

        else:

            # Get model
            old_model = dr.Model.get(project.id, model_id)
            old_model_features = [
                x
                for x in project.get_featurelists()
                if x.id == old_model.featurelist_id
            ][0].features

        # Removing only redundant features for a specific model, maintaining their original feature set
        model_features = [x for x in old_model_features if x not in redundant_features]

        # Create non-redundant feature list
        featurelist_id = _create_datarobot_featurelist_id(
            project_id=project.id,
            featurelist_name=f"Removed Redundancy from M{old_model.model_number}",
            feature_names=model_features,
        )

        # Save
        featurelist_ids.append(featurelist_id)

        # Retrain
        _run_blueprints(
            project_id=project.id,
            blueprint_ids=[old_model.blueprint_id],
            featurelist_ids=[featurelist_id],
            training_duration=old_model.training_duration,
            use_project_settings=use_project_settings,
        )

    # Wait for them to complete
    _wait_for_jobs_to_process(project.id, timeout=wait_for_jobs_to_process_timeout)

    # Now train blender (if it's a blender)
    if model.model_category == "blend":

        # Blend them
        models_to_blend = [
            x for x in project.get_models() if x.featurelist_id in featurelist_ids
        ]

        _run_blenders(
            project_id=project.id,
            models_to_blend=models_to_blend,
            blend_methods=[blender_model.blender_method],
            run_combinations=False,
        )

    # Wait for them to complete
    _wait_for_jobs_to_process(project.id, timeout=wait_for_jobs_to_process_timeout)

    # Get models associated with featurelist_ids
    all_models = project.get_models()
    featurelist_models = [x for x in all_models if x.featurelist_id in featurelist_ids]

    # If blender, filter by model ids
    if model.model_category == "blend":

        model_ids_used_to_blend = sorted([x.id for x in featurelist_models])
        blender_models = [
            dr.BlenderModel.get(project.id, x.id)
            for x in all_models
            if x.model_category == "blend"
        ]

        # Now get blender
        new_model = [
            x for x in blender_models if sorted(x.model_ids) == model_ids_used_to_blend
        ][0]

    else:

        # Should only be 1 if it's just 1 model.
        new_model = featurelist_models[0]

    return new_model


def _redundancy_check(
    project_id: str,
    model: dr.Model,
    wait_for_jobs_to_process_timeout: int = 60,
) -> dr.Model:

    """
    Checks and rebuilds model without redundant features iteratively (if detected)

    project_id: DataRobot project id
    model: A DataRobot model
    wait_for_jobs_to_process_timeout: seconds to wait until pinging DataRobot to see if computed jobs are done

    """

    # Initialize
    project = dr.Project.get(project_id)

    # Compute feature impact
    fi = _feature_impact_try_higher_sample_size(project.id, model)

    while (~fi["redundantWith"].isnull()).any():

        print("Redundancy detected - rebuilding model without redundant features...")

        # Compute feature impact and see if any redundant features
        model = _create_nonredundant_model(
            project_id=project.id,
            model=model,
            wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
        )

        print("Checking new model for redundancy...")

        fi = _feature_impact_try_higher_sample_size(project.id, model)

    # Run cross-validation / backtesting (if needed)
    _run_cv(
        project_id=project.id,
        models=[model],
        wait_for_jobs_to_process_timeout=wait_for_jobs_to_process_timeout,
    )

    # Wait for cross-validation / backtesting to complete
    _wait_for_jobs_to_process(project_id)

    # Repull model so cv metrics are updated
    model = [x for x in project.get_models() if x.id == model.id][0]

    return model


def _print_performance_info(
    project_id: str, model: dr.Model, metric: str = None, partition: str = "validation"
):

    """
    Prints leaderboard performance

    project_id: DataRobot project id
    model: DataRobot model
    metric: user-supplied metric
    partition: string representing which partition to use, by default it uses the "validation"

    """

    # Initialize
    project = dr.Project.get(project_id)

    # If no metric, print a default
    if metric is None:

        # Print interpretable metric
        if project.target_type is None:

            metric_name = "Synthetic AUC"

        else:

            if project.target_type == "Regression":

                if project.use_time_series:

                    metric_name = "MASE"

                else:

                    if "Poisson" in project.metric:

                        metric_name = "FVE Poisson"

                    elif "Gamma" in project.metric:

                        metric_name = "FVE Gamma"

                    elif "Tweedie" in project.metric:

                        metric_name = "FVE Tweedie"

                    else:

                        metric_name = "R Squared"

            if project.target_type == "Binary":

                metric_name = "AUC"

            if project.target_type == "Multiclass":

                metric_name = "Balanced Accuracy"

    else:

        metric_name = metric

    # Detect if weights were used
    if "Weighted" in project.metric and "Weighted" not in metric_name:

        metric_name = "Weighted " + metric_name

    #  Print info
    metric_value = round(model.metrics[metric_name][partition], 4)
    print(f"Best model has a {metric_name} value of {metric_value}!")


def _mark_project_name(project_id: str):

    """
    Appends the phrase '[beyond autopilot]' to the project name denoting the beyond autopilot function has been executed

    project_id: DataRobot project id

    """

    # Initialize
    project = dr.Project.get(project_id)

    # Rename if not marked
    if "[beyond autopilot]" not in project.project_name:

        project.rename(project.project_name + " [beyond autopilot]")

        print("Added '[beyond autopilot]' to project name!")