import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
from datetime import datetime, timedelta


class ChurnModelEvaluator:
    def __init__(self, 
                 pred_date_col='observation_date',
                 intent_target_col='churned',
                 churn_target_col='true_churned',
                 intent_score_col='churned_1_PREDICTION',
                 churn_score_col='churn_label',
                 intent_date_col='stop_intent_date',
                 intent_pred_label='churned_PREDICTION',
                 churn_pred_label='churn_label',
                 segment_cols=None):
        """
        Initialize the evaluator with column mappings
        
        Parameters:
        -----------
        pred_date_col: str, prediction date column
        intent_target_col: str, intent model target column
        churn_target_col: str, churn model target column
        intent_score_col: str, intent model probability column
        churn_score_col: str, churn model score column
        intent_date_col: str, date when intention was expressed
        segment_cols: list, columns to use for segmentation
        """
        self.pred_date_col = pred_date_col
        self.intent_target_col = intent_target_col
        self.churn_target_col = churn_target_col
        self.intent_score_col = intent_score_col
        self.churn_score_col = churn_score_col
        self.intent_date_col = intent_date_col
        self.intent_pred_label = intent_pred_label
        self.churn_pred_label = churn_pred_label
        self.segment_cols = segment_cols or []

    
    # def evaluate_models(self, df, intention_scores, churn_scores):
    #     """
    #     Evaluate both models using various metrics and approaches
        
    #     Parameters:
    #     -----------
    #     df : pandas.DataFrame
    #         DataFrame with aligned targets
    #     intention_scores : array-like
    #         Predicted probabilities from intention model
    #     churn_scores : array-like
    #         Predicted probabilities from churn model
            
    #     Returns:
    #     --------
    #     Dict containing evaluation metrics
    #     """
    #     metrics = {}
        
    #     # Evaluate on original targets
    #     metrics['intention_auc'] = roc_auc_score(
    #         df[df['has_sufficient_history']]['intention_target'],
    #         intention_scores[df['has_sufficient_history']]
    #     )
    #     metrics['churn_auc'] = roc_auc_score(
    #         df[df['has_sufficient_history']]['churn_target'],
    #         churn_scores[df['has_sufficient_history']]
    #     )
        
    #     # Evaluate on aligned target
    #     metrics['intention_aligned_auc'] = roc_auc_score(
    #         df[df['has_sufficient_history']]['aligned_target'],
    #         intention_scores[df['has_sufficient_history']]
    #     )
    #     metrics['churn_aligned_auc'] = roc_auc_score(
    #         df[df['has_sufficient_history']]['aligned_target'],
    #         churn_scores[df['has_sufficient_history']]
    #     )
        
    #     # Calculate precision-recall curves
    #     precision_intention, recall_intention, _ = precision_recall_curve(
    #         df[df['has_sufficient_history']]['aligned_target'],
    #         intention_scores[df['has_sufficient_history']]
    #     )
    #     precision_churn, recall_churn, _ = precision_recall_curve(
    #         df[df['has_sufficient_history']]['aligned_target'],
    #         churn_scores[df['has_sufficient_history']]
    #     )
        
    #     metrics['intention_avg_precision'] = average_precision_score(
    #         df[df['has_sufficient_history']]['aligned_target'],
    #         intention_scores[df['has_sufficient_history']]
    #     )
    #     metrics['churn_avg_precision'] = average_precision_score(
    #         df[df['has_sufficient_history']]['aligned_target'],
    #         churn_scores[df['has_sufficient_history']]
    #     )
        
    #     return metrics

    # def analyze_temporal_patterns(self, df):
    #     """
    #     Analyze temporal patterns between intention and actual churn
        
    #     Parameters:
    #     -----------
    #     df : pandas.DataFrame
    #         DataFrame with intention and churn dates
            
    #     Returns:
    #     --------
    #     Dict containing temporal analysis metrics
    #     """
    #     temporal_metrics = {}
        
    #     # Calculate time between intention and churn
    #     mask = df[self.intention_date_col].notna() & df[self.churn_date_col].notna()
    #     time_to_churn = (
    #         df[mask][self.churn_date_col] - 
    #         df[mask][self.intention_date_col]
    #     ).dt.days
        
    #     temporal_metrics['median_days_to_churn'] = time_to_churn.median()
    #     temporal_metrics['mean_days_to_churn'] = time_to_churn.mean()
    #     temporal_metrics['std_days_to_churn'] = time_to_churn.std()
        
    #     # Calculate conversion rates
    #     temporal_metrics['intention_to_churn_rate'] = (
    #         df[df[self.churn_date_col].notna()][self.intention_date_col].notna().mean()
    #     )
        
    #     return temporal_metrics


    def evaluate_models(self, df):
        """Evaluate both models performance"""
        metrics = {}
        
        # Basic performance metrics
        # valid_mask = df['has_both_targets']
        
        # Intent model metrics
        metrics['intent'] = {
            'auc': roc_auc_score(
                df[self.intent_target_col],
                df[self.intent_score_col]
            ),
            'confusion_matrix': confusion_matrix(
                df[self.intent_target_col],
                df[self.intent_pred_label]
            ),
            'classification_report': classification_report(
                df[self.intent_target_col],
                df[self.intent_pred_label]
            )
        }
        
        # Rule model metrics
        metrics['rule'] = {
            'confusion_matrix': confusion_matrix(
                df[self.churn_target_col],
                df[self.churn_pred_label]
            ),
            'classification_report': classification_report(
                df[self.churn_target_col],
                df[self.churn_pred_label]
            )
        }
        
        return metrics


    def analyze_segments(self, df, threshold=0.5):
        """Analyze model performance by segments"""
        segment_metrics = {}
        
        for segment_col in self.segment_cols:
            segment_metrics[segment_col] = {}
            
            for segment in df[segment_col].unique():
                mask = (df[segment_col] == segment)
                
                if mask.sum() > 0:  # Only analyze if we have data
                    # Calculate metrics for this segment
                    segment_metrics[segment_col][segment] = {
                        'size': mask.sum(),
                        'intent_auc': roc_auc_score(
                            df[mask][self.intent_target_col],
                            df[mask][self.intent_score_col]
                        ),
                        'intent_precision': precision_score(
                            df[mask][self.intent_target_col],
                            df[mask][self.intent_pred_label]
                        ),
                        'rule_precision': precision_score(
                            df[mask][self.churn_target_col],
                            df[mask][self.churn_pred_label]
                        )
                    }
        
        return segment_metrics


    def analyze_false_positives(self, df, threshold=0.5):
        """Analyze false positives and their impact"""
        fp_analysis = {}
        
        # Calculate false positives for both models
        # valid_mask = df['has_both_targets']
        
        # Intent model FP analysis
        intent_fps = df[#valid_mask & 
                       (df[self.intent_pred_label] >= 1) & 
                       (df[self.intent_target_col] == 0)]
        
        # Rule model FP analysis
        rule_fps = df[#valid_mask & 
                     (df[self.churn_pred_label] >= 1) & 
                     (df[self.churn_target_col] == 0)]
        
        fp_analysis['intent'] = {
            'fp_rate': len(intent_fps) / len(df),#[valid_mask]
            'fp_by_segment': {
                col: intent_fps[col].value_counts(normalize=True)
                for col in self.segment_cols
            }
        }
        
        fp_analysis['rule'] = {
            'fp_rate': len(rule_fps) / len(df),#[valid_mask]
            'fp_by_segment': {
                col: rule_fps[col].value_counts(normalize=True)
                for col in self.segment_cols
            }
        }
        
        return fp_analysis

    
    def analyze_intention_patterns(self, df):
        """Analyze patterns between intention to stop and actual churn"""
        intent_analysis = {}
        
        # Analyze intention to churn relationship
        intent_mask = df['has_intent'] #& df['has_both_targets']
        
        intent_analysis['intent_to_churn'] = {
            'intent_churn_rate': (
                df[intent_mask][self.churn_target_col].mean()
            ),
            'no_intent_churn_rate': (
                df[~df['has_intent']][self.churn_target_col].mean() #& df['has_both_targets']
            )
        }
        
        # Analyze model performance on intention cases
        if intent_mask.any():
            intent_analysis['model_performance_on_intent'] = {
                'intent_model_auc': roc_auc_score(
                    df[intent_mask][self.intent_target_col],
                    df[intent_mask][self.intent_score_col]
                ),
                'rule_model_precision': precision_score(
                    df[intent_mask][self.churn_target_col],
                    df[intent_mask][self.churn_score_col]
                )
            }
        
        # Analyze intention patterns by segment
        intent_analysis['intent_by_segment'] = {
            col: df[intent_mask][col].value_counts(normalize=True)
            for col in self.segment_cols
        }
        
        return intent_analysis
    
    def calculate_business_impact(self, df, cost_per_intervention=100, 
                                revenue_per_save=1000):
        """Calculate business impact of false positives and true positives"""
        impact = {}
        
        for model_type in ['intent', 'churn']:
            pred_col = (self.intent_pred_label if model_type == 'intent' 
                         else self.churn_pred_label)
            target_col = (self.intent_target_col if model_type == 'intent' 
                         else self.churn_target_col)
            
            # Calculate costs and benefits
            interventions = df[pred_col].sum()
            true_pos = ((df[pred_col] >= 1) & (df[target_col] >= 1)).sum()
            false_pos = ((df[pred_col] >= 1) & (df[target_col] == 0)).sum()
            
            total_cost = interventions * cost_per_intervention
            total_benefit = true_pos * revenue_per_save
            
            impact[model_type] = {
                'interventions': interventions,
                'true_positives': true_pos,
                'false_positives': false_pos,
                'total_cost': total_cost,
                'total_benefit': total_benefit,
                'roi': (total_benefit - total_cost) / total_cost if total_cost > 0 else 0,
                'cost_per_true_positive': (total_cost / true_pos 
                                         if true_pos > 0 else float('inf'))
            }
        
        return impact



class ChurnAnalyzer(ChurnModelEvaluator):
    def __init__(self,
                 intervention_success_rate=0.5,    # P() - probability to prevent churn
                 intervention_cost=150,            # Cost of retention action
                 customer_value=1000,              # Value of retained customer
                 top_n_percent=10,                 # Target top N% of customers
                 churn_bucket='High',              # Corresponding bucket of churn model for top risk
                 retention_program='base',         # Type of retention program
                 **kwargs
                ):      
        
        super().__init__(**kwargs)
        self.intervention_success_rate = intervention_success_rate
        self.intervention_cost = intervention_cost
        self.customer_value = customer_value
                     
        self.top_n_percent = top_n_percent
        self.churn_bucket = churn_bucket

        
        # Define different retention programs and their costs/benefits
        self.retention_programs = {
            'base': {
                'cost': intervention_cost,  # Cost to provide  somepremium features
                'success_rate': intervention_success_rate,  # Higher success rate due to added value
                'additional_value': 0#customer_value  # Additional revenue from upsell
            },
            'optimistic': {
                'cost': intervention_cost,  # Cost
                'success_rate': intervention_success_rate*1.25, #+25% to initial success rate
                'additional_value': customer_value*0.1
            },
            'pessimistic': {
                'cost': intervention_cost,  # Cost
                'success_rate': intervention_success_rate*0.75, #-25%
                'additional_value': 0
            }
        }
        
        self.program = self.retention_programs[retention_program]


    def prioritize_interventions(self, df):
        """
        Prioritize customers for intervention based on predictions
        # TODO: prioritize based on expected value
        """
        
        # Calculate cutoff point for top N%
        n_interventions = int(len(df) * self.top_n_percent / 100)
        
        # Prioritize based on each model
        df['churn_priority'] = (
            #(1*(df['risk_label']==self.churn_bucket))
            df['bin_sum']
            .rank(ascending=False, method='first') <= n_interventions
        )
        
        df['intent_priority'] = (
            df[self.intent_score_col]
            .rank(ascending=False, method='first') <= n_interventions
        )
        
        return df
        
    
    def calculate_costs_benefits(self, df_raw):
        """
        Calculate costs and benefits based on model predictions with value-add retention actions
        """
        results = {}
        df = self.prioritize_interventions(df_raw)
            
        for model in ['churn', 'intent']:
            pred_col = (self.intent_pred_label if model == 'intent' 
                     else self.churn_pred_label)
            target_col = (self.intent_target_col if model == 'intent' 
                         else self.churn_target_col)
            priority_mask = df[f'{model}_priority']
            subset = df[priority_mask]
            
            tp = ((subset[pred_col] >= 1) & (subset[target_col] >= 1)).sum()
            fp = ((subset[pred_col] >= 1) & (subset[target_col] == 0)).sum()
            tn = ((subset[pred_col] == 0) & (subset[target_col] == 0)).sum()
            fn = ((subset[pred_col] == 0) & (subset[target_col] >= 1)).sum()
            
            # Calculate financial impact
            # True Positives: Successful interventions
            saved_revenue = tp * self.program['success_rate'] * self.customer_value
            
            additional_revenue = tp * self.program['success_rate'] * self.program['additional_value']

            # Costs of useful interventions
            intervention_costs = tp * self.program['cost']
            
            # False Positives: Unnecessary interventions
            wasted_costs = fp * self.program['cost']
            
            # False Negatives: Missed opportunities
            missed_revenue = fn * self.customer_value
            
            net_impact = (saved_revenue + additional_revenue - 
                        intervention_costs - wasted_costs - missed_revenue)
            
            results[f'{model}_model'] = {
                f'confusion_matrix': {
                    'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
                },
                'metrics': {
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'intervention_rate': (tp + fp) / len(subset)
                },
                'financial_impact': {
                    'saved_revenue': saved_revenue,
                    'additional_revenue': additional_revenue,
                    'intervention_costs': intervention_costs,
                    'wasted_costs': wasted_costs,
                    'missed_revenue': missed_revenue,
                    'net_impact': net_impact,
                    'roi': (saved_revenue + additional_revenue - intervention_costs - wasted_costs) / (intervention_costs + wasted_costs)
                           if intervention_costs > 0 else 0
                }
            }
        
        return results

    def analyze_intervention_strategy(self, df_raw):
        """
        Calculate costs and benefits based on model predictions with value-add retention actions
        """
        results = {}
        df = self.prioritize_interventions(df_raw)
        for model in ['churn', 'intent']:
            priority_mask = df[f'{model}_priority']
            target_col = (self.intent_target_col if model == 'intent' 
                             else self.churn_target_col)
            # Calculate metrics for prioritized interventions
            n_interventions = priority_mask.sum()
            true_positives = (priority_mask & (df[target_col] >= 1)).sum()
            false_positives = (priority_mask & (df[target_col] == 0)).sum()
            
            # Financial impact
            total_cost = n_interventions * self.intervention_cost
            potential_savings = (
                true_positives * self.intervention_success_rate * self.customer_value
            )
            net_impact = potential_savings - total_cost
            
            results[model] = {
                'metrics': {
                    'n_interventions': n_interventions,
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'precision': true_positives / n_interventions if n_interventions > 0 else 0
                },
                'financial': {
                    'total_cost': total_cost,
                    'potential_savings': potential_savings,
                    'net_impact': net_impact,
                    'roi': (net_impact / total_cost if total_cost > 0 else 0)
                }
            }
            
        return results
    