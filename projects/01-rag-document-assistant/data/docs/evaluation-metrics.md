# Choosing Evaluation Metrics

## Why accuracy can mislead

Accuracy is the fraction of predictions that are correct. On imbalanced datasets it is misleading: if only 1% of transactions are fraudulent, a model that predicts "not fraud" for everything achieves 99% accuracy while catching no fraud at all. For this reason, classification problems with rare positive classes are usually evaluated with precision, recall, and related metrics.

## Precision, recall and F1

Precision is the share of predicted positives that are truly positive: TP / (TP + FP). Recall, also called sensitivity or the true positive rate, is the share of actual positives that the model finds: TP / (TP + FN). There is a trade-off between the two that is controlled by the decision threshold. Raising the threshold usually increases precision and lowers recall. The F1 score is the harmonic mean of precision and recall, so it is only high when both are high. When false negatives are much more costly than false positives, as in cancer screening, recall should be prioritized. When false positives are costly, as in spam filtering where a legitimate email could be lost, precision matters more.

## ROC AUC and PR AUC

The ROC curve plots the true positive rate against the false positive rate across all thresholds, and ROC AUC summarizes it as a single number where 0.5 is random guessing and 1.0 is perfect ranking. ROC AUC can look optimistic on heavily imbalanced data because the false positive rate is diluted by the large number of negatives. The precision-recall curve and its area, PR AUC or average precision, focus on the positive class and are more informative when positives are rare.

## Regression metrics

Mean absolute error (MAE) is the average absolute difference between predictions and targets and is expressed in the units of the target. Root mean squared error (RMSE) squares errors before averaging, so it penalizes large errors much more heavily than MAE and is more sensitive to outliers. R squared measures the proportion of variance in the target explained by the model; a value of 0 means the model does no better than always predicting the mean.

## Calibration

A classifier is well calibrated when its predicted probabilities match observed frequencies: among all cases given a probability of 0.8, about 80% should be positive. Calibration is checked with a reliability diagram and can be improved with Platt scaling or isotonic regression. The Brier score measures the mean squared error of predicted probabilities and rewards both calibration and discrimination.
