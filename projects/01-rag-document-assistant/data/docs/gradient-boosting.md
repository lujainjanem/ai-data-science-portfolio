# Gradient Boosting and Tree Ensembles

## Bagging versus boosting

Both bagging and boosting combine many decision trees, but they do so in different ways. Bagging, as used by random forests, trains deep trees independently on bootstrap samples of the data and averages their predictions, which mainly reduces variance. Random forests also consider only a random subset of features at each split, which decorrelates the trees. Boosting trains shallow trees sequentially, where each new tree focuses on the errors of the ensemble built so far, which mainly reduces bias.

## How gradient boosting works

Gradient boosting starts with a simple prediction, such as the mean of the target. At every round it computes the negative gradient of the loss with respect to the current predictions, called pseudo-residuals, and fits a new small tree to them. The new tree's output is multiplied by the learning rate, also called shrinkage, and added to the ensemble. For squared error loss the pseudo-residuals are simply the ordinary residuals, but the same recipe works for any differentiable loss, including log loss for classification.

## Key hyperparameters

The learning rate and the number of trees interact: a smaller learning rate needs more trees but usually generalizes better. Tree depth, or the maximum number of leaves, controls how complex the interactions captured by each tree can be; depths between 3 and 8 are typical. Subsampling rows and columns for each tree adds randomness that reduces overfitting. Early stopping on a validation set is the standard way to choose the number of trees.

## XGBoost, LightGBM and CatBoost

XGBoost added a regularized objective with penalties on the number of leaves and leaf weights, plus efficient handling of missing values by learning a default direction at each split. LightGBM speeds up training with histogram-based splitting and grows trees leaf-wise instead of level-wise, which can reach lower loss with fewer nodes but may overfit on small datasets. CatBoost is designed for categorical features: it uses ordered target statistics to encode categories without leaking the target, and it builds symmetric (oblivious) trees.

## When to use tree ensembles

On tabular data with a mix of numeric and categorical columns, gradient boosted trees are usually the strongest baseline and often outperform neural networks. They need no feature scaling, handle non-linear interactions automatically, and provide feature importance scores, although SHAP values give a more reliable explanation of individual predictions.
