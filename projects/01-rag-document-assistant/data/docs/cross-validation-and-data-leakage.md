# Cross-Validation and Data Leakage

## Train, validation and test splits

A reliable evaluation keeps three kinds of data apart. The training set is used to fit model parameters. The validation set is used to compare models and tune hyperparameters. The test set is used once, at the very end, to estimate performance on unseen data. If the test set is used repeatedly to make decisions, its score stops being an unbiased estimate, because the modelling choices have been fitted to it.

## K-fold cross-validation

In k-fold cross-validation the data is split into k folds of roughly equal size. The model is trained k times, each time holding out a different fold for validation and training on the remaining k minus 1 folds. The k validation scores are averaged, and their spread gives a sense of how stable the estimate is. Five or ten folds are common choices. Stratified k-fold keeps the class proportions the same in every fold, which is important for imbalanced classification. Group k-fold ensures that all rows from the same group, such as the same patient or customer, land in the same fold.

## Time series need special handling

Shuffling rows before splitting is wrong for time-ordered data because it lets the model train on the future and predict the past. Instead, use a forward-chaining scheme, sometimes called rolling-origin or time series split, where every validation window comes strictly after the data used for training.

## What data leakage is

Data leakage happens when information that would not be available at prediction time finds its way into training. Leakage produces evaluation scores that look excellent and then collapse in production. A common form is target leakage, where a feature is a consequence of the label, for example using "number of days in hospital" to predict whether a patient will be admitted.

## Preprocessing leakage and pipelines

Another common source of leakage is fitting preprocessing steps on the full dataset before splitting. If a scaler, imputer, or feature selector sees the validation rows, statistics from those rows leak into training. The fix is to fit every preprocessing step only on the training portion of each fold. In scikit-learn this is done by wrapping preprocessing and the model together in a Pipeline and passing the whole pipeline to cross_val_score or GridSearchCV. Duplicated rows that appear in both training and test sets are a further, easily overlooked, source of leakage.
