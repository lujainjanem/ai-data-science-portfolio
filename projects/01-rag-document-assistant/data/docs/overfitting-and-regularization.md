# Overfitting and Regularization

## What overfitting looks like

A model overfits when it learns patterns that are specific to the training set, including its noise, instead of patterns that generalize. The classic symptom is a large gap between training performance and validation performance: training error keeps falling while validation error flattens out or starts to rise. Underfitting is the opposite problem, where the model is too simple to capture the real structure and performs poorly on both sets.

## The bias-variance trade-off

Prediction error can be decomposed into bias, variance, and irreducible noise. High bias means the model makes strong simplifying assumptions and misses relevant relationships. High variance means the model is very sensitive to the particular training sample, so a small change in the data produces a very different model. Increasing model complexity usually lowers bias but raises variance. Regularization is the main tool for moving along this trade-off without changing the model family.

## L1 and L2 penalties

L2 regularization, also called ridge or weight decay, adds the sum of squared weights to the loss. It shrinks all weights towards zero smoothly but rarely makes any of them exactly zero. L1 regularization, also called the lasso, adds the sum of absolute weights. Because of the shape of the absolute value penalty, L1 drives many weights to exactly zero, which makes it useful for feature selection. Elastic net combines both penalties and is helpful when groups of features are strongly correlated. The regularization strength, often called lambda or alpha, is a hyperparameter that should be tuned on validation data rather than on the test set.

## Regularization in neural networks

Dropout randomly sets a fraction of activations to zero during training, which prevents units from co-adapting and acts like training an ensemble of thinned networks. At inference time dropout is switched off. Early stopping monitors validation loss and halts training when it has not improved for a set number of epochs, called the patience. Data augmentation creates modified copies of training examples, such as flipped or cropped images, which increases the effective size of the dataset. Batch normalization also has a mild regularizing effect because of the noise introduced by batch statistics.

## More data and simpler models

Collecting more representative training data is often the most reliable cure for overfitting, because it reduces variance directly. When that is not possible, reducing the number of features, pruning decision trees, or choosing a smaller architecture are practical alternatives.
