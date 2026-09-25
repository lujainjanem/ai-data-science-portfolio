# Feature Engineering and Preprocessing

## Handling missing values

Missing values can be dropped, imputed, or modelled. Dropping rows is simple but wastes data and can introduce bias if values are not missing at random. Simple imputation replaces missing entries with the mean, median or most frequent value; the median is more robust when a feature has outliers. More advanced options include k-nearest-neighbours imputation and iterative imputation, which predicts each feature from the others. Adding a binary indicator column that records whether the value was missing often helps, because missingness itself can be informative.

## Encoding categorical variables

One-hot encoding creates a binary column for each category and works well when the number of categories is small. Ordinal encoding maps categories to integers and is appropriate only when the categories have a natural order, such as small, medium and large. For high-cardinality features such as postcodes or product IDs, target encoding replaces each category with the average target value for that category; it must be computed out-of-fold or with smoothing, otherwise it leaks the target and overfits.

## Feature scaling

Standardization subtracts the mean and divides by the standard deviation, giving features zero mean and unit variance. Min-max scaling maps values into a fixed range, usually 0 to 1, and is sensitive to outliers. Robust scaling uses the median and interquartile range instead. Scaling matters for algorithms based on distances or gradients, such as k-nearest neighbours, support vector machines, k-means and neural networks, and for models with L1 or L2 penalties. Tree-based models are invariant to monotonic transformations of individual features and do not need scaling.

## Transforming skewed features

Strongly right-skewed features such as income or house prices are often log-transformed to reduce the influence of extreme values and make relationships more linear. The Box-Cox transformation generalizes this but requires strictly positive values, while the Yeo-Johnson transformation also handles zero and negative values.

## Creating new features

Domain knowledge is the richest source of features. Examples include ratios such as debt to income, time-based features such as day of week or time since the last purchase, and aggregations such as a customer's average order value. Interaction features multiply or combine two variables when their joint effect matters. For dates, cyclical encoding with sine and cosine keeps December close to January.
