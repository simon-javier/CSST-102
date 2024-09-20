# **House Price Prediction Using Linear Regression**

## Notebook Link:
> https://colab.research.google.com/drive/1Fmf3F9Aj6w7OR8O41_GDv4dbotID2ug3?usp=sharing

### 1. **Introduction**

Linear regression is a foundational technique in predictive modeling used to determine the relationship between a dependent variable and one or more independent variables. The goal is to model this relationship through a linear equation that can predict the value of the dependent variable (in this case, house prices) based on the values of the independent variables (e.g., size, number of bedrooms, age of the house, and proximity to downtown). Linear regression is widely used due to its simplicity and interpretability, making it a popular choice for analyzing real-world data and making future predictions.

In this report, we implemented a linear regression model from scratch using Python and applied it to a dataset containing house prices. The model was trained using the least squares method to minimize the difference between the predicted and actual house prices. We evaluated the model’s performance using Mean Squared Error (MSE) and visualized its predictions.

### 2. **Data Preprocessing**

Before fitting the linear regression model, we performed several data preprocessing steps:
- **Loading the dataset**: The dataset was loaded into a Pandas DataFrame.
- **Handling missing values**: We identified any missing values in the dataset and opted to drop rows with missing data to avoid complications during modeling. In a real-world scenario, we could have used imputation techniques to fill in missing values.
- **Normalization**: To ensure all features were on a similar scale, we normalized the features (`Size (sqft)`, `Bedrooms`, `Age`, and `Proximity to Downtown (miles)`) using `StandardScaler`. This step was crucial as it ensures that no one feature dominates the others in terms of magnitude, allowing the model to weigh each feature appropriately.

### 3. **Model Implementation**

The linear regression model was implemented from scratch using the normal equation method. The normal equation directly computes the best-fitting model parameters (weights and intercept) without requiring iterative optimization.

The formula used was:
$\[
\theta = (X^T X)^{-1} X^T y
\]$
Where:
- $\(X\)$ is the matrix of feature values.
- $\(y\)$ is the vector of target values (house prices).
- $\(\theta\)$ represents the model parameters (coefficients and intercept).

We also defined a `predict` function that allows the model to generate predictions based on new input features. The features were normalized before making predictions to maintain consistency with the training data.

### 4. **Model Training & Evaluation**

The dataset was split into training (80%) and testing (20%) sets to evaluate the model's ability to generalize to new data. The model was trained on the training set, and the following evaluation metrics were used:
- **Mean Squared Error (MSE)**: MSE was used to quantify the difference between the predicted and actual house prices. A lower MSE indicates that the model’s predictions are closer to the true values.

Results:
- **Training Set MSE**: The MSE for the training set was calculated to assess how well the model fit the data.
- **Testing Set MSE**: The MSE for the testing set was calculated to evaluate the model’s performance on unseen data.

We also created a plot to visualize the relationship between `Size (sqft)` and `Price`, showing the actual house prices and the regression line predicted by the model. The plot helped illustrate the model’s accuracy in predicting house prices based on house size.

### 5. **Conclusion**

The linear regression model was successfully implemented from scratch and trained on the house price dataset. The model was able to predict house prices with reasonable accuracy, as demonstrated by the low MSE values. However, a few challenges were encountered during the process:
- **Missing Data**: Dropping rows with missing values reduced the amount of data available for training. In future work, imputation techniques or more sophisticated methods could be used to handle missing data without discarding information.
- **Feature Scaling**: Normalizing the features was crucial for the model’s performance. Without normalization, certain features (like `Size (sqft)`) would dominate others due to their scale.

To further improve the model, we could explore additional techniques such as feature engineering (creating new features from the existing ones), regularization (to prevent overfitting), or using more complex models (such as polynomial regression) for better prediction accuracy.
