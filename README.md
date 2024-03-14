![logo](https://github.com/vasanthgx/regularisation_in_ml/blob/main/images/resizedlogo1.png)
# Regularization in ML
 Effect of Lasso and Ridge Regularization on the California Housing Dataset
 
 <img src="https://github.com/Anmol-Baranwal/Cool-GIFs-For-GitHub/assets/74038190/b3fef2db-e671-4610-bb84-1d65533dc5fb" width="300" align='center'>

<br><br>

# Project Title

Analysing the Effect of Regularization in Feature Selection


## Implementation Details

- Dataset: California Housing Dataset (view below for more details)
- Model: [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- Model: [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- Input: 8 features - Median Houshold income, House Area, ...
- Output: House Price

## Dataset Details

This dataset was obtained from the StatLib repository ([Link](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html))

This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surprisingly large values for block groups with few households and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the sklearn.datasets.fetch_california_housing function.

- [California Housing Dataset in Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- 20640 samples
- 8 Input Features: 
    - MedInc median income in block group
    - HouseAge median house age in block group
    - AveRooms average number of rooms per household
    - AveBedrms average number of bedrooms per household
    - Population block group population
    - AveOccup average number of household members
    - Latitude block group latitude
    - Longitude block group longitude
- Target: Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000)

## Evaluation and Results

#### Effect of Lasso Regularization on the features
![alt text](https://github.com/vasanthgx/regularisation_in_ml/blob/main/images/lasso.png)

#### No Effect of Ridge Regularization on the features
![alt text](https://github.com/vasanthgx/regularisation_in_ml/blob/main/images/ridge.png)




## Comparison of R2 scores for the california housing dataset

| model | R2 score |No of features|
|-------|--------|------|
|Ridge Regressor| 0.60|8|
| Lasso Regressor | 0.55 |5|
| SVM Regressor | 0.16|8|
| KNN Regressor | 0.14|8|
                 

The above quant results show that we have a resonable R2 score with Lasso when compared with Ridge - with reduced number of features.

## Key Takeaways

**Lasso Regularization (L1 regularization):**

Lasso regularization adds a penalty term to the linear regression cost function, which is proportional to the absolute values of the coefficients.
It encourages sparsity in the coefficient values by shrinking less important features towards zero.
Lasso regularization can effectively perform feature selection by setting some coefficients to exactly zero, effectively removing those features from the model.
**Ridge Regularization (L2 regularization):**

Ridge regularization adds a penalty term to the linear regression cost function, which is proportional to the squared magnitudes of the coefficients.
It tends to shrink the coefficients of correlated features towards each other, rather than eliminating them entirely.
Ridge regularization helps in reducing the complexity of the model and improving its robustness against multicollinearity.
**Impact on Feature Selection:**

Lasso regularization tends to produce sparse models, where only a subset of features are chosen, effectively performing feature selection.
Ridge regularization, on the other hand, generally includes all features in the model but reduces the impact of less important features.
The choice between Lasso and Ridge regularization depends on the problem at hand and the desired behavior regarding feature selection and model complexity.
**Regularization Strength:**

Both Lasso and Ridge regularization techniques have hyperparameters that control the strength of regularization.
Increasing the regularization strength in Lasso regularization tends to drive more coefficients to zero, thereby increasing the sparsity of the model.
Similarly, increasing the regularization strength in Ridge regularization shrinks the coefficients towards zero but typically does not lead to exact zeros.
**Trade-off between Bias and Variance:**

Both regularization techniques introduce a bias into the model to reduce variance, helping to prevent overfitting.
The choice of the regularization parameter balances the trade-off between bias and variance in the model, affecting its predictive performance.

## How to Run

The code is built on Jupyter notebook

```bash
Simply download the repository, upload the notebook and dataset on colab, and hit play!
```


## Roadmap

We can do the following and try to get better results

- Try more models

- Wrapped Based Feature Selection


## Libraries 

**Language:** Python

**Packages:** Sklearn, Matplotlib, Pandas, Seaborn


## FAQ

#### How does the Lasso Regularization work?

 Lasso regularization adds an L1 penalty to the standard linear regression cost function, proportional to the absolute values of coefficients. This penalty encourages sparsity in the coefficients, effectively performing feature selection by shrinking less important features towards zero. The optimization problem is solved to minimize the modified cost function, adjusting coefficients to strike a balance between fit and complexity. As the regularization parameter increases, more coefficients are shrunk towards zero, potentially leading to some coefficients being exactly zero, thus eliminating corresponding features. Tuning the regularization parameter is crucial to control the trade-off between bias and variance in the model.

#### How do you train the model on a new dataset?

**Data Preparation**: Organize your dataset into features (independent variables) and the target variable (dependent variable).

**Split Data**: Divide the dataset into two subsets: training data and testing data. The training set is used to train the model, while the testing set is used to evaluate its performance.

**Model Training**: Use the training data to fit the linear regression model. This involves finding the coefficients (weights) that minimize the difference between the predicted values and the actual values of the target variable.

**Model Evaluation**: Assess the performance of the trained model using the testing data. Common evaluation metrics for linear regression include mean squared error (MSE), root mean squared error (RMSE), and coefficient of determination (R-squared).

**Fine-tuning (Optional)**: If the model performance is not satisfactory, you can fine-tune hyperparameters or consider feature engineering to improve its accuracy.

**Prediction**: Once the model is trained and evaluated satisfactorily, you can use it to make predictions on new data by inputting the values of the independent variables into the model equation.

**Deployment**: Finally, if the model performs well on new data, it can be deployed into production for making real-world predictions.

#### What is the California Housing Dataset?

The California Housing Dataset is a widely used dataset in machine learning and statistics. It contains data related to housing in California, particularly focusing on the state's census districts. The dataset typically includes features such as median house value, median income, housing median age, average number of rooms, average number of bedrooms, population, and geographical information like latitude and longitude.

The main objective of using this dataset is often to build predictive models, such as regression models, to predict the median house value based on other attributes present in the dataset. It's commonly used for practicing and learning regression techniques, particularly in the context of supervised learning.

This dataset has been used in various research studies, educational settings, and competitions due to its relevance to real-world problems and its accessibility for educational purposes.
## Acknowledgements


 - ![Hands on machine learning - by Geron](https://github.com/vasanthgx/house_prices/blob/main/images/bookcover.jpg)
 - [github repo for handsonml-3](https://github.com/ageron/handson-ml3)
 - [EDA on the California housing dataset - kaggle notebook](https://www.kaggle.com/code/olanrewajurasheed/california-housing-dataset)
 


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at vasanth1627@gmail.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)