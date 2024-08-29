# Market Capitalization Analysis and Prediction

This project analyzes market capitalization data from [CompaniesMarketCap.com](https://companiesmarketcap.com/) to gain insights into company valuation and predict future market caps.

## Dataset

The dataset consists of information about publicly traded companies, including:

- **Rank:** Ranking based on market cap
- **Name:** Company name
- **Symbol:** Stock symbol
- **marketcap:** Market capitalization in USD
- **price (USD):** Current stock price in USD
- **country:** Company headquarters location

**##Requirements**
To run the notebook, you need to have the following Python packages installed:

-**Pandas**
-**NumPy
-Matplotlib
-Seaborn
-Scikit-learn
-Jupyter Notebook or Jupyter Lab**

## Analysis

The Jupyter Notebook (`Market_Cap_Analysis.ipynb`) explores the dataset with various data analysis techniques and visualizations:

**Exploratory Data Analysis (EDA):**
- Data cleaning (handling missing values)
- Descriptive statistics for key features
- Distribution of market cap and stock price (histograms with logarithmic scales)
- Top countries by the number of companies (bar plot)
- Relationships between:
    - Market cap vs. rank (scatter plot with log scale)
    - Price vs. market cap (scatter plot with log scales for both axes)

**In-depth Country Analyses:**
- Detailed exploration of Indian, American, and Chinese companies
- Descriptive statistics, market cap distributions (histograms), and top companies by market cap (bar plots and pie charts)

**Correlation Matrices:**
- Examination of correlations between numerical features within Indian, American, and Chinese company subsets using heatmaps

## Model Building and Prediction

The project evaluates three different regression models:

- **Random Forest Regressor:** 
    - Handles non-linear relationships well and performs feature selection intrinsically.
    - May not provide as much interpretability as linear models.
- **Linear Regression:** 
    - Simpler to understand and provides coefficients indicating feature impact on market cap.
    - Assumes linear relationships and can be affected by outliers.
- **Support Vector Regression:** 
    - Effective for non-linear relationships, can be computationally demanding for large datasets.
    - Hyperparameters require tuning and may not always outperform other models.

The code performs the following:

- One-hot encoding for categorical features (specifically 'country')
- Data splitting into training (80%) and testing (20%) sets
- Standard scaling to normalize features before training linear models
- Model fitting using `sklearn` library
- Prediction on the testing set
- Metric calculations: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared

## Evaluation

- The code displays evaluation metrics for all three models.
- Visualizations include:
    - Scatter plot of actual vs. predicted market cap for Linear Regression
    - Histogram of predicted probabilities for Logistic Regression

**Best Performing Model:**
The evaluation results will help you identify the best-performing model. Further research and analysis can focus on improving its accuracy by experimenting with hyperparameter tuning, adding more relevant features, or exploring other advanced machine learning techniques.

## Getting Started

1. Clone this repository
2. Install necessary Python libraries: `pandas`, `scikit-learn`, `seaborn`, `matplotlib`
3. Open and run the Jupyter Notebook `Market_Cap_Analysis.ipynb`

## License

This project is licensed under the MIT License.
content_copy
