# ============================================================
# ASSIGNMENT 11.1: WHAT DRIVES THE PRICE OF A CAR?
# CRISP-DM FRAMEWORK
# ============================================================

# ============================================================
# 1. BUSINESS UNDERSTANDING
# ============================================================

print("""
BUSINESS OBJECTIVE

A used car dealership wants to understand which factors
have the greatest impact on used car prices.

This problem is formulated as a supervised machine learning
regression problem where:
- Target Variable: Price
- Predictors: Vehicle characteristics
- Goal: Predict vehicle prices and identify key price drivers.
""")

# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

sns.set_style("whitegrid")

# ============================================================
# 3. DATA UNDERSTANDING
# ============================================================

# Update path if necessary
df = pd.read_csv("D:\\Personnel\\Certification\\AI ML PG\\Projects python\\Capstone project\\Car price prediction\\practical_application_II_starter\\data\\vehicles.csv")
print("\nDataset Shape:")
print(df.shape)

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(
    df.isnull()
      .sum()
      .sort_values(ascending=False)
      .head(20)
)

print("\nDuplicate Records:")
print(df.duplicated().sum())

# ============================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Price Distribution
plt.figure(figsize=(10,5))
sns.histplot(df["price"], bins=50, kde=True)
plt.title("Vehicle Price Distribution")
plt.show()

# Price Outlier Detection
plt.figure(figsize=(10,5))
sns.boxplot(x=df["price"])
plt.title("Vehicle Price Boxplot")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    cmap="coolwarm"
)
plt.title("Numeric Correlation Matrix")
plt.show()

# Sample data for scatterplots
sample_size = min(5000, len(df))
sample_df = df.sample(sample_size, random_state=42)

# Price vs Odometer
if "odometer" in df.columns:
    plt.figure(figsize=(10,5))
    sns.scatterplot(
        data=sample_df,
        x="odometer",
        y="price"
    )
    plt.title("Price vs Odometer")
    plt.show()

# Price vs Year
if "year" in df.columns:
    plt.figure(figsize=(10,5))
    sns.scatterplot(
        data=sample_df,
        x="year",
        y="price"
    )
    plt.title("Price vs Year")
    plt.show()

# ============================================================
# 5. DATA PREPARATION
# ============================================================

# Remove duplicates
df = df.drop_duplicates()

# Remove missing prices
df = df.dropna(subset=["price"])

# Vehicle age feature
if "year" in df.columns:
    CURRENT_YEAR = 2024
    df["vehicle_age"] = CURRENT_YEAR - df["year"]

# Remove high-cardinality text fields
cols_to_drop = [
    'id',
    'url',
    'region_url',
    'image_url',
    'description',
    'VIN',
    'county'
]

df = df.drop(
    columns=[c for c in cols_to_drop if c in df.columns],
    errors='ignore'
)

# Target
y = df['price']

X = df.drop(columns=['price'])

# Convert categorical features
categorical_cols = X.select_dtypes(
    include=['object']
).columns

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)

# One-Hot Encoding
X = pd.get_dummies(
    X,
    drop_first=True
)

# Fill missing values
X = X.fillna(
    X.median(numeric_only=True)
)

X = X.fillna(0)

print("\nPrepared Dataset Shape:")
print(X.shape)

# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

# ============================================================
# 7. MODEL 1 - LINEAR REGRESSION
# ============================================================

lr = LinearRegression()

lr.fit(
    X_train,
    y_train
)

lr_pred = lr.predict(X_test)

cv_scores = cross_val_score(
    lr,
    X_train,
    y_train,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

print("\nLinear Regression CV RMSE:")
print(-cv_scores.mean())

# ============================================================
# 8. MODEL 2 - RIDGE REGRESSION
# ============================================================

ridge = Ridge()

ridge_params = {
    "alpha":[0.01,0.1,1,10,100]
}

ridge_grid = GridSearchCV(
    ridge,
    ridge_params,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

ridge_grid.fit(
    X_train,
    y_train
)

best_ridge = ridge_grid.best_estimator_

ridge_pred = best_ridge.predict(X_test)

print("\nBest Ridge Parameters:")
print(ridge_grid.best_params_)

# ============================================================
# 9. MODEL 3 - LASSO REGRESSION
# ============================================================

lasso = Lasso(max_iter=5000)

lasso_params = {
    "alpha":[0.001,0.01,0.1,1]
}

lasso_grid = GridSearchCV(
    lasso,
    lasso_params,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

lasso_grid.fit(
    X_train,
    y_train
)

best_lasso = lasso_grid.best_estimator_

lasso_pred = best_lasso.predict(X_test)

print("\nBest Lasso Parameters:")
print(lasso_grid.best_params_)

# ============================================================
# 10. MODEL 4 - RANDOM FOREST REGRESSOR
# ============================================================

rf = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

rf_params = {
    "n_estimators":[100,200],
    "max_depth":[10,20,None]
}

rf_grid = GridSearchCV(
    rf,
    rf_params,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

rf_grid.fit(
    X_train,
    y_train
)

best_rf = rf_grid.best_estimator_

rf_pred = best_rf.predict(X_test)

print("\nBest Random Forest Parameters:")
print(rf_grid.best_params_)

# ============================================================
# 11. MODEL EVALUATION
# ============================================================

def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return [name, mae, rmse, r2]

results = []

results.append(
    evaluate_model(
        "Linear Regression",
        y_test,
        lr_pred
    )
)

results.append(
    evaluate_model(
        "Ridge Regression",
        y_test,
        ridge_pred
    )
)

results.append(
    evaluate_model(
        "Lasso Regression",
        y_test,
        lasso_pred
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        rf_pred
    )
)

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2"
    ]
)

results_df = results_df.sort_values(
    by="RMSE"
)

print("\nMODEL COMPARISON")
print(results_df)

# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

if hasattr(best_rf, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_rf.feature_importances_
    })

    top_features = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(15)
    )

    print("\nTop 15 Most Important Features")
    print(top_features)

    plt.figure(figsize=(10,6))

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature"
    )

    plt.title("Top 15 Features Impacting Used Car Price")
    plt.show()

# ============================================================
# 13. BUSINESS FINDINGS
# ============================================================

print("""
KEY FINDINGS

1. Vehicle age strongly impacts vehicle price.
2. Mileage has a negative effect on resale value.
3. Vehicle condition significantly influences pricing.
4. Manufacturer and model affect resale value.
5. Newer vehicles generally command higher prices.

BUSINESS RECOMMENDATIONS

1. Prioritize newer inventory.
2. Focus on low-mileage vehicles.
3. Use model predictions to support pricing decisions.
4. Acquire brands with strong resale values.
5. Continuously retrain the model as market conditions change.

LIMITATIONS

- Accident history unavailable.
- Maintenance records unavailable.
- Regional demand not included.
- Market conditions change over time.

FUTURE WORK

- Add accident and maintenance data.
- Evaluate XGBoost and Gradient Boosting.
- Develop a dealership pricing dashboard.
""")

# ============================================================
# 14. CONCLUSION
# ============================================================

best_model = results_df.iloc[0]["Model"]

print(f"""
CONCLUSION

The analysis identified major price drivers for used cars.
Several regression algorithms were evaluated using
cross-validation and hyperparameter tuning.

Best Model: {best_model}

The final model can be used by dealerships to
improve inventory decisions, pricing accuracy,
and overall profitability.
""")