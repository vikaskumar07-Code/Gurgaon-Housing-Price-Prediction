import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
# 1. Load the data
housing = pd.read_csv("housing.csv")

# 2. Create a stratified test set based on income category
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

# Work on a copy of training data
housing = strat_train_set.copy()

# 3. Separate predictors and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# 4. Separate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# 5. Pipelines
# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical pipeline
cat_pipeline = Pipeline([
    # ("ordinal", OrdinalEncoder())  # Use this if you prefer ordinal encoding
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)

# housing_prepared is now a NumPy array ready for training
print(housing_prepared.shape)

#Linear Regression
lin_reg=LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)

#Decision Tree
tree_reg=DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)

# Random Forest
forest_Reg = RandomForestRegressor()
forest_Reg.fit(housing_prepared,housing_labels)

# Predict Using Data
lin_preds = lin_reg.predict(housing_prepared)
tree_preds = lin_reg.predict(housing_prepared)
forest_preds = lin_reg.predict(housing_prepared)

#Calculate RMSE
lin_RMSE = root_mean_squared_error(housing_labels,lin_preds)
tree_RMSE = root_mean_squared_error(housing_labels,tree_preds)
forest_RMSE = root_mean_squared_error(housing_labels,forest_preds)

tree_RMSE = cross_val_score(tree_reg,housing_prepared,housing_labels,cv=10,scoring="neg_root_mean_squared_error")   
print(pd.Series(tree_RMSE).describe())
