import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

df = pd.read_excel("Athletes.xlsx")


df['Age Group'] = df['Age Group'].astype('category').cat.codes
df['Gender'] = df['Gender'].astype('category').cat.codes

# Define independent (X) and dependent (y) variables
X = df.drop(columns=['Finish'])
y = df['Finish']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate model performance
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Train R²: {r2_train:.3f}")
print(f"Test R²: {r2_test:.3f}")
print(f"Train RMSE: {rmse_train:.3f}")
print(f"Test RMSE: {rmse_test:.3f}")

# Feature importance using Model Coefficients (for linear models)
if model.booster == 'gblinear':
    coefficients = model.coef_
    feature_importance_linear = pd.Series(coefficients, index=X.columns).sort_values(ascending=False)
    print("Feature importance using Model Coefficients:")
    print(feature_importance_linear)

# Feature importance using Mean Decrease in Impurity (MDI)
feature_importance_mdi = model.feature_importances_
mdi_importance = pd.Series(feature_importance_mdi, index=X.columns).sort_values(ascending=False)
print("Feature importance using Mean Decrease in Impurity (MDI):")
print(mdi_importance)

# Feature importance using Permutation Importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame(perm_importance.importances_mean, index=X.columns, columns=["Importance"]).sort_values(by="Importance", ascending=False)
print("Feature importance using Permutation Importance:")
print(perm_importance_df)

# Feature importance using SHAP (Shapley Additive Explanations)
explainer = shap.Explainer(model)
shap_values = explainer(X)
shap.summary_plot(shap_values, X, plot_type="bar")

# Feature importance using LIME (Local Interpretable Model-agnostic Explanations)
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=['Finish'], verbose=True, mode='regression')
i = 0  # Index of the instance to explain
lime_exp = lime_explainer.explain_instance(X_test.values[i], model.predict, num_features=len(X.columns))
lime_exp.show_in_notebook(show_table=True)

# Plotting MDI and Permutation Importance
plt.figure(figsize=(12, 6))

# Plot Mean Decrease in Impurity (MDI)
plt.subplot(1, 2, 1)
mdi_importance.plot(kind='bar', title='MDI Feature Importance')
plt.ylabel('Mean Decrease in Impurity')

# Plot Permutation Importance
plt.subplot(1, 2, 2)
perm_importance_df.plot(kind='bar', title='Permutation Feature Importance')
plt.ylabel('Importance')

plt.tight_layout()
plt.show()
