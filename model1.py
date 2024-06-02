import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor #Random forest kullandık.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel("Athletes.xlsx")

df['Age Group'] = df['Age Group'].astype('category').cat.codes
df['Gender'] = df['Gender'].astype('category').cat.codes

# Bağımsız ve bağımlı değişkenleri belirleme
X = df.drop(columns=['Finish'])
y = df['Finish']

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modeli oluşturma
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Modelin tahmin performansı
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 (Determinasyon Katsayısı): {r2}")

# 1. Model Feature Importance (Model Katsayıları)
feature_importance = model.feature_importances_
feature_names = X.columns
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)
print("Model Feature Importance (Model Katsayıları):")
print(fi_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=fi_df)
plt.title('Model Feature Importance')
plt.show()

# 2. Mean Decrease in Impurity (MDI)
mdi_importance = model.feature_importances_
mdi_df = pd.DataFrame({'Feature': feature_names, 'MDI Importance': mdi_importance}).sort_values(by='MDI Importance', ascending=False)
print("Mean Decrease in Impurity (MDI):")
print(mdi_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='MDI Importance', y='Feature', data=mdi_df)
plt.title('Mean Decrease in Impurity (MDI)')
plt.show()

# 3. Permutation Importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({'Feature': feature_names, 'Permutation Importance': perm_importance.importances_mean}).sort_values(by='Permutation Importance', ascending=False)
print("Permutation Importance:")
print(perm_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Permutation Importance', y='Feature', data=perm_df)
plt.title('Permutation Importance')
plt.show()

# 4. SHAP (Shapley Additive Explanations)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# 5. LIME
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=feature_names, class_names=['Finish'], verbose=True, mode='regression')
i = np.random.randint(0, X_test.shape[0])
exp = lime_explainer.explain_instance(X_test.values[i], model.predict, num_features=5)
exp.show_in_notebook(show_table=True)
