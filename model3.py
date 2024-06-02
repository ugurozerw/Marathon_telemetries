import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Veri okuma
df = pd.read_excel("Athletes.xlsx")

# Kategorik değişkenlerin dönüştürülmesi
df['Age Group'] = df['Age Group'].astype('category').cat.codes
df['Gender'] = df['Gender'].astype('category').cat.codes

# Bağımsız (X) ve bağımlı (y) değişkenlerin tanımlanması
X = df.drop(columns=['Finish'])
y = df['Finish']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting modelinin eğitimi
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Tahminler
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Model performansının değerlendirilmesi
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Train R²: {r2_train:.3f}")
print(f"Test R²: {r2_test:.3f}")
print(f"Train RMSE: {rmse_train:.3f}")
print(f"Test RMSE: {rmse_test:.3f}")

# Model Katsayıları (Coefficients)
coefficients = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["Importance"]).sort_values(by="Importance", ascending=False)
print("Model Coefficients:")
print(coefficients)

# MDI (Mean Decrease In Impurity)
mdi_importance = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["Importance"]).sort_values(by="Importance", ascending=False)
print("Mean Decrease In Impurity (MDI):")
print(mdi_importance)

# Permutation Importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame(perm_importance.importances_mean, index=X.columns, columns=["Importance"]).sort_values(by="Importance", ascending=False)
print("Permutation Importance:")
print(perm_importance_df)

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP değerlerinin özet grafiği
shap.summary_plot(shap_values, X_test)

# LIME
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, mode='regression', feature_names=X_train.columns)
lime_exp = explainer.explain_instance(X_test.iloc[0].values, model.predict, num_features=len(X_test.columns))

# LIME açıklamasını çizme
lime_exp.as_pyplot_figure()

plt.show()
