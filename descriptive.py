import pandas as pd
from scipy.stats import skew, kurtosis, normaltest

df = pd.read_excel("Athletes.xlsx")
print(df.describe())

numeric_columns = ["Age", "First Half", "Second Half", "Finish", "Positive Split", "Percent Change"]

# Calculate skewness, kurtosis, and perform normality test for each selected column
for column in numeric_columns:
    skewness = skew(df[column])
    kurt = kurtosis(df[column])
    norm_test = normaltest(df[column])

    print(f"Column: {column}")
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurt}")
    print(f"Normality test p-value: {norm_test.pvalue}")
    if norm_test.pvalue < 0.05:
        print("The data is not normally distributed.")
    else:
        print("The data is normally distributed.")
    print("\n")
