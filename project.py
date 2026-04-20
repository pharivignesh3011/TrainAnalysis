change

# ==============================
# DATA SCIENCE PROJECT (FINAL FIXED)
# ==============================

# STEP 1 — Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# STEP 2 — Load Dataset
data = pd.read_csv("cleaned_train.csv")

print("Dataset Loaded Successfully")

print("\nShape of Dataset:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data.head())


# STEP 3 — Convert to Numeric (SAFE)
data_numeric = data.apply(pd.to_numeric, errors='coerce')

# Keep only useful numeric columns (not empty)
numeric_data = data_numeric.dropna(axis=1, how='all')

# Remove columns with too many NaNs
numeric_data = numeric_data.dropna(axis=1, thresh=len(numeric_data)*0.5)

print("\nFinal Numeric Columns:")
print(numeric_data.columns)


# ==============================
# VISUALIZATION (FIXED)
# ==============================

# HISTOGRAMS
for col in numeric_data.columns:
    plt.figure()
    sns.histplot(numeric_data[col].dropna())
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


# SCATTER PLOTS (ONLY VALID PAIRS)
cols = numeric_data.columns

for i in range(len(cols)-1):
    plt.figure()
    sns.scatterplot(
        x=numeric_data[cols[i]],
        y=numeric_data[cols[i+1]]
    )
    plt.title(f"{cols[i]} vs {cols[i+1]}")
    plt.xlabel(cols[i])
    plt.ylabel(cols[i+1])
    plt.show()


# BOX PLOTS
for col in numeric_data.columns:
    plt.figure()
    sns.boxplot(y=numeric_data[col])
    plt.title(f"Box Plot of {col}")
    plt.show()


# ==============================
# HEATMAP
# ==============================

if numeric_data.shape[1] > 1:
    plt.figure(figsize=(8,6))
    sns.heatmap(numeric_data.corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.show()


# ==============================
# PAIRPLOT
# ==============================

if numeric_data.shape[1] > 1:
    sns.pairplot(numeric_data.dropna())
    plt.show()


# ==============================
# LINEAR REGRESSION (FIXED)
# ==============================

if numeric_data.shape[1] >= 2:

    # Choose two best numeric columns automatically
    X = numeric_data.iloc[:, [0]].dropna()
    Y = numeric_data.iloc[:, 1].dropna()

    # Align rows
    common_index = X.index.intersection(Y.index)
    X = X.loc[common_index]
    Y = Y.loc[common_index]

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Train
    model = LinearRegression()
    model.fit(X_train, Y_train)

    print("\nModel Trained Successfully")

    # Predict
    Y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print("\nModel Evaluation:")
    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)

    # Plot regression line
    plt.figure()
    plt.scatter(X_test, Y_test)
    plt.plot(X_test, Y_pred)
    plt.title("Linear Regression Line")
    plt.xlabel(X.columns[0])
    plt.ylabel("Target")
    plt.show()

else:
    print("\nNot enough numeric data for regression")


print("\nProject Completed Successfully")
