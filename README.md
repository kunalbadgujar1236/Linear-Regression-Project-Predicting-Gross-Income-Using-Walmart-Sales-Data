# 📊 Predicting Gross Income Using Linear Regression

## 📖 Project Overview
This project applies **Linear Regression** to predict **Gross Income** based on various factors like Total Sales, Customer Type, Payment Method, and other sales attributes.

## 🔍 Problem Statement
We aim to predict **Gross Income** using **Total Sales and Other Features**. This helps businesses understand revenue drivers and make better financial decisions.

---

## 📌 Step 1: Understanding Linear Regression

### 🔹 What is Linear Regression?
Linear Regression is a statistical technique used to **find a relationship between independent variables (X) and a dependent variable (Y)** by fitting a straight line.

### 🔹 Formula for Simple Linear Regression:
\[
Y = mX + b
\]
Where:
- **Y** = Predicted Value (Gross Income)
- **X** = Independent Variable (Total Sales, etc.)
- **m** = Slope (Rate of change)
- **b** = Intercept (Starting point when X = 0)

For **Multiple Linear Regression**, the formula extends to:
\[
Y = b_0 + b_1X_1 + b_2X_2 + ... + b_nX_n
\]

---

## 📌 Step 2: Dataset Description

The dataset contains **sales transaction records** with the following features:

| Column | Description |
|--------|------------|
| Invoice ID | Unique sales identifier |
| Branch | Store location branch |
| City | Store city |
| Customer Type | Membership or walk-in |
| Gender | Male/Female |
| Product Line | Type of product sold |
| Unit Price | Price per unit |
| Quantity | Number of items purchased |
| Tax 5% | Tax applied to total sales |
| Total | Total cost including tax |
| Date | Purchase date |
| Time | Purchase time |
| Payment | Payment method (Cash, Credit, etc.) |
| COGS | Cost of goods sold |
| Gross Margin Percentage | Percentage margin |
| Gross Income | **Target Variable (Y) - What we want to predict** |
| Rating | Customer rating for transaction |

---

## 📌 Step 3: Steps to Prepare the Dataset

1️⃣ **Load the dataset**  
2️⃣ **Select relevant features**  
3️⃣ **Convert categorical variables** into numeric using One-Hot Encoding  
4️⃣ **Scale numerical features** using StandardScaler  
5️⃣ **Split the dataset** into **Training (80%)** and **Testing (20%)**  
6️⃣ **Train the Linear Regression model**  
7️⃣ **Make predictions** on test data  
8️⃣ **Evaluate performance** using **MAE, MSE, and R² Score**  
9️⃣ **Analyze residuals** to check model accuracy  
🔟 **Interpret feature importance** from model coefficients  

---

## 📌 Step 4: Implementation (Python Code)

```python
# 📌 Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 📌 Step 2: Load the Dataset
df = pd.read_csv("your_dataset.csv")  # Replace with actual dataset

# 📌 Step 3: Select Relevant Features
df = df[['Total', 'Customer type', 'Payment', 'Rating', 'Unit price', 'Quantity', 'Branch', 'gross income']]

# 📌 Step 4: Convert Categorical Variables to Numeric
df = pd.get_dummies(df, columns=['Customer type', 'Payment', 'Branch'], drop_first=True)

# 📌 Step 5: Define Features (X) and Target Variable (y)
X = df.drop(columns=['gross income'])  # Independent Variables
y = df['gross income']  # Dependent Variable

# 📌 Step 6: Scale the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 📌 Step 7: Split the Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 📌 Step 8: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 📌 Step 9: Make Predictions
y_pred = model.predict(X_test)

# 📌 Step 10: Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# 📌 Step 11: Display Feature Importance
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\nFeature Importance in Linear Regression:")
print(coefficients.sort_values(by="Coefficient", ascending=False))
