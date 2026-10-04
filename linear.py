import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv(r"D:\projects\house prices model\hour.csv")
df["dteday"] = pd.to_datetime(df["dteday"])

# Extract date features
df["day"] = df["dteday"].dt.day
df["month"] = df["dteday"].dt.month
df["year"] = df["dteday"].dt.year

# Remove original date column
df = df.drop(columns=["dteday"])

# Create X and y AFTER preprocessing
X = df.drop(columns=["cnt", "casual", "registered"])
y = df["cnt"]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,)
#poly=PolynomialFeatures(degree=3)
#x_train_poly = poly.fit_transform(x_train)
#x_test_poly = poly.transform(x_test)
model=LinearRegression()
model.fit(x_train,y_train)
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

print("Training R² Score:", train_score)
print("Testing R² Score:", test_score)