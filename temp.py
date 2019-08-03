import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Salary_Data.csv")

X=dataset.iloc[:,[0]].values
y=dataset.Salary.values
#print(y)
#print(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(y_test)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
print(y_pred)











