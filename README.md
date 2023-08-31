# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KOWSALYA M
RegisterNumber:  212222230069

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()
X =df.iloc[:,:1].values
X
Y=df.iloc[:,-1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test

plt.scatter(X_train,Y_train,color="yellow")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours VS Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="yellow")
plt.plot(X_test,regressor.predict(X_test),color="pink")
plt.title("Hours VS Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```
## Output:
![m1](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/03ba3bed-5253-43af-b766-8af0e6862129)

![m2](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/d761c825-2ae4-4ea0-a102-d46e0ebc9df2)

![m3](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/edaa8ac4-ef1c-4b4a-9b32-638085b7c822)

![m4](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/59fd464f-6f9e-416b-875a-76671f7f6bcf)

![m5](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/7c328dfe-e3a6-4912-b3d3-a540bd705e6f)

![m6](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/73b51c7d-155a-4c1c-b28c-aa92f0284c00)

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
