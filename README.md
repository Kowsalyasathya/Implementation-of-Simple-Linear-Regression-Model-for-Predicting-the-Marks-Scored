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
```Program to implement the simple linear regression model for predicting the marks scored.
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
print('RMSE = ',rmse
```
## Output:
## df.head():
![EX 02 (1)](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/9c467cb5-90a8-49fc-bdb9-92d32d08ee5a)
### df.tail():
![EX 02 (2)](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/04e557ae-60ff-438f-bd49-f6429072a357)
### Array values of X:
![EX 02 (3)](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/48633c27-fd92-4e96-acd0-4edd51083c76)
### Array values of Y:
![EX 02 (4)](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/5980e9b9-4029-48b3-8bed-4227d5c23434)
### Values of prediction:
![EX 02 (5)](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/51d2232c-616a-4f3f-a30e-8e16deaa077f)
### Array values of Y test:
![EX 02 (6)](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/827370af-56b4-4e0f-b97a-d31be2112f5a)
### Training set graph:
![EX 02 (7)](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/54c869bd-b9bc-4e42-ab79-55a4f2137dc4)
### Testing set graph:
![EX 02 (8)](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/1a4cdbfa-99e8-449e-99e5-2b6aa337404c)
### Values of MSE,MAE,RMSE:
![EX 02 (9)](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/61e8a540-1c13-459e-b909-c1219b95453d)

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
