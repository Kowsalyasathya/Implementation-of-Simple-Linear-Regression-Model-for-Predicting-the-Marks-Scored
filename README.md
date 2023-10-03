![image](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/2dbd9502-57e5-4dba-a628-9bd8fd465685)# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

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
```
```
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
## df.head() "
![3 1](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/5214d5e0-301d-4323-bea8-9fc23f6346d7)
### df.tail() :
![3 2](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/3553c266-e0cf-4a56-bdac-d75a1f40abdd)
### Array values of X:
![3 3](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/031e1e6d-2a02-45cf-a974-d0fe5f648121)
### Array values of Y:
![3 4](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/e56b869b-b006-48c3-9494-3c549f37382e)
### Values of prediction:
![3 5](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/8e98a514-1e9c-414e-8c92-42dd8eb1f6fb)
### Testing set graph:
![3 7](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/428d8e3c-d948-4798-99db-85192a9b3f89)
### Training set graph:
![3 6](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/a520ed70-1f26-4aed-a32f-6e1a12c626c9)

### Values of MSE,MAE,RMSE:
![3 8](https://github.com/Kowsalyasathya/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118671457/6bff2ece-e83b-4137-8466-720d0542685f)



## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
