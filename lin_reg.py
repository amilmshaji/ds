import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.show()

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error((y_test,y_pred)
rmse=np.sqrt(mse)

