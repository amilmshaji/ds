from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

dataset=pd.read_csv('Iris_Data.csv')
x=dataset.iloc[:,:4].values
y=dataset.iloc[:,4].values
print("The data values",x)
print("The target values",y)

#splitting the dataset

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#initialling the model
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_predict=knn.predict(x_test)

print("\n")
print("The test data result",y_predict)
print("\n")
print("The actual output",y_test)
print("\n")


acc=accuracy_score(y_predict,y_test)
print("The accuracy of the algorothm",acc)