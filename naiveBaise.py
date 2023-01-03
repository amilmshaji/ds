import sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()
a=iris.data
b=iris.target

x_train,x_test,y_train,y_test=train_test_split(a,b,test_size=0.3,random_state=42)

model=GaussianNB()
model.fit(x_train,y_train)

predictions=model.predict(x_test)
print(predictions)

acc=accuracy_score(y_test,predictions)
print(acc)

