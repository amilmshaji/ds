# import necessary libraries
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# load the iris dataset as an example
iris = datasets.load_iris()
X = iris.data
y = iris.target

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.1)

# train the decision tree model
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# make predictions on the testing data
predictions = model.predict(X_test)

# calculate the accuracy of the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# plot the decision tree
plt.figure(figsize=[10,10])
tree.plot_tree(model, filled=True)
plt.show()