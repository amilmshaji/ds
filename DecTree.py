import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'], random_state=42,test_size=0.1)
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
y_pred=clf.predict(X_test)
from sklearn import metrics, tree

print("Decision tree model accuracy(in %):", metrics.accuracy_score(Y_test, y_pred)*100)

# plot the decision tree
plt.figure(figsize=[10,10])
tree.plot_tree(clf, filled=True)
plt.show()