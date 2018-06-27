import numpy
from sklearn.datasets import load_iris
#loading iris data set
iris=load_iris()
#to print
print(iris.feature_names)
print(iris.target_names)
#training data
#features data
print(iris.data)
#target data means flowers data
print(iris.target)
#now splitting into test and train data sets 
from sklearn.model_selection import train_test_split
x,y,z,a=train_test_split(iris.data,iris.target,test_size=0.1)
'''
here
x is train_iris with full features value containing 90% data
y is remaining test_iris
z is train_target {90% of iris.target data}
a is test_target {10% if iris.target data}''' 

#calling algo

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
clf=tree.DecisionTreeClassifier()
#now trianing data with decision
trained=clf.fit(x,z)
#now time for prediction
output=trained.predict(y)
print(output)
#checking % of accuracy
from sklearn.metrics import accuracy_score
check_pct=accuracy_score(a,output)
print(check_pct)
#exporting graph for decision tree 
from sklearn.tree import export_graphviz
tree.export_graphviz(clf, out_file="tree.dot", feature_names=iris.feature_names, filled=True,  rounded=True, special_characters=True, precision=3)

