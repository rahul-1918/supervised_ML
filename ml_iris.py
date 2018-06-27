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
setosa=iris.data[0:50]
print(setosa)
s_data=iris.target[0:50]
print(s_data)
x=[0,50,100]
#training target
only_target_training=numpy.delete(iris.target,x,axis=0)
only_data_training=numpy.delete(iris.data,x,axis=0)
#target data value
print(only_target_training)
#flower data feature 
print(only_data_training)
print(only_target_training.size)
#testing target
test_target=iris.target[x]
test_data=iris.data[x]
print(test_target)
print(test_data)

#calling algo
clf=tree.DecisionTreeClassifier()
trained=clf.fit(only_data_training,only_target_training)
output=trained.predict(test_data)
print(output)
