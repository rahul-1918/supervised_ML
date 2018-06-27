from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# features about apple-0 and orange 1
data=[[100,0],[130,0],[135,1],[150,1]]
output=["apple","apple","orange","orange"]
# decision tree algo call
algo=tree.DecisionTreeClassifier()
# train data
trained_algo=algo.fit(data,output)
#testing phase
predict=trained_algo.predict([[130,1]])
#printing output
print(predict)  
