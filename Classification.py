# Decision Tree Classifier
import numpy as np
from sklearn import datasets
from sklearn import tree
# load the iris datasets
dataset = datasets.load_iris()
#print(dataset.feature_names)
#print(dataset.target_names)
#Keeping 3 record a side for testing purpose
#index 0 indicates the first record of Setosa, Index 50 for Versicolor and 100 for virginica
testing = [0,50,100]

#deleting records that kept for testing
train_target = np.delete(dataset.target,testing)
train_data = np.delete(dataset.data,testing, axis=0)

#storing 3 data for testing
test_target = dataset.target[testing]
test_data = dataset.data[testing]

# fit a CART model to the Training data
model = tree.DecisionTreeClassifier()
model.fit(train_data, train_target)

#checking the data we stored for testing
print(test_target)

#testing if the model can recognize this set of data it has never seen
print(model.predict(test_data))