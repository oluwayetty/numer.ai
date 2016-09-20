import pandas as pd
from sklearn import cross_validation
from sklearn.svm import SVC as svc
from sklearn.metrics import accuracy_score

training_data = pd.read_csv('../datasets/numerai_training_data.csv')
tournament_data = pd.read_csv('../datasets/numerai_tournament_data.csv')

#this returns four arrays which is in the order of features_train, features_test, labels_train, labels_test
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(training_data.iloc[:,0:21], training_data['target'], test_size=0.3, random_state=0)

clf = svc(C=1.0).fit(features_train, labels_train)

#predicting our target value with the 30% remnant of the training_data
predictions = clf.predict(features_test)
print predictions

accuracy = accuracy_score(predictions,labels_test)
print accuracy
#c = 1.0 -> 0.514361849391
#c = 100.0 -> 0.518133997785
