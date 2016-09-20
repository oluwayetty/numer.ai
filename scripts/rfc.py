import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, log_loss

training_data = pd.read_csv('../datasets/numerai_training_data.csv')
tournament_data = pd.read_csv('../datasets/numerai_tournament_data.csv')

# splitting my arrays in ratio of 30:70 percent
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(training_data.iloc[:,0:21], training_data['target'], test_size=0.3, random_state=0)

# implementing my classifier
clf = RFC(n_estimators=25, random_state=0).fit(features_train, labels_train)
predictions = clf.predict(features_test)

# predict class probabilities
prob_predictions = clf.predict_proba(features_test)

#probability of being in a class 1
probability_class_of_one = np.array([x[1] for x in prob_predictions[:]]) # List comprehension

np.savetxt(
    '../numerai.csv',          # file name
    probability_class_of_one,  # array to savela
    fmt='%.2f',               # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character
    footer='end of file',   # file footer
    comments='# ',          # character to use for comments
    header='probability')   # file header

#Accuracy Score
accuracy = accuracy_score(labels_test, predictions, normalize=True,sample_weight=None)
print accuracy

logloss = log_loss(labels_test,prob_predictions)
print logloss
#write the predictions to a csv file with headers; `t_id, probability`
