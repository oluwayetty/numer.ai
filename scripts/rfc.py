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
acc_pred = clf.predict(features_test)
# print acc_pred
pred = clf.predict_proba(features_test)
#probability of being in a class 1
y = np.array([x[1] for x in pred[:]]) # List comprehension

print y
# print len(pred)

np.savetxt(
    '../numerai.csv',          # file name
    y,                      # array to savela
    fmt='%f',               # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character
    footer='end of file',   # file footer
    comments='# ',          # character to use for comments
    header='probability')   # file header

#Accuracy Score
acc = accuracy_score(labels_test, acc_pred, normalize=True,sample_weight=None)
print acc

logloss = log_loss(labels_test,pred)
print logloss
#write the predictions to a csv file with headers; `t_id, probability`
