import pandas as pd

df = pd.read_csv('numerai_training_data.csv')
ab = pd.read_csv('numerai_tournament_data.csv')

full_length = len(df) #getting the full length of the trainingdata, this gave me 96320
seventy_percent_of_training_data = (70 * full_length) / 100
thirty_percent_of_training_data = (30 * full_length) / 100

#X - getting the 21 columns by 70percent of the rows
features_train = df[0:seventy_percent_of_training_data].iloc[:,1:21]

#Y - getting the target column by 70percent of the rows
labels_train = df['target'][0:seventy_percent_of_training_data]

#getting the 21 columns by the remaining rows
features_test = df[seventy_percent_of_training_data:full_length].iloc[:,1:21]

# getting the target column by the remaining rows
labels_test = df['target'][seventy_percent_of_training_data:full_length]

from sklearn.svm import SVC as svc
clf = svc().fit(features_train, labels_train)

#predicting our target value with the 30% remnant of the training_data
pred = clf.predict(features_test)
#len(pred) = 28896

#Accuracy Score
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred,labels_test) #0.520037375415
