import os
import csv
import pickle
import warnings
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

#To remove warnings from the system
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 12
epochs = 500
WIDTH = 224

file_name = 'batch_{1}_shape_{2}'.format(batch_size, WIDTH)

pathTrainResult = 'TrainResult'
classifier_model = '{0}.h5'.format(file_name)

loaded_classifier = load_model('{0}/{1}'.format(pathTrainResult, classifier_model))
print("Loaded model from disk")

# Load One Hot Encoding
lb = pickle.loads(open('{0}/{1}_one_hot_encoding.txt'.format(pathTrainResult, file_name), "rb").read())
print("Loaded One Hot Encoding from disk")

# Load X_tes
X_test = pickle.loads(open('{0}/{1}_X_test.txt'.format(pathTrainResult, file_name), "rb").read())
print("Loaded X_test from disk")

# Load y_test 
y_test = pickle.loads(open('{0}/{1}_y_test.txt'.format(pathTrainResult, file_name), "rb").read())
print("Loaded y_test from disk")

#Predictions of the images
predictions = []
for i in loaded_classifier.predict(X_test):
    idx = np.argmax(i)
    predictions.append(lb.classes_[idx])

#Original labels of each image
true_labels = lb.inverse_transform(y_test)

#Get the metrics based on the results
mylist = []
mylist.append('Accuracy: {0}'.format(accuracy_score(true_labels, predictions)))
mylist.append('Precision: {0}'.format(precision_score(true_labels, predictions, average='weighted')))
mylist.append('Recall: {0}'.format(recall_score(true_labels, predictions, average='weighted')))
mylist.append('F1: {0}'.format(f1_score(true_labels, predictions, average='weighted')))

print(confusion_matrix(true_labels, predictions))
print(mylist)