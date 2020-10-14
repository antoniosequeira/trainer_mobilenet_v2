import os
import csv
import time
import warnings
import cv2
import pickle
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from keras.preprocessing import image
from keras.applications import mobilenet_v2
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler, Normalizer

#To remove warnings from the system
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Keras mobilenet only allows this sizes of images
#(224, 192, 160, 128, and 96).
WIDTH = 224
HEIGHT = WIDTH
batch_size = 12
epochs = '500'

test_class = 'trash'
path_model = 'TrainResult'
path_dataset = 'Real\{0}'.format(test_class)
classifier_model = 'batch_{0}_shape_{2}.h5'.format(batch_size, epochs, WIDTH)
file_name = 'batch_{1}_shape_{2}'.format(batch_size, WIDTH)

labels = ['cardboard','glass','metal','paper','plastic','trash']

lb = LabelBinarizer()

lista_imagens, x, y = [], [], []
path = 'Real'
for category in os.listdir(path):
    path_category = path + '/' + category
    folder = os.path.join(path_category)
    images = os.listdir(folder)
    print('{0} - {1}'.format(category, len(images)))
    for j in images:
        imagePath = os.path.join(folder + '/' + j)
        lista_imagens.append((imagePath, category))
print('Total: {0} imagens'.format(len(lista_imagens)))

start = time.time()

# loop over the input images
for imagePath, category in lista_imagens:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    image = img_to_array(image)
    x.append(image)
    y.append(category)

end = time.time()
print('Resized images in {0} seconds'.format(round(end-start,0)))

# transform multi-class labels to binary labels
y = np.array(y)
y = lb.fit_transform(y)

# Generate test dataset, this step is done just to format the train into tuples
X_real, X_test, Y_real, Y_test = train_test_split(x, y, test_size=0)

X_list = []
for i in X_real:
    X_list.append(i.reshape(-1))
X_real = np.array(X_list)
X_real.reshape(-1)

# Image Standardization
scaler = StandardScaler()
scaler.fit(X_real)
X_real = scaler.transform(X_real)

# Image Normalization
scaler = Normalizer()
scaler.fit(X_real)
X_real = scaler.transform(X_real)


X_list = []
for i in X_real:
    X_list.append(i.reshape((WIDTH,HEIGHT, 3)))
X_real = np.array(X_list)

real_backups = 'Real_Backups'

if not os.path.exists(real_backups):
    os.makedirs(real_backups)


# save One Hot Encoding
f = open('{0}/{1}_one_hot_encoding.txt'.format(real_backups, file_name), "wb") 
f.write(pickle.dumps(lb))
f.close()
print("Saved One Hot Encoding to disk")

# save X_real
f = open('{0}/{1}_X_train.txt'.format(real_backups, file_name), "wb") 
f.write(pickle.dumps(X_real))
f.close()
print("Saved X_real to disk")

# save Y_test
f = open('{0}/{1}_y_train.txt'.format(real_backups, file_name), "wb") 
f.write(pickle.dumps(Y_real))
f.close()
print("Saved Y_real to disk")

# Load One Hot Encoding
lb = pickle.loads(open('{0}/{1}_one_hot_encoding.txt'.format(real_backups, file_name), "rb").read())
print("Loaded One Hot Encoding from disk")

# Load X_real
X_real = pickle.loads(open('{0}/{1}_X_train.txt'.format(real_backups, file_name), "rb").read())
print("Loaded X_real from disk")

# Load Y_real 
Y_real = pickle.loads(open('{0}/{1}_y_train.txt'.format(real_backups, file_name), "rb").read())
print("Loaded Y_real from disk")

#Loads the model
loaded_classifier = load_model('{0}/{1}'.format(path_model, classifier_model))
print("Loaded model from disk")

#Predictions of the images
predictions = []
for i in loaded_classifier.predict(X_real):
    idx = np.argmax(i)
    predictions.append(lb.classes_[idx])

#Original labels of each image
true_labels = lb.inverse_transform(Y_real)

#Get the metrics based on the results
mylist = []
mylist.append('Accuracy: {0}'.format(accuracy_score(true_labels, predictions)))
mylist.append('Precision: {0}'.format(precision_score(true_labels, predictions, average='weighted')))
mylist.append('Recall: {0}'.format(recall_score(true_labels, predictions, average='weighted')))
mylist.append('F1: {0}'.format(f1_score(true_labels, predictions, average='weighted')))

print(confusion_matrix(true_labels, predictions))
print(mylist)


