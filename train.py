# Head Notes
# Tensorflow doesn't like numpy 1.17 and gives a lot of warnings, to remove them use the following command:
# pip install "numpy<1.17"

# Python API
import os
import random
import time
import warnings
# 3rd party API
import pickle
import numpy as np
import cv2
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler, Normalizer
from keras import Model
from keras.layers.core import Dense
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.utils import plot_model
from matplotlib import pyplot as plt

#To remove warnings from the system
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Variables
path = 'Trashnet'
pathTrainResult = 'TrainResult'
batch_size = 12
epochs = 500
WIDTH = 192 # (224, 192, 160, 128, and 96).
HEIGHT = WIDTH
lista_imagens, x, y = [], [], []
lb = LabelBinarizer()


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

# Generate test dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# Generate validation train and validation dataset
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

x_l = []
for i in X_train:
    x_l.append(i.reshape(-1))
X_train = np.array(x_l)
X_train.reshape(-1)

x_l = []
for i in X_val:
    x_l.append(i.reshape(-1))
X_val = np.array(x_l)

x_l = []
for i in X_test:
    x_l.append(i.reshape(-1))
X_test = np.array(x_l)


# Image Standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
print('Standardized images')

# Image Normalization
scaler = Normalizer()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
print('Normalized Images')

X_trein = []
for i in X_train:
    X_trein.append(i.reshape((WIDTH,HEIGHT, 3)))
X_train = np.array(X_trein)

X_vali = []
for i in X_val:
    X_vali.append(i.reshape((WIDTH,HEIGHT, 3)))
X_val = np.array(X_vali)

X_teste = []
for i in X_test:
    X_teste.append(i.reshape((WIDTH,HEIGHT, 3)))
X_test = np.array(X_teste)


print('Defining classifier')
mobilenet = MobileNetV2(input_shape=(WIDTH, HEIGHT, 3), include_top=False, weights='imagenet')

x = mobilenet.output
x = GlobalAveragePooling2D()(x)

x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3

predictions = Dense(6, activation='softmax')(x)
classifier = Model(inputs= mobilenet.input, outputs=predictions)
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.summary()
print('Finished defining classifier')

if not os.path.exists(pathTrainResult):
    os.makedirs(pathTrainResult)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")


file_name = 'batch_{0}_shape_{1}'.format(batch_size, WIDTH)

# to save best model
bestcheckpoint = ModelCheckpoint(pathTrainResult + '/batch_'+ str(batch_size) +'_epochs_'+ str(epochs) +'_shape_'+ str(WIDTH) +'.h5', save_best_only=True, monitor='val_loss', mode='min')
callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, mode='auto')
csv_logger = CSVLogger('PlotResults/batch_'+ str(batch_size) +'_epochs_'+ str(epochs) +'_shape_'+ str(WIDTH) +'_training.log')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
steps = int( np.ceil(X_train.shape[0] / batch_size) )

# fit() should be used for small datasets, loads everything into memory
# fit_generator() should be used for larger datasets, which loads into memory only small batches of data.
H = classifier.fit_generator(aug.flow(X_train, y_train, batch_size=batch_size), validation_data = (X_val, y_val),steps_per_epoch=steps, epochs = epochs, verbose = 1, callbacks=[bestcheckpoint, csv_logger, reduce_lr])

# Plot training & validation accuracy values
plt.figure()
plt.plot(H.history['acc'])
plt.plot(H.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('PlotResults/{0}_accplot.png'.format(file_name))  

# Plot training & validation loss values
plt.figure()
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('PlotResults/{0}_lossplot.png'.format(file_name))

# save One Hot Encoding
f = open('{0}/{1}_one_hot_encoding.txt'.format(pathTrainResult, file_name), "wb") 
f.write(pickle.dumps(lb))
f.close()
print("Saved One Hot Encoding to disk")

# save X_test
f = open('{0}/{1}_X_test.txt'.format(pathTrainResult, file_name), "wb") 
f.write(pickle.dumps(X_test))
f.close()
print("Saved X_test to disk")

# save y_test
f = open('{0}/{1}_y_test.txt'.format(pathTrainResult, file_name), "wb") 
f.write(pickle.dumps(y_test))
f.close()
print("Saved y_test to disk")
