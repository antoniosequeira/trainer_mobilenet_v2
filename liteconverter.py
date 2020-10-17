import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#has to be with tensorflow>=2.0

pathTrainResult = 'TrainResult'
combination = 5
batch_size = 32
epochs = 500
WIDTH = 128
HEIGHT = WIDTH
file_name = 'comb_{0}_batch_{1}_shape_{2}.h5'.format(combination, batch_size, WIDTH)

model = load_model('{0}/{1}'.format(pathTrainResult, file_name))


# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open('{0}.tflite'.format(file_name), "wb").write(tflite_model)
