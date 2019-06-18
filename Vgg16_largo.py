import sys
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers import  Convolution2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras import applications

K.clear_session()
np.random.seed(0)

def modelo():
    vgg=applications.vgg16.VGG16()
    cnn=Sequential()
    for capa in vgg.layers:
        cnn.add(capa)
    cnn.layers.pop()
    for layer in cnn.layers:
        layer.trainable=False
    cnn.add(Dense(3,activation='softmax'))
    return cnn

#Se almacenan en variables los directorios en los que se encuentran las imágenes
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

#Parámetros importantes:
epocas=20
longitud, altura = 224, 224
batch_size = 32 #Imágenes a procesar en cada paso
pasos = 100
validation_steps = 100 #Imágenes de validación que se pasan al final de cada época
clases = 3
lr = 0.0004 #Learning rate

###Procesamiento del conjunto de entrenamieto:
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255, 
    shear_range=0.2, #Inclina las imágenes
    zoom_range=0.2, #Zoom a algunas imágenes
    horizontal_flip=True) #Invierte imágenes para distinguir direcionalidad

###Procesamiento del conjunto de validación:
#No es necesario inclinar, hacer zoom ni invertir las imágenes.
test_datagen = ImageDataGenerator(rescale=1. / 255)

###Generación del conjunto de entrenamieto:
entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical') #Se busca una clasificación categórica

###Generación del conjunto de validación:
validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

model=modelo()
model.summary()
model.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])

model.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps)

print(entrenamiento_generador.class_indices)

#score = model.evaluate_generator(validacion_generador, steps=20, verbose=1)
#print('Test accuracy:', score[1])

###Función predicción:
def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  print(array)  
  result = array[0]
  print(result)
  answer = np.argmax(result)
  print(answer)  
  if answer == 0:
    print("pred: Perro")
  elif answer == 1:
    print("pred: Gato")
  elif answer == 2:
    print("pred: Gorila")
  return answer

predict('dog.4022.jpg')


dir = './vgg16_largo/'
if not os.path.exists(dir):
    os.mkdir(dir)
model.save('./vgg16_largo/modelo_vgg16_largo.h5')#Se guarda la estructura de la cnn
model.save_weights('./vgg16_largo/pesos_vgg16_largo.h5')#Se guardan los pesos de la cnn