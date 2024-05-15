
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import vgg16
from tensorflow.keras.optimizers import Adam, SGD
import scipy  
import os
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras import layers 
from matplotlib.patches import Rectangle 
from skimage.feature.peak import peak_local_max  

image_directory = 'SelectedFrames/TrainGridCame/'
SIZE = 224

dataset = [] 
label = []  

Clear_Abnormal = os.listdir(image_directory + 'Abnormal/')
for i, image_name in enumerate(Clear_Abnormal):   
    
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'Abnormal/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)

Clear_Normal_images = os.listdir(image_directory + 'Normal/')
for i, image_name in enumerate(Clear_Normal_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'Normal/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)

dataset = np.array(dataset)
label = np.array(label)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)

X_train = X_train /255.
X_test = X_test /255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def get_model(input_shape = (224,224,3)):
    vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape = input_shape)
    for layer in vgg.layers[:-5]:    
        print(layer.name)
        layer.trainable = False 

    x = vgg.output
    x = GlobalAveragePooling2D()(x) 
    x = Dense(2, activation="softmax")(x)   

    model = Model(vgg.input, x)
    model.compile(loss = "categorical_crossentropy", 
                  optimizer = SGD(lr=0.001, momentum=0.9), metrics=["accuracy"])
    return model

model = get_model(input_shape = (224,224,3))
print(model.summary())


'''def get_model(input_shape = (224,224,3)):
     inputs = layers.Input(input_shape)
     conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
     conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
     pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
     conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
     conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
     pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
     conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
     conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
     pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
     conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
     conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
     drop4 = layers.Dropout(0.5)(conv4)
     pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

     conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
     conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
     drop5 = layers.Dropout(0.5)(conv5)

     up6 = layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(drop5))
     merge6 = layers.concatenate([drop4,up6], axis = 3)
     conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
     conv6 =layers. Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

     up7 = layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(conv6))
     merge7 = layers.concatenate([conv3,up7], axis = 3)
     conv7 =layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
     conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

     up8 = layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(conv7))
     merge8 = layers.concatenate([conv2,up8], axis = 3)
     conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
     conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
     gap1=layers.GlobalAveragePooling2D()(conv8)
     x = Dense(2, activation="softmax")(gap1)
     model = Model(inputs, x)
     model.compile(loss = "categorical_crossentropy",
                   optimizer = SGD(lr=0.00001, momentum=0.9), metrics=["accuracy"])
     return model'''


model = get_model(input_shape = (224,224,3))
print(model.summary())

history = model.fit(X_train, y_train, batch_size=16, epochs=30, verbose = 1, 
                    validation_data=(X_test,y_test))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

n=10  
img = X_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) 
print("The prediction for this image is: ", np.argmax(model.predict(input_img)))
print("The actual label for this image is: ", np.argmax(y_test[n]))

#Print confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = np.argmax(model.predict(X_test), axis=1)
cm=confusion_matrix(np.argmax(y_test, axis=1), y_pred)  
sns.heatmap(cm, annot=True)

Clear_Abnormal_idx = np.where(y_pred == 1)[0]
predicted_as_para=[]

inputFramesPath=r'inputData\SelectedFrames\abnormal01/' #abnormal01
#inputFramesPath=r'inputData\SelectedFrames\noarml01/'
inputdata=os.listdir(inputFramesPath)
for frame in inputdata:
    fullPathImage=os.path.join(inputFramesPath,frame)
    testImage=cv2.imread(fullPathImage)
    dim = (224, 224)
    testImage=cv2.resize(testImage, dim)
    testImage=testImage /255.
    ###########
    pred = model.predict(np.expand_dims(testImage, axis=0))
    pred_class = np.argmax(pred)
    print(pred_class)
    if pred_class==1:
        last_layer_weights = model.layers[-1].get_weights()[0] 
        last_layer_weights_for_pred = last_layer_weights[:, pred_class]
        last_conv_model = Model(model.input, model.get_layer("block5_conv3").output)
        last_conv_output = last_conv_model.predict(testImage[np.newaxis,:,:,:])
        last_conv_output = np.squeeze(last_conv_output)
        h = int(testImage.shape[0]/last_conv_output.shape[0])
        w = int(testImage.shape[1]/last_conv_output.shape[1])
        upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
        heat_map = np.dot(upsampled_last_conv_output.reshape((testImage.shape[0]*testImage.shape[1], 512)), 
                     last_layer_weights_for_pred).reshape(testImage.shape[0],testImage.shape[1])
        heat_map[testImage[:,:,0] == 0] = 0  
        peak_coords = peak_local_max(heat_map, num_peaks=5, threshold_rel=0.5, min_distance=10) 
        plt.imshow(testImage.astype('float32').reshape(testImage.shape[0],testImage.shape[1],3))
        plt.imshow(heat_map, cmap='jet', alpha=0.30)
        plt.show()
    else:
        plt.imshow(testImage)
        plt.show()
        
        