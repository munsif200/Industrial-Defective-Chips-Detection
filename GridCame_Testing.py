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
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pandas as pd
import tensorflow as tf


SIZE = 224

resultsPath=r'Output/'

model=tf.keras.models.load_model('Model/model.h5')

inputFramesPath=r'Input\Abnormal/' #abnormal01
# inputFramesPath=r'Input\SelectedFrames\TrainGridCame\Normal/'
inputdata=os.listdir(inputFramesPath)
for frame in inputdata:
    fullPathImage=os.path.join(inputFramesPath,frame)
    testImage=cv2.imread(fullPathImage)
    dim = (SIZE, SIZE)
    testImage=cv2.resize(testImage, dim)
    testImage=testImage /255.
    ###########
    pred = model.predict(np.expand_dims(testImage, axis=0))
    pred_class = np.argmax(pred)
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
        plt.axis('off')
        plt.savefig(resultsPath+'Results/'+str(frame), bbox_inches='tight', pad_inches = 0)
       
    else:
        plt.imshow(testImage)
        plt.show()
        