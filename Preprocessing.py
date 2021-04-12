import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import math
import shutil
import PIL
import matplotlib.pyplot as plt
from numpy import asarray
from PIL import Image 
import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.layers as layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
#from tensorflow.python.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import VGG16
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Activation

# Instantiate global constants
data_dir = os.path.abspath('./drive/MyDrive/final_assignment_xrays/XRays')
test_path = data_dir + '/test'
train_path = data_dir + '/train'
normal_path = train_path + '/NORMAL'  
pneumonia_path = train_path + '/PNEUMONIA'

# Class for creating Data Generators
class GenerateData:
    
    def Random_Contrast(img: tf.Tensor) -> tf.Tensor:
        
        # Contrast Augmentation for addition to the ImageDataGenerator       
        img_ = tf.image.random_contrast(img, 1, 1.5)
        return img_

    def Random_Contrast_denoising(img: tf.Tensor) -> tf.Tensor:
        # Contrast Augmentation for addition to the ImageDataGenerator  
        img = tf.image.random_contrast(img, 1, 1.5)
        img = tf.reshape(img,[162,128])
        img = tf.dtypes.cast(img, tf.uint8)
        img = np.array(img).astype('uint8')
        img_ = cv2.fastNlMeansDenoising(img,h=10,templateWindowSize=7,searchWindowSize=21)
        #img_denoised = GenerateData.denoising(img_)
        return img_.reshape(162, 128, 1).astype('float64')

    def Random_Contrast_denoising_rgb(img: tf.Tensor) -> tf.Tensor:
        img = tf.image.random_contrast(img, 1, 1.5)
        img = tf.dtypes.cast(img, tf.uint8)
        img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
        img = tf.reshape(img,[162,128,-1])
        img = np.array(img).astype('uint8')
        img_ = cv2.fastNlMeansDenoising(img,h=10,templateWindowSize=7,searchWindowSize=21)
        #img_denoised = GenerateData.denoising(img_)
        return img_.reshape(162, 128, 3).astype('float64')

    def denoising(img: tf.Tensor) -> tf.Tensor:
      img = img.reshape(162, 128).astype('uint8')
      img_ = cv2.fastNlMeansDenoising(img,h=10,templateWindowSize=7,searchWindowSize=21)
      return img_.reshape(162, 128, 1).astype('float64')

    def denoising_rgb(img: tf.Tensor) -> tf.Tensor:
      img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY)
      #img = cv2.Canny(enhance_contrast(img, disk(6)), 50, 210)
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
      img = img.astype('float64')
      img = img.reshape(162, 128,-1).astype('uint8')
      img_ = cv2.fastNlMeansDenoising(img,h=10,templateWindowSize=7,searchWindowSize=21)
      return img_.reshape(162, 128, 3).astype('float64')
    
    #Simple thresholding
    def simple_thresholding(img: tf.Tensor) -> tf.Tensor:
        img = img.reshape(162, 128).astype('uint8')
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        return img.reshape(162, 128, -1).astype('float64')
    
    def edge_detect(img: tf.Tensor) -> tf.Tensor:
      # Canny edge detection for addition to the ImageDataGenerator
      img_ = cv2.Canny(enhance_contrast(img.reshape(162, 128).astype('uint8'), disk(6)), 50, 210)
      return img_.reshape(162, 128, 1).astype('float64')

    def rgb_edge_detect(img: tf.Tensor) -> tf.Tensor:
      # Canny edge detection for addition to the ImageDataGenerator for RGB images
      grey_img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY)
      img_ = cv2.Canny(enhance_contrast(grey_img, disk(6)), 50, 210)
      color_img = cv2.cvtColor(img_, cv2.COLOR_GRAY2RGB)
      return color_img.astype('float64')

    def initialise():
        
        # Instantiate Generator Object, holding back 10% of samples for Validation, and normalising the pixel values.
        # Note that the holdout is only relevant if 'subset' is defined when instantiating a generator, otherwise
        # the whole set is returned
        global generator, generator_denoising,generator_denoising_rgb, generator_edge, generator_edge_rgb, augmented_generator, augmented_generator_denoising,augmented_generator_denoising_rgb,augmented_generator_edge, augmented_generator_edge_rgb,generator_ST 
        
        generator = ImageDataGenerator(rescale=1./255,
                                       validation_split=0.1)
        
        generator_ST = ImageDataGenerator(rescale=1./255,validation_split=0.1,
                                                    preprocessing_function=GenerateData.simple_thresholding)
        
        generator_denoising = ImageDataGenerator(rescale=1./255,
                                       validation_split=0.1,
                                       preprocessing_function=GenerateData.denoising)    

        generator_denoising_rgb = ImageDataGenerator(rescale=1./255,
                                validation_split=0.1,
                                preprocessing_function=GenerateData.denoising_rgb)
        
        generator_edge = ImageDataGenerator(rescale=1./255,
                                       validation_split=0.1,
                                       preprocessing_function=GenerateData.edge_detect)
        
        generator_edge_rgb = ImageDataGenerator(rescale=1./255,
                                       validation_split=0.1,
                                       preprocessing_function=GenerateData.rgb_edge_detect)
        
        augmented_generator = ImageDataGenerator(rescale=1./255,
                                                 rotation_range=10,
                                                 width_shift_range=0,
                                                 height_shift_range=0,
                                                 vertical_flip=False,
                                                 horizontal_flip=False,
                                                 validation_split=0.1,
                                                 preprocessing_function = GenerateData.Random_Contrast)

        augmented_generator_denoising = ImageDataGenerator(rescale=1./255,
                                                 rotation_range=10,
                                                 width_shift_range=0,
                                                 height_shift_range=0,
                                                 vertical_flip=False,
                                                 horizontal_flip=False,
                                                 validation_split=0.1,
                                                 preprocessing_function = GenerateData.Random_Contrast_denoising) 

        augmented_generator_denoising_rgb = ImageDataGenerator(rescale=1./255,
                                                 rotation_range=10,
                                                 width_shift_range=0,
                                                 height_shift_range=0,
                                                 vertical_flip=False,
                                                 horizontal_flip=False,
                                                 validation_split=0.1,
                                                 preprocessing_function = GenerateData.Random_Contrast_denoising_rgb)
        
        augmented_generator_edge = ImageDataGenerator(rescale=1./255,
                                                 rotation_range=10,
                                                 validation_split=0.1,
                                                 preprocessing_function = GenerateData.edge_detect) 

        augmented_generator_edge_rgb = ImageDataGenerator(rescale=1./255,
                                                 rotation_range=10,
                                                 validation_split=0.1,
                                                 preprocessing_function = GenerateData.rgb_edge_detect) 
     
        
    def data_flow(train_path, color,denoise=False,ST=False):
        
        # Instantiate training data generators. Convert image to grayscale, and resize image to 162*128 pixels for 
        # LeNet5 architecture, also used for examining class imbalance.
        if denoise and color=='grayscale':
          data = generator_denoising.flow_from_directory(train_path,
                                             target_size=(162,128),
                                             color_mode=color,
                                             batch_size=32,
                                             class_mode="categorical",
                                             shuffle=True)

        elif denoise and color=='rgb':
          data = generator_denoising_rgb.flow_from_directory(train_path,
                                             target_size=(162,128),
                                             color_mode=color,
                                             batch_size=32,
                                             class_mode="categorical",
                                             shuffle=True)
        elif denoise == False and ST:
            print('\n entered the if condition')
            data = generator_ST.flow_from_directory(train_path,
                                             target_size=(162,128),
                                             color_mode=color,
                                             batch_size=32,
                                             class_mode="categorical",
                                             shuffle=True)
            
        else:
          data = generator.flow_from_directory(train_path,
                                             target_size=(162,128),
                                             color_mode=color,
                                             batch_size=32,
                                             class_mode="categorical",
                                             shuffle=True)
        
        return data
    
    
    def data_flow_augmented(train_path, color,denoise=False):
        
        # Instantiate training data generators using the augmented generator. Convert image to grayscale, and 
        # resize image to 162*128 pixels for classic LeNet5 architecture, also used for examining class imbalance.
        if denoise and color=='grayscale':
          data = augmented_generator_denoising.flow_from_directory(train_path,
                                                       target_size=(162,128),
                                                       color_mode=color,
                                                       batch_size=32,
                                                       class_mode="categorical",
                                                       shuffle=True)
        elif denoise and color=='rgb':
          data = augmented_generator_denoising_rgb.flow_from_directory(train_path,
                                                       target_size=(162,128),
                                                       color_mode=color,
                                                       batch_size=32,
                                                       class_mode="categorical",
                                                       shuffle=True)
        else:
          data = augmented_generator.flow_from_directory(train_path,
                                                       target_size=(162,128),
                                                       color_mode=color,
                                                       batch_size=32,
                                                       class_mode="categorical",
                                                       shuffle=True)
        
        return data
    
    
    def training_data_flow(train_path, color, denoise=False,ST=False):
        
        # Create Training Set Generator from training subset, for quick runs not executing a k-fold strategy
        print(denoise)
        print(ST)
        print(color)
        if denoise and color=='grayscale':
          data_train = generator_denoising.flow_from_directory(train_path,
                                                   target_size=(162,128),
                                                   color_mode=color,
                                                   batch_size=32,
                                                   class_mode="categorical",
                                                   shuffle=True,
                                                   subset='training')
        elif denoise and color=='rgb':
          data_train = generator_denoising_rgb.flow_from_directory(train_path,
                                                   target_size=(162,128),
                                                   color_mode=color,
                                                   batch_size=32,
                                                   class_mode="categorical",
                                                   shuffle=True,
                                                   subset='training')
        elif denoise == False and ST:
            print('\n entered the if condition')
            data_train = generator_ST.flow_from_directory(train_path,
                                             target_size=(162,128),
                                             color_mode=color,
                                             batch_size=32,
                                             class_mode="categorical",
                                             shuffle=True,
                                             subset='training')
        else:
          data_train = generator.flow_from_directory(train_path,
                                                   target_size=(162,128),
                                                   color_mode=color,
                                                   batch_size=32,
                                                   class_mode="categorical",
                                                   shuffle=True,
                                                   subset='training')
        return data_train
    
    
    def training_data_flow_augmented(train_path, color,denoise=False):
        
        # Create Training Set Generator from training subset using the augmented generator,  
        # for quick runs not executing a k-fold strategy
        if denoise and color=='grayscale':
          data_train = augmented_generator_denoising.flow_from_directory(train_path,
                                                             target_size=(162,128),
                                                             color_mode=color,
                                                             batch_size=32,
                                                             class_mode="categorical",
                                                             shuffle=True,
                                                             subset='training')
        elif denoise and color=='rgb':
          data_train = augmented_generator_denoising_rgb.flow_from_directory(train_path,
                                                             target_size=(162,128),
                                                             color_mode=color,
                                                             batch_size=32,
                                                             class_mode="categorical",
                                                             shuffle=True,
                                                             subset='training')
        else:
          data_train = augmented_generator.flow_from_directory(train_path,
                                                             target_size=(162,128),
                                                             color_mode=color,
                                                             batch_size=32,
                                                             class_mode="categorical",
                                                             shuffle=True,
                                                             subset='training')
        return data_train
    
    
    def validation_data_flow(train_path, color,denoise=False,ST=False):
        
        # Create Validation Set Generator from validation subset for quick runs not executing a k-fold strategy
        if denoise and color=='grayscale':
          data_val = generator_denoising.flow_from_directory(train_path,
                                                 target_size=(162,128),
                                                 color_mode=color,
                                                 batch_size=32,
                                                 class_mode="categorical",
                                                 shuffle=True,
                                                 subset='validation')
        elif denoise and color=='rgb':
          data_val = generator_denoising_rgb.flow_from_directory(train_path,
                                                 target_size=(162,128),
                                                 color_mode=color,
                                                 batch_size=32,
                                                 class_mode="categorical",
                                                 shuffle=True,
                                                 subset='validation')
        elif denoise == False and ST:
            print('\n entered the if condition')
            data_val = generator_ST.flow_from_directory(train_path,
                                             target_size=(162,128),
                                             color_mode=color,
                                             batch_size=32,
                                             class_mode="categorical",
                                             shuffle=True,
                                             subset='validation')
        else:
          data_val = generator.flow_from_directory(train_path,
                                                 target_size=(162,128),
                                                 color_mode=color,
                                                 batch_size=32,
                                                 class_mode="categorical",
                                                 shuffle=True,
                                                 subset='validation')
        return data_val
    
    
    def validation_data_flow_augmented(train_path, color,denoise=False):
        
        # Create Validation Set Generator from validation subset using the augmented generator, 
        # for quick runs not executing a k-fold strategy
        if denoise and color=='grayscale':
          data_val = augmented_generator_denoising.flow_from_directory(train_path,
                                                           target_size=(162,128),
                                                           color_mode=color,
                                                           batch_size=32,
                                                           class_mode="categorical",
                                                           shuffle=True,
                                                           subset='validation')
        elif denoise and color=='rgb':
          data_val = augmented_generator_denoising_rgb.flow_from_directory(train_path,
                                                           target_size=(162,128),
                                                           color_mode=color,
                                                           batch_size=32,
                                                           class_mode="categorical",
                                                           shuffle=True,
                                                           subset='validation')
        else:
          data_val = augmented_generator.flow_from_directory(train_path,
                                                           target_size=(162,128),
                                                           color_mode=color,
                                                           batch_size=32,
                                                           class_mode="categorical",
                                                           shuffle=True,
                                                           subset='validation')
        return data_val
    
    
    def test_data_flow(test_path, color,denoise=False,ST=False):
        
        # Create test data generator. Note shuffle=false here, this is important for extracting the 'true' class
        # labels later on
        if denoise and color=='grayscale':
          data_test = generator_denoising.flow_from_directory(test_path,  
                                                  target_size=(162,128),
                                                  color_mode=color,
                                                  batch_size=16,
                                                  class_mode="categorical",
                                                  shuffle=False,
                                                  seed = 42)
        elif denoise and color=='rgb':
          data_test = generator_denoising_rgb.flow_from_directory(test_path,  
                                                  target_size=(162,128),
                                                  color_mode=color,
                                                  batch_size=16,
                                                  class_mode="categorical",
                                                  shuffle=False,
                                                  seed = 42)
        elif denoise == False and ST:
            print('\n entered the if condition')
            data_test = generator_ST.flow_from_directory(test_path,
                                             target_size=(162,128),
                                             color_mode=color,
                                             batch_size=32,
                                             class_mode="categorical",
                                             shuffle=True,
                                             seed=42)
        else:
          data_test = generator.flow_from_directory(test_path,  
                                                  target_size=(162,128),
                                                  color_mode=color,
                                                  batch_size=16,
                                                  class_mode="categorical",
                                                  shuffle=False,
                                                  seed = 42)
        return data_test

    def test_data_flow_augmented(test_path, color,denoise=False):
        
        # Create test data generator. Note shuffle=false here, this is important for extracting the 'true' class
        # labels later on
        if denoise and color=='grayscale':
          data_test = augmented_generator_denoising.flow_from_directory(test_path,  
                                                  target_size=(162,128),
                                                  color_mode=color,
                                                  batch_size=16,
                                                  class_mode="categorical",
                                                  shuffle=False,
                                                  seed = 42)
        elif denoise and color=='rgb':
          data_test = augmented_generator_denoising_rgb.flow_from_directory(test_path,  
                                                  target_size=(162,128),
                                                  color_mode=color,
                                                  batch_size=16,
                                                  class_mode="categorical",
                                                  shuffle=False,
                                                  seed = 42)
        else:
          data_test = augmented_generator.flow_from_directory(test_path,  
                                                  target_size=(162,128),
                                                  color_mode=color,
                                                  batch_size=16,
                                                  class_mode="categorical",
                                                  shuffle=False,
                                                  seed = 42)
        return data_test 

def ShowImages():
    
    # Method to print some x-rays to the screen for evaluation by my extraordinarily medically learned self ...... 
    fig, ax = plt.subplots(4, 4, figsize=(15, 7))
    ax = ax.ravel()
    plt.tight_layout()

    for i in range(0,8):
            dir_ = 'train'
            set_dir = data_dir + '/' + dir_
            ax[i].imshow(plt.imread(set_dir+'/NORMAL/'+os.listdir(set_dir+'/NORMAL')[i]), cmap='gray')
            ax[i].set_title('Set: {}, Condition: Normal'.format(dir_))
            dir_ = 'test'
            set_dir = data_dir + '/' + dir_
            ax[i+8].imshow(plt.imread(set_dir+'/PNEUMONIA/'+os.listdir(set_dir+'/PNEUMONIA')[i]), cmap='gray')
            ax[i+8].set_title('Set: {}, Condition: Pneumonia'.format(dir_))
            
            
def CalculateDataStats():
    
    data = GenerateData.data_flow(train_path, "grayscale")
    
    # Examine class imbalance across training and validation data, using the 'data' generator 
    df = pd.DataFrame({'data':data.classes})

    # Class Counts and ratio
    normal = int(df[df.data==data.class_indices['NORMAL']].count())
    pneumonia = int(df[df.data==data.class_indices['PNEUMONIA']].count())
    ratio = round(pneumonia / normal, 2)

    # Class Weights
    normal_weight = ratio
    pneumonia_weight = 1.0

    class_weights = {
     data.class_indices['NORMAL']:normal_weight,
     data.class_indices['PNEUMONIA']:pneumonia_weight
    }

    text = "Normal:{:.0f}\nPneumonia:{:.0f}\nImbalance Ratio: {:.2f}\n".format(normal, pneumonia, ratio)
    print(text)
    text = "Weighting classes by:\nNormal:{:.2f}\nPneumonia:{:.2f}\n".format(normal_weight, pneumonia_weight)
    print(text)
    
    return class_weights


def MakeDirectories(k):

    # Main loop - 'for each fold'
    for i in range (0, k):
        
        print('Creating Directory for fold ' + str(i+1))
        
        # Create Fold Directories

        if not os.path.exists(data_dir + '/fold' + str(i+1)):
            os.mkdir(data_dir + '/fold' + str(i+1))

        # Create Train and Validate Directories in each Fold

        if not os.path.exists(data_dir + '/fold' + str(i+1) + '/train'):
            os.mkdir(data_dir + '/fold' + str(i+1) + '/train')

        if not os.path.exists(data_dir + '/fold' + str(i+1) + '/validate'):
            os.mkdir(data_dir + '/fold' + str(i+1) + '/validate')    

        # Create Class Directories in Train and Validate Directories

        if not os.path.exists(data_dir + '/fold' + str(i+1) + '/train' + '/NORMAL'):
            os.mkdir(data_dir + '/fold' + str(i+1) + '/train' + '/NORMAL')

        if not os.path.exists(data_dir + '/fold' + str(i+1) + '/train' + '/PNEUMONIA'):
            os.mkdir(data_dir + '/fold' + str(i+1) + '/train' + '/PNEUMONIA')

        if not os.path.exists(data_dir + '/fold' + str(i+1) + '/validate' + '/NORMAL'):
            os.mkdir(data_dir + '/fold' + str(i+1) + '/validate' + '/NORMAL')

        if not os.path.exists(data_dir + '/fold' + str(i+1) + '/validate' + '/PNEUMONIA'):
            os.mkdir(data_dir + '/fold' + str(i+1) + '/validate' + '/PNEUMONIA') 
         
        # Create Checkpoint Directories 
        
        if not os.path.exists(data_dir + '/checkpoints'):
            os.mkdir(data_dir + '/checkpoints')
            
        if not os.path.exists(data_dir + '/checkpoints' + '/LogReg'):
            os.mkdir(data_dir + '/checkpoints'+ '/LogReg')
            
        if not os.path.exists(data_dir + '/checkpoints' + '/VGG16'):
            os.mkdir(data_dir + '/checkpoints'+ '/VGG16')
        
        if not os.path.exists(data_dir + '/fold' + str(i+1) + '/checkpoints'):
            os.mkdir(data_dir + '/fold' + str(i+1) + '/checkpoints')          
            
        if not os.path.exists(data_dir + '/fold' + str(i+1) + '/checkpoints' + '/augmented'):
            os.mkdir(data_dir + '/fold' + str(i+1) + '/checkpoints' + '/augmented')
            

def Create_KFold_TrainingData(k):
            
    # Copy All Training Data Into Train Folds        
    
    # Main loop - 'for each fold'
    for k in range (0, k):
        
        # Set Directory to the current fold for NORMAL class
        dir_ = (data_dir + '/fold' + str(k+1) + '/train' + '/NORMAL')
        print('Copying all Training Data into Fold ' + str(k+1) + ' Directory')
        
        # Iterate over every 'NORMAL' x-ray and copy
        for filename in os.listdir(normal_path):

            shutil.copy(normal_path + '/' + filename, dir_)
        
        # Set Directory to the current fold for PNEUMONIA class
        dir_ = (data_dir + '/fold' + str(k+1) + '/train' + '/PNEUMONIA')
        
        # Iterate over every 'PNEUMONIA' x-ray and copy
        for filename in os.listdir(pneumonia_path):

            shutil.copy(pneumonia_path + '/' + filename, dir_)

def Create_KFold_ValidationData(k):
    
    # Move Validation Fold Data out of Train Fold Directories and into Validation Fold Directories
    
    data = GenerateData.data_flow(train_path, "grayscale")
    
    # Class Counts and ratio
    df = pd.DataFrame({'data':data.classes})
    normal = int(df[df.data==data.class_indices['NORMAL']].count())
    pneumonia = int(df[df.data==data.class_indices['PNEUMONIA']].count())
    count_normal = math.ceil(normal/k)
    count_pneumonia = math.ceil(pneumonia/k)
    
    
    # 'NORMAL' x-rays
    # Instantiate counts
    i = 0
    j = 1
    
    # Set source and target directories to fold 1
    dir_Normal = (data_dir + '/fold' + str(j) + '/train' + '/NORMAL')
    dir_ = (data_dir + '/fold' + str(j) + '/validate' + '/NORMAL')
    print('Moving NORMAL Validation Data Out Of Fold ' + str(j) + ' Train Directory')
    
    # Iterate over every 'NORMAL' x-ray in source directory
    for filename in os.listdir(dir_Normal):
        
        # Move Files from source to. target directories
        shutil.move(dir_Normal + '/' + filename, dir_)
        
        # When we have moved 1/k'th of the images, Set source and target directories to the next fold
        if i > j*count_normal:

            j = j + 1
            dir_ = (data_dir + '/fold' + str(j) + '/validate' + '/NORMAL')
            dir_Normal = (data_dir + '/fold' + str(j) + '/train' + '/NORMAL')
            print('Moving NORMAL Validation Data Out Of Fold ' + str(j) + ' Train Directory')

        i = i + 1    
    
    
    # 'PNEUMONIA' x-rays
    # Instantiate counts
    i = 0
    j = 1

    # Set source and target directories to fold 1
    dir_Pneumonia = (data_dir + '/fold' + str(j) + '/train' + '/PNEUMONIA')
    dir_ = (data_dir + '/fold' + str(j) + '/validate' + '/PNEUMONIA')
    print('Moving PNEUMONIA Validation Data Out Of Fold ' + str(j) + ' Train Directory')

    # Iterate over every 'PNEUMONIA' x-ray in source directory
    for filename in os.listdir(dir_Pneumonia):

        # Move Files from source to. target directories
        shutil.move(dir_Pneumonia + '/' + filename, dir_)

        # When we have moved 1/k'th of the images, Set source and target directories to the next fold
        if i > j*count_pneumonia:

            j = j + 1
            dir_ = (data_dir + '/fold' + str(j) + '/validate' + '/PNEUMONIA')
            dir_Pneumonia = (data_dir + '/fold' + str(j) + '/train' + '/PNEUMONIA')
            print('Moving PNEUMONIA Validation Data Out Of Fold ' + str(j) + ' Train Directory')

        i = i + 1   

def LogReg():
    
    classifier = Sequential()
    
    classifier.add(Flatten(input_shape=(162,128,1)))
    
    classifier.add(Dense(2))
    
    classifier.add(Activation('softmax'))
    
    classifier.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.0001), metrics=['accuracy'])
    
    classifier.summary()
    
    return classifier
    
    
def VGG_16(freeze):
    
    # Instantiate a pre-trained model, trained on the imagenet database, and unfreeze the final convolutional layer

    base_model = VGG16(weights='imagenet', input_shape=(162,128,3), include_top=False)

    x = base_model.output
    
    x = Flatten()(x)
    
    x = Dense(64, activation='relu')(x)
    
    x = Dropout(0.33)(x)
    
    x = BatchNormalization()(x)
    
    output = Dense(2, activation='softmax')(x)

    classifier = Model(inputs=base_model.input, outputs=output)
    
    # Unfreeze the final convolutional layer
    if freeze == True:
        for layer in base_model.layers:
            if layer.name != 'block5_conv3':
                layer.trainable = False
            else:
                layer.trainable = True
                print("Unfreezing layer: block5_conv3")
    
    # If we do not want to unfreeze the final layer    
    else: 
        for layer in base_model.layers:
             layer.trainable = False

    classifier.compile(loss='binary_crossentropy', optimizer = Adam(learning_rate=0.0001), metrics=['accuracy'])

    classifier.summary()
    
    return classifier


def LeNet5():
    
    # Define the Lenet5 model

    classifier = Sequential()
    
    classifier.add(layers.Conv2D(6,(5,5), input_shape=(162,128,1),strides=1, padding='valid', activation='relu'))
    
    classifier.add(layers.AveragePooling2D(pool_size=(2,2),strides=2))
    
    classifier.add(layers.Conv2D(16,(5,5),strides=1, padding='valid',  activation='relu'))
    
    classifier.add(layers.AveragePooling2D(pool_size=(2,2),strides=2))
    
    classifier.add(layers.Conv2D(120,(5,5),strides=1, padding='valid',  activation='relu'))
    
    classifier.add(layers.Flatten())
    
    classifier.add(layers.Dense(84, input_shape=(120,)))
        
    classifier.add(layers.Dense(2, activation='softmax'))
    
    classifier.summary()
    
    classifier.compile(loss=keras.losses.categorical_crossentropy, optimizer = Adam(lr=0.0001), metrics=['accuracy'])
    
    return classifier