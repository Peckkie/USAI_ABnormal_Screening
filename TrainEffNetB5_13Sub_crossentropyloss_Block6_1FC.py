import PIL
from keras import models
from keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import os
from tensorflow.keras import callbacks
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 16
epochs = 200

#Train
dataframe = pd.read_csv('/home/yupaporn/codes/USAI/traindf_fold1.csv')
base_dir = '/media/tohn/SSD/Images/Image1'
os.chdir(base_dir)
train_dir = os.path.join(base_dir, 'train')

#validation
valframe = pd.read_csv( '/home/yupaporn/codes/USAI/validationdf_fold1.csv')
validation_dir = os.path.join(base_dir, 'validation')

#load model
import efficientnet.tfkeras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

model_dir = '/media/tohn/SSD/ModelTrainByImages/R1_1/models/B5_R1_sub.h5'
model = load_model(model_dir)
height = width = model.input_shape[1]

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      brightness_range=[0.5,1.5],
      shear_range=0.4,
      zoom_range=0.2,
      horizontal_flip=False,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe = dataframe,
        directory = train_dir,
        x_col = 'Path Crop',
        y_col = 'Sub Position',
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='categorical')
test_generator = test_datagen.flow_from_dataframe(
        dataframe = valframe,
        directory = validation_dir,
        x_col = 'Path Crop',
        y_col = 'Sub Position',
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='categorical')

os.chdir('/media/tohn/SSD/ModelTrainByImages/R2_1')

root_logdir = '/media/tohn/SSD/ModelTrainByImages/R2_1/my_logs_block62_5FP_13sub_1FC'
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)
run_logdir = get_run_logdir()

tensorboard_cb = callbacks.TensorBoard(log_dir = run_logdir)


# os.makedirs("./models_6", exist_ok=True)

def avoid_error(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass

#Unfreez
model.trainable = True
set_trainable = False
for layer in model.layers:
    if layer.name == 'block6a_se_excite':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights))  

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

run_logdir = get_run_logdir()

tensorboard_cb = callbacks.TensorBoard(run_logdir)
#early_stop_cb = callbacks.EarlyStopping(monitor='val_acc', patience=66, mode= 'max')

history = model.fit_generator(
      avoid_error(train_generator),
      steps_per_epoch= len(dataframe)//batch_size,
      epochs=epochs,
      validation_data=avoid_error(test_generator), 
      validation_steps= len(valframe) //batch_size,
      callbacks = [tensorboard_cb])

model.save('./models/B5R2_block6_13sub_1FC.h5')
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
