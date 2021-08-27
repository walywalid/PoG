from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
config.gpu_options.allow_growth = True
session=InteractiveSession(config=config)

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
from glob import glob

image_size = [224, 224]

train_path = 'w.b/Desktop/Élever/PoG/Machine_Learning/Datasets/train'
valid_path = 'w.b/Desktop/Élever/PoG/Machine_Learning/Datasets/test'

resnet = ResNet50(input_shape=image_size + [3], weights='imagenet',include_top=False)
for layer in resnet.layers:
    layer.trainable = False #makes sure not all layers are trained as imagenet weights are used

folders - glob('w.b/Desktop/Élever/PoG/Machine_Learning/Datasets/train/*')
