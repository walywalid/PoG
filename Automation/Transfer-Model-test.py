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

from google.colab import drive
drive.mount('/content/drive/')

image_size = [224, 224]


train_path = '/content/drive/My Drive/Colab Notebooks/Data/train'
valid_path = '/content/drive/My Drive/Colab Notebooks/Data/test'


#train_path = '/Users/w.b/Desktop/ÉLEVER/PoG/Predictive_Machine_Learning/Datasets/train'
#valid_path = '/Users/w.b/Desktop/ÉLEVER/PoG/Predictive_Machine_Learning/Datasets/test'

resnet = ResNet50(input_shape=image_size + [3], weights='imagenet',include_top=False)
for layer in resnet.layers:
    layer.trainable = False #makes sure not all layers are trained as imagenet weights are used
folders = glob('/content/drive/My Drive/Colab Notebooks/Data/train/*')
#folders - glob('/Users/w.b/Desktop/ÉLEVER/PoG/Predictive_Machine_Learning/Datasets/train/*')

len(folders)

x = Flatten()(resnet.output)

prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=resnet.input, outputs=prediction)

model.summary()

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/drive/My Drive/Colab Notebooks/Data/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


test_set = test_datagen.flow_from_directory('/content/drive/My Drive/Colab Notebooks/Data/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)



import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')



from tensorflow.keras.models import load_model

model.save('model_resnet50.h5')


y_pred = model.predict(test_set)
y_pred

import numpy as np
y_pred = np.argmax(y_pred, axis=1)


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model=load_model('model_resnet50.h5')
