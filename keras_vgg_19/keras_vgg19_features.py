import sys
sys.path.append('../fchollet-deep-learning-models')

from vgg19 import VGG19
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)

img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
np.save(file='block4_pool_features', arr=block4_pool_features)
print block4_pool_features

features_loaded = np.load('block4_pool_features.npy')
assert np.array_equal(block4_pool_features, features_loaded)
