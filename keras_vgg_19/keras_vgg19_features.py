import sys
sys.path.append('../fchollet-deep-learning-models')
sys.path.append('../google_refexp_py_lib')
# sys.path.append('../')
sys.path.append('../external/coco/PythonAPI')

from vgg19 import VGG19
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model
import numpy as np
from refexp import Refexp

base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)

refexp_filename='../google_refexp_dataset_release/google_refexp_train_201511_coco_aligned.json'
coco_filename='../external/coco/annotations/instances_train2014.json'
# datasetDir = 'external/coco/'
# datasetType = 'train2014'

    # Create Refexp instance.
refexp = Refexp(refexp_filename, coco_filename)

for file_id in refexp.getImgIds():

# for x in captions['annotations']
	number = str(('0000000000000'+str(x['image_id']))[-12:])
	img_path = '/Users/reuben/Downloads/train2014/COCO_train2014_'+number+'.jpg'


# img_path = 'cat.jpg'
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	ouput = model.predict(x)
	#h5py alternatively
	np.save(file='file_id', arr=output)
	print output

	# features_loaded = np.load('block4_pool_features.npy')
	# assert np.array_equal(block4_pool_features, features_loaded)
