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
datasetDir = '../external/coco/'
datasetType = 'images/train2014/COCO_train2014_'

    # Create Refexp instance.
refexp = Refexp(refexp_filename, coco_filename)



# for x in captions['annotations']

def predict_image(file_id):
	# img = refexp.loadImgs(file_id)[0]
	# img = 
	# print img
	# print file_id
	# print img
	number = str(('_0000000000000'+str(file_id))[-12:])
	# img_path = '/Users/reuben/Downloads/train2014/COCO_train2014_'+number+'.jpg'


# img_path = 'cat.jpg'
	x = image.load_img(datasetDir+datasetType+number+'.jpg', target_size=(224, 224))
	# print x
	# break
	x = image.img_to_array(x)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	output = model.predict(x)
	#h5py alternatively
	# f = open(datasetDir+'processed/'+img['file_name'],'w')
	# f.write(output)
	np.save(file=datasetDir+'processed/'+number, arr=output)
	# f.close()
	return output
	# print(file_id)
	
if __name__ == '__main__':
	for file_id in refexp.getImgIds()[:20]:
		print(file_id,"file_id")
		print(type(file_id))
		predict_image(file_id)
	# predict_image('318556')
	# features_loaded = np.load('block4_pool_features.npy')
	# assert np.array_equal(block4_pool_features, features_loaded)
