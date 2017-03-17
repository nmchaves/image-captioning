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
from PIL import Image

base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
model2 = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
model3 = Model(input=base_model.input, output=base_model.get_layer('predictions').output)

refexp_filename='../google_refexp_dataset_release/google_refexp_train_201511_coco_aligned.json'
coco_filename='../external/coco/annotations/instances_train2014.json'
datasetDir = '../external/coco/'
datasetType = 'images/train2014/COCO_train2014_'

    # Create Refexp instance.
refexp = Refexp(refexp_filename, coco_filename)



# for x in captions['annotations']

def predict_image(file_id,bbox=[]):
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
	output2 = model2.predict(x)
	output3 = model3.predict(x)
	#h5py alternatively
	# f = open(datasetDir+'processed/'+img['file_name'],'w')
	# f.write(output)
	storage_dir = '/extra/'
	np.save(file=storage_dir+'processed/'+number, arr=output)
    np.save(file=storage_dir+'processed_flatten/'+number, arr=output2)
    np.save(file=storage_dir+'processed_predictions/'+number, arr=output3)
	'''
	np.save(file=datasetDir+'processed/'+number, arr=output)
        np.save(file=datasetDir+'processed_flatten/'+number, arr=output2)
	np.save(file=datasetDir+'processed_predictions/'+number, arr=output3)
	'''
	# f.close()
	if not bbox:
		return output,output2,output3
	else:
		original = image.load_img(datasetDir+datasetType+number+'.jpg')
		region = original.crop((int(bbox[0]),int(bbox[1]),int(bbox[0])+int(bbox[2]),int(bbox[1])+int(bbox[3])))
		region = region.resize((224,224))
		region = image.img_to_array(region)
        	region = np.expand_dims(region, axis=0)
        	region = preprocess_input(region)

        	reg_output = model.predict(region)
        	reg_output2 = model2.predict(region)
        	reg_output3 = model3.predict(region)

        	np.save(file=storage_dir+'processed/'+number+'_b', arr=output)
    		np.save(file=storage_dir+'processed_flatten/'+number+'_b', arr=output2)
    		np.save(file=storage_dir+'processed_predictions/'+number+'_b', arr=output3)

		return output,output2,output3,reg_output,reg_output2,reg_output3
	# print(file_id)

if __name__ == '__main__':
	for file_id in refexp.getImgIds()[:20]:
		print(file_id,"file_id")
		print(type(file_id))
		predict_image(file_id)
	# predict_image('318556')
	# features_loaded = np.load('block4_pool_features.npy')
	# assert np.array_equal(block4_pool_features, features_loaded)
