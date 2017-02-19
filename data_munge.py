import json
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
from keras.preprocessing import sequence
from utils.preprocessing import preprocess_captions,preprocess_image



captions_path = '/Users/reuben/Downloads/annotations/captions_train2014.json'


captions = open(captions_path,'r')

captions = json.loads(captions.read())



# print([x for x in captions])
X = []
y = []

# make image dictionary first
# image_dictionary = {}
# for x in BLAH:
# 	image_dictionary[x] = preprocess_image(x)

for x in range(2):
	sentence = (captions['annotations'][x]['caption'])
	number = captions['annotations'][0]['image_id']
	number = ('0000000000000'+str(number))[-12:]
	image_path = '/Users/reuben/Downloads/train2014/COCO_train2014_'+number+'.jpg'
	image = preprocess_image(image_path)
	unique, word_to_idx, idx_to_word, partial_caps, next_words = (preprocess_captions([str(sentence)]))
	X += [partial_caps,image]
	y += next_words

print(X,y)
