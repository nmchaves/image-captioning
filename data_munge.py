import json
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
import cPickle as pickle
from keras.preprocessing import sequence
from utils.preprocessing import preprocess_captions,preprocess_image

captions_path = '/Users/reuben/Downloads/annotations/captions_train2014.json'


captions = open(captions_path,'r')

captions = json.loads(captions.read())


# print(preprocess_captions(["hello what is your name","my name is blah"]))

# image_dictionary = {}
# pickle.dump(image_dictionary,open("images","r+"))
# new_dict = {}
# for i,x in enumerate(captions['annotations']):
# 	number = ('0000000000000'+str(x['image_id']))[-12:]
# 	image_path = '/Users/reuben/Downloads/train2014/COCO_train2014_'+number+'.jpg'
# 	new_dict[number] = preprocess_image(image_path)
# 	print number,i

# 	if i%100 == 0:
# 		image_dictionary = pickle.load(open("images","r+"))
# 		print([x for x in image_dictionary])
# 		image_dictionary.update(new_dict)
# 		pickle.dump(new_dict,open("images","r+"))
# 		new_dict = {}
# 	if i == 300:
# 		break

# image_dictionary = pickle.load(open("images","r+"))
# image_dictionary.update(new_dict)
# pickle.dump(new_dict,open("images","r+"))
# new_dict = {}
# pickle.dump(image_dictionary,open("images","r+"))
# image_dictionary = pickle.load(open("images","r+"))
# print(len(image_dictionary))
# print([x for x in captions])
# X1 = []
# X2 = []
# y = []
num_of_caps = 1000

sentences = []
for i,x in enumerate(captions['annotations']):
	sentences.append(str(x['caption']))
	print(i)

unique, word_to_idx, idx_to_word, partial_caps, next_words = (preprocess_captions(sentences))

vocab_size = len(word_to_idx)+1
# print "NEXT WORDS",next_words

# new_next_words = []
# for x in next_words:
# 	print x
# 	a = np.zeros(vocab_size)
# 	a[x-1] = 1
# 	new_next_words.append(a)
# next_words = new_next_words

sentences = partial_caps
image_ids = []
for i,x in enumerate(captions['annotations']):
	number = str(('0000000000000'+str(x['image_id']))[-12:])
	image_path = '/Users/reuben/Downloads/train2014/COCO_train2014_'+number+'.jpg'
	# image = preprocess_image(image_path)
	# image = image[0]
	# print(len(x['caption'].split()))
	# print(x['caption'])
	for y in range(len(x['caption'].split())-1):
		image_ids.append(number)
	print(i)

# print(sentences,images)
# print(len(sentences),len(images),len(next_words))
# print(sentences[0].shape)

X1 = np.asarray(image_ids)
X2 = np.asarray(sentences)
X = [X1,X2]
# print(y)
y = np.asarray(next_words)

out = X,y,vocab_size,idx_to_word

# print(X1.shape,X2.shape,y.shape,vocab_size)

# print(captions['annotations'][:2])
# print(word_to_idx,idx_to_word)
pickle.dump(out, open( "savedoc", "r+" ) )

# assert (len(sentences)==len(images))

# #hmm, seems like i should do a single run of preprocess_captions
# for x in range(2):
# 	item = captions['annotations'][x]
# 	sentence = (item['caption'])
# 	# number = str(('0000000000000'+str(item['image_id']))[-12:])
# 	# image = image_dictionary[number]
# 	unique, word_to_idx, idx_to_word, partial_caps, next_words = (preprocess_captions([str(sentence)]))
# 	vocab_size = len(word_to_idx)
# 	assert (len(partial_caps) == len(next_words))
# 	for partial_cap,next_word in zip(partial_caps,next_words):

# 		# X1.append(image)
# 		X2.append(partial_cap)
# 		z = np.zeros(vocab_size)
# 		y.append(next_word)

# print(y.shape)
	# X = image

# # X = preprocess_image(image_path)


# # print y

