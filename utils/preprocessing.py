import sys
sys.path.append('../external/coco/PythonAPI')
sys.path.append('../google_refexp_py_lib')
from pycocotools.coco import COCO

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence, image
from keras.applications.imagenet_utils import preprocess_input

import numpy as np
import cPickle as pickle

STOP_TOKEN_IDX = 0

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


# this function has been edited to read in pickled dictionaries produced by preprocess_coco function
# also now returns just the two dictionaries, partial captions, and next words as indices (not one-hot)
# also takes caption_seqs instead of captions, which makes it easier to repeat the image_ids
def preprocess_captions(caption_seqs):
	word_to_idx = pickle.load(open('coco_word_to_idx','rb'))
	idx_to_word = pickle.load(open('coco_idx_to_word','rb'))
	database_stats = pickle.load(open('coco_stats','rb'))

	partial_caps, next_words = partial_captions_and_next_words(caption_seqs, word_to_idx, database_stats['max_cap_len'])

	return word_to_idx, idx_to_word, partial_caps, next_words


# this function creates dictionaries and finds max caption length for the entire coco dataset
def preprocess_coco(include_val=True):
	# get annotations
	train_ann_filename = '../external/coco/annotations/captions_train2014.json'
	coco_train = COCO(train_ann_filename)
	trainAnnIds = coco_train.getAnnIds()
	trainAnns = coco_train.loadAnns(trainAnnIds)

	if include_val:
		val_ann_filename = '../external/coco/annotations/captions_val2014.json'
		coco_val = COCO(val_ann_filename)
		valAnnIds = coco_val.getAnnIds()
		valAnns = coco_val.loadAnns(valAnnIds)

		anns = trainAnns + valAnns
	else:
		anns = trainAnns

	# get captions from annotations and find longest
	captions = [ann['caption'].encode('ascii') for ann in anns]
	caption_seqs = [text_to_word_sequence(c) for c in captions]
	max_cap_len = max([len(seq) for seq in caption_seqs])
	# a pickled dictionary is overkill here
	# but i implemented like this in case we want to get any other
	# databse info here 
	database_stats = {'max_cap_len': max_cap_len}

	# create dictionaries for dataset
	unique = unique_words(caption_seqs)
	word_to_idx = {}
	idx_to_word = {}
	for i, word in enumerate(unique):
	# Start indices at 1 since 0 will represent padding
		word_to_idx[word] = i+1
		idx_to_word[i+1] = word
	
	pickle.dump(word_to_idx,open('coco_word_to_idx','wb'))
	pickle.dump(idx_to_word,open('coco_idx_to_word','wb'))
	pickle.dump(database_stats,open('coco_stats','wb'))
	print 'MS COCO dataset processed'


def unique_words(caption_seqs):
    unique = set()
    for seq in caption_seqs:
        for word in seq:
            unique.add(word)

    return list(unique)


def partial_captions_and_next_words(caption_seqs, word_to_idx, max_cap_len):
    partial_caps = []
    next_words = []
    for seq in caption_seqs:
        for i, word in enumerate(seq[:-1]):
            partial_caps.append([word_to_idx[w] for w in seq[:i+1]])
            next_words.append(word_to_idx[seq[i+1]])
        # Append the full sequence and use the stop token as the next word
        partial_caps.append([word_to_idx[w] for w in seq])
        next_words.append(STOP_TOKEN_IDX)

    # Pad sequences with 0's such that they all have length 'max_caption_len'. Note that the
    # last word of a caption will always be included in the partial caption so that we can
    # predict the stop token
    partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_cap_len, padding='post')
    return partial_caps, next_words


if __name__ == '__main__':
    # first preprocess dataset - this only needs to be done once and then the files are saved
    #preprocess_coco()

    #choose caption set
    coco_filename='../external/coco/annotations/instances_train2014.json'
    ann_filename = '../external/coco/annotations/captions_train2014.json'
    coco = COCO(coco_filename)
    coco_caps = COCO(ann_filename)

    # choose categories/images
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    annIds = coco_caps.getAnnIds(imgIds)
    anns = coco_caps.loadAnns(annIds)

    # get caption sequences
    image_ids = [ann['image_id'] for ann in anns]
    captions = [ann['caption'].encode('ascii') for ann in anns]
    caption_seqs = [text_to_word_sequence(c) for c in captions]

    # get image ids for each partial caption
    num_partials = [len(seq) for seq in caption_seqs]
    repeated_ids = [[img_id]*n for img_id,n in zip(image_ids,num_partials)]
    image_ids = [img_id for rep_id in repeated_ids for img_id in rep_id]

    word_to_idx, idx_to_word, partial_caps, next_words = preprocess_captions(caption_seqs)

    print(len(image_ids), len(partial_caps))
    assert(len(image_ids)==len(partial_caps))

    number_of_items = 50

    X = [0,0]
    X[0] = np.asarray(image_ids[:number_of_items])
    print(partial_caps.shape, "PARTIAL CAP SHAPE")
    X[1] = np.asarray(partial_caps[:number_of_items])
    print([idx_to_word[x] for x in next_words[:number_of_items]])
    y = np.asarray(next_words[:number_of_items])
    # print(X[0])

    out = X, y, word_to_idx, idx_to_word

    handle = open( "../keras_vgg_19/savedoc", "r+" )
    pickle.dump(out, handle)
    handle.close()
