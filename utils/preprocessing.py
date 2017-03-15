import sys
sys.path.append('../external/coco/PythonAPI')
sys.path.append('../google_refexp_py_lib')
from pycocotools.coco import COCO
from refexp import Refexp

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence, image
from keras.applications.imagenet_utils import preprocess_input

import numpy as np
import cPickle as pickle
import skimage.io as io
import matplotlib.pyplot as plt
import os

STOP_TOKEN = '$STOP$'
STOP_TOKEN_IDX = 0
START_TOKEN = '$START$'
START_TOKEN_IDX = 1


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def unique_words(caption_seqs):
    unique = set()
    for seq in caption_seqs:
        for word in seq:
            unique.add(word)

    return list(unique)

# TODO: pass util path as argument
# this function has been edited to read in pickled dictionaries produced by preprocess_coco function
# also now returns just the two dictionaries, partial captions, and next words as indices (not one-hot)
# also takes caption_seqs instead of captions, which makes it easier to repeat the image_ids
def preprocess_captions(caption_seqs, word_to_idx, max_cap_len):
    partial_caps, next_words = partial_captions_and_next_words(caption_seqs, word_to_idx, max_cap_len)
    return partial_caps, next_words


def load_dicts():
    word_to_idx = pickle.load(open('../utils/coco_word_to_idx','rb'))
    idx_to_word = pickle.load(open('../utils/coco_idx_to_word','rb'))
    database_stats = pickle.load(open('../utils/coco_stats','rb'))
    return word_to_idx, idx_to_word

def load_refexp_dicts():
    word_to_idx = pickle.load(open('../utils/refexp_word_to_idx','rb'))
    idx_to_word = pickle.load(open('../utils/refexp_idx_to_word','rb'))
    database_stats = pickle.load(open('../utils/refexp_stats','rb'))
    return word_to_idx, idx_to_word

# this function creates dictionaries and finds max caption length for the entire coco dataset
def preprocess_coco(coco_dir, out_dir, include_val=True):
    # get annotations
    train_ann_filename = coco_dir+'/annotations/captions_train2014.json'
    coco_train = COCO(train_ann_filename)
    trainAnnIds = coco_train.getAnnIds()
    trainAnns = coco_train.loadAnns(trainAnnIds)

    if include_val:
        val_ann_filename = coco_dir+'/annotations/captions_val2014.json'
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

    database_stats = {'max_cap_len': max_cap_len}

    # create dictionaries for dataset
    unique = unique_words(caption_seqs)
    word_to_idx = {
        STOP_TOKEN: STOP_TOKEN_IDX,
        START_TOKEN: START_TOKEN_IDX
    }
    idx_to_word = {
        STOP_TOKEN_IDX: STOP_TOKEN,
        START_TOKEN_IDX: START_TOKEN
    }

    for i, word in enumerate(unique):
        # Start indices at 2 since 0 and 1 are reserved for start and stop tokens
        word_to_idx[word] = i+2
        idx_to_word[i+2] = word

    # Basic sanity checks
    assert(idx_to_word[word_to_idx['the']] == 'the')
    assert(word_to_idx[STOP_TOKEN] == STOP_TOKEN_IDX)
    assert(idx_to_word[STOP_TOKEN_IDX] == STOP_TOKEN)
    assert(word_to_idx[START_TOKEN] == START_TOKEN_IDX)
    assert(idx_to_word[START_TOKEN_IDX] == START_TOKEN)
    assert(word_to_idx[idx_to_word[2]] == 2)

    # Save the data
    with open(out_dir+'/coco_word_to_idx', 'w+') as handle:
        pickle.dump(word_to_idx, handle)
    with open(out_dir+'/coco_idx_to_word', 'w+') as handle:
        pickle.dump(idx_to_word, handle)
    with open(out_dir+'/coco_stats', 'w+') as handle:
        pickle.dump(database_stats, handle)

    print 'Finished processing MS COCO dataset'


def partial_captions_and_next_words(caption_seqs, word_to_idx, max_cap_len):
    partial_caps = []
    next_words = []
    for seq in caption_seqs:
        for i, word in enumerate(seq[:-1]):
            partial_caps.append([word_to_idx[w] for w in seq[:i+1]])
            next_words.append(word_to_idx[seq[i+1]])

    # Pad sequences with stop token indices such that they all have length 'max_caption_len'-1 (the -1 accounts for
    # the fact that we never predict a start token. Note that the last word of a caption will always be included in
    # the partial caption so that we can predict the stop token. The stop token should never appear in a partial caption.
    partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_cap_len-1, padding='post',value=STOP_TOKEN_IDX)
    return partial_caps, next_words

def preprocess_refexp():
    # get annotations
    coco_filename= '../external/coco/annotations/instances_train2014.json'
    refexp_filename='../google_refexp_dataset_release/google_refexp_train_201511_coco_aligned.json'

    refexp = Refexp(refexp_filename, coco_filename)
    # ignore specific captions that have weird symbols in them
    ref_ids = [ref_id for ref_id in refexp.dataset['refexps'] if ref_id not in [146,400,923,21409,35384,38589,46630,47673,65639,70715,82742]]
    captions = [refexp.dataset['refexps'][ref_id]['raw'].encode('ascii') for ref_id in ref_ids]
    caption_seqs = [text_to_word_sequence(c) for c in captions]
    max_cap_len = max([len(seq) for seq in caption_seqs])

    database_stats = {'max_cap_len': max_cap_len}

    # create dictionaries for dataset
    unique = unique_words(caption_seqs)
    word_to_idx = {
        STOP_TOKEN: STOP_TOKEN_IDX,
        START_TOKEN: START_TOKEN_IDX
    }
    idx_to_word = {
        STOP_TOKEN_IDX: STOP_TOKEN,
        START_TOKEN_IDX: START_TOKEN
    }

    for i, word in enumerate(unique):
        # Start indices at 2 since 0 and 1 are reserved for start and stop tokens
        word_to_idx[word] = i+2
        idx_to_word[i+2] = word

    # Basic sanity checks
    assert(idx_to_word[word_to_idx['the']] == 'the')
    assert(word_to_idx[STOP_TOKEN] == STOP_TOKEN_IDX)
    assert(idx_to_word[STOP_TOKEN_IDX] == STOP_TOKEN)
    assert(word_to_idx[START_TOKEN] == START_TOKEN_IDX)
    assert(idx_to_word[START_TOKEN_IDX] == START_TOKEN)
    assert(word_to_idx[idx_to_word[2]] == 2)

    # Save the data
    with open('/refexp_word_to_idx', 'w+') as handle:
        pickle.dump(word_to_idx, handle)
    with open('/refexp_idx_to_word', 'w+') as handle:
        pickle.dump(idx_to_word, handle)
    with open('/refexp_stats', 'w+') as handle:
        pickle.dump(database_stats, handle)

    print 'Finished processing refexp dataset'
    
    
# stream size is in terms of # of partial captions
def preprocess_refexp_images(stream_num, stream_size, word_to_idx, max_cap_len, coco_dir, category_names=[],
                                out_file='../keras_vgg_19/savedoc', NO_PADDING=False):
    
    coco_filename= coco_dir+'/annotations/instances_train2014.json'
    refexp_filename='../google_refexp_dataset_release/google_refexp_train_201511_coco_aligned.json'

    refexp = Refexp(refexp_filename, coco_filename)
    
    # choose categories/images
    catIds = refexp.getCatIds(catNms=category_names)
    imgIds = list(set(refexp.getImgIds(catIds=catIds)))
    annIds = refexp.getAnnIds(imgIds=imgIds)
    anns = refexp.loadAnns(ids=annIds)

    refIds = []
    bboxes = []
    refImgIds = []
    # get all refexp ids and bboxes and imageids in these annotations, except for captions with weird symbols
    for ann in anns:
        for ref_id in ann['refexp_ids']:
            if ref_id not in [146,400,923,21409,35384,38589,46630,47673,65639,70715,82742]:
                refIds.append(ref_id)
                bboxes.append(ann['bbox'])
                refImgIds.append(ann['image_id'])

    # get caption sequences, with added start and stop tokens
    captions = [refexp.dataset['refexps'][ref_id]['raw'].encode('ascii') for ref_id in refIds]
    caption_seqs = [[START_TOKEN] + text_to_word_sequence(c) + [STOP_TOKEN] for c in captions]
    caption_lengths = [len(seq) for seq in caption_seqs]

    # filter out the long captions
    refImgIds = [img_id for i, img_id in enumerate(refImgIds) if caption_lengths[i] <= max_cap_len]
    bboxes = [bbox for i, bbox in enumerate(bboxes) if caption_lengths[i] <= max_cap_len]
    caption_seqs = [seq for i, seq in enumerate(caption_seqs) if caption_lengths[i] <= max_cap_len]
    caption_lengths = [l for l in caption_lengths if l <= max_cap_len] # do not move this before the other filter steps!
    total_num_partial_captions = sum(caption_lengths)

    # repeat image id and bounding box for each partial caption
    repeated_ids = [[img_id]*(l-1) for img_id,l in zip(refImgIds,caption_lengths)]
    image_ids = [img_id for rep_id in repeated_ids for img_id in rep_id]
    repeated_bboxes = [[bbox]*(l-1) for bbox,l in zip(bboxes,caption_lengths)]
    cap_bbox = [bbox for rep_bbox in repeated_bboxes for bbox in rep_bbox]

    partial_caps, next_words = partial_captions_and_next_words(caption_seqs, word_to_idx, max_cap_len) #preprocess_captions(caption_seqs, word_to_idx, max_cap_len)

    print(len(image_ids), len(partial_caps), len(cap_bbox))
    assert(len(image_ids) == len(partial_caps))
    assert(len(image_ids) == len(cap_bbox))

    '''
    # Determine how many (partial caption, image) examples to take to obtain
    # `num_imgs_to_sample` total distinct images (including all partial captions)
    if num_caps_to_sample < total_num_images:
        number_of_items = 0
        for i, l in enumerate(caption_lengths):
            if i >= num_caps_to_sample:
                break
            number_of_items += l
    else:
        print total_num_images, ' were requested, but only ', num_caps_to_sample, \
            ' are available in this category. Processing all images in the category...'
        number_of_items = len(partial_caps)
    '''

    X = [0,0]
    # TODO: handle the case where you request indices out of range
    number_of_items = min(stream_size, total_num_partial_captions)
    item_range = range((stream_num - 1) * stream_size, stream_num * stream_size)
    X[0] = np.asarray(zip(image_ids,cap_bbox))[item_range]
    X[1] = np.asarray(partial_caps[item_range])
    y = np.asarray(next_words)[item_range]
    out = X, y

    with open(out_file, 'w') as handle:
        pickle.dump(out, handle)
       



# stream size is in terms of # of partial captions
def preprocess_captioned_images(stream_num, stream_size, word_to_idx, max_cap_len, coco_dir, category_names=[],
                                out_file='../keras_vgg_19/savedoc', NO_PADDING=False):

    coco_filename= coco_dir+'/annotations/instances_train2014.json'
    ann_filename = coco_dir+'/annotations/captions_train2014.json'

    coco = COCO(coco_filename)
    coco_caps = COCO(ann_filename)

    # choose categories/images
    catIds = coco.getCatIds(catNms=category_names)
    imgIds = list(set(coco.getImgIds(catIds=catIds)))
    annIds = coco_caps.getAnnIds(imgIds)
    anns = coco_caps.loadAnns(annIds)

    # get caption sequences. insert a start token at the beginning of each caption
    ann_image_ids = [ann['image_id'] for ann in anns]
    captions = [ann['caption'].encode('ascii') for ann in anns]
    caption_seqs = [[START_TOKEN] + text_to_word_sequence(c) + [STOP_TOKEN] for c in captions]
    caption_lengths = [len(seq) for seq in caption_seqs]

    # filter out the long captions
    ann_image_ids = [id for i, id in enumerate(ann_image_ids) if caption_lengths[i] <= max_cap_len]
    caption_seqs = [seq for i, seq in enumerate(caption_seqs) if caption_lengths[i] <= max_cap_len]
    caption_lengths = [l for l in caption_lengths if l <= max_cap_len] # do not move this before the other filter steps!
    total_num_partial_captions = sum(caption_lengths)

    # repeat an image id for each partial caption
    repeated_ids = [[img_id]*(n-1) for img_id,n in zip(ann_image_ids,caption_lengths)]
    image_ids = [img_id for rep_id in repeated_ids for img_id in rep_id]

    partial_caps, next_words = partial_captions_and_next_words(caption_seqs, word_to_idx, max_cap_len) #preprocess_captions(caption_seqs, word_to_idx, max_cap_len)

    print(len(image_ids), len(partial_caps))
    assert(len(image_ids) == len(partial_caps))

    '''
    # Determine how many (partial caption, image) examples to take to obtain
    # `num_imgs_to_sample` total distinct images (including all partial captions)
    if num_caps_to_sample < total_num_images:
        number_of_items = 0
        for i, l in enumerate(caption_lengths):
            if i >= num_caps_to_sample:
                break
            number_of_items += l
    else:
        print total_num_images, ' were requested, but only ', num_caps_to_sample, \
            ' are available in this category. Processing all images in the category...'
        number_of_items = len(partial_caps)
    '''

    X = [0,0]
    # TODO: handle the case where you request indices out of range
    number_of_items = min(stream_size, total_num_partial_captions)
    item_range = range((stream_num - 1) * stream_size, stream_num * stream_size)
    X[0] = np.asarray(image_ids)[item_range]
    X[1] = np.asarray(partial_caps[item_range])
    y = np.asarray(next_words)[item_range]
    out = X, y

    with open(out_file, 'w') as handle:
        pickle.dump(out, handle)


if __name__ == '__main__':
    # first preprocess dataset - this only needs to be done once and then the files are saved
    #preprocess_coco(coco_dir='../external/coco', out_dir='')
    preprocess_refexp()

    #preprocess_captioned_images(stream_num=1, stream_size=2, max_cap_len=20, coco_dir='../external/coco',
    #                            category_name='person', out_file='test')
