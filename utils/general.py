import sys
sys.path.append('../external/coco/PythonAPI')
sys.path.append('../google_refexp_py_lib')
from pycocotools.coco import COCO
import numpy as np


def get_original_imgs(img_ids, coco):
    if coco is None:
        coco_filename = '../external/coco/annotations/instances_train2014.json'
        coco = COCO(coco_filename)

    return coco.loadImgs(img_ids)


def indices_to_captions(caption_indices, idx_to_word):
    captions = []
    n_captions = caption_indices.shape[0]
    for i in range(n_captions):
        cap_list = [idx_to_word[idx] for idx in caption_indices[i,:]]
        cap = ' '.join(cap_list)
        captions.append(cap)
    return captions


def next_word_idx_to_word(next_word_one_hot, idx_to_word):
    word_idx = np.nonzero(next_word_one_hot)[0]
    return idx_to_word[word_idx]