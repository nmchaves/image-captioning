import json
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
from keras.preprocessing import sequence
# from keras.preprocessing.text import text_to_word_sequence
from utils.preprocessing import preprocess_captions


# def preprocess_captions(captions):
#     caption_seqs = [text_to_word_sequence(c) for c in captions]
#     unique = unique_words(caption_seqs)

#     word_to_idx = {}
#     idx_to_word = {}
#     for i, word in enumerate(unique):
#         # Start indices at 1 since 0 will represent padding
#         word_to_idx[word] = i+1
#         idx_to_word[i+1] = word

#     partial_caps, next_words = partial_captions_and_next_words(caption_seqs, word_to_idx)

#     return unique, word_to_idx, idx_to_word, partial_caps, next_words


# def unique_words(caption_seqs):
#     unique = set()
#     for seq in caption_seqs:
#         for word in seq:
#             unique.add(word)

#     return list(unique)


# def partial_captions_and_next_words(caption_seqs, word_to_idx):
#     max_caption_len = max([len(seq) for seq in caption_seqs])
#     partial_caps = []
#     next_words = []
#     for seq in caption_seqs:
#         for i, word in enumerate(seq[:-1]):
#             partial_caps.append([word_to_idx[w] for w in seq[:i+1]])
#             next_words.append(word_to_idx[seq[i+1]])

#     # Pad sequences with 0's such that they all have length 'max_caption_len-1'. We subtract 1 b/c the
#     # last word of a caption will never be included in any partial caption
#     partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_caption_len-1, padding='post')
#     return partial_caps, next_words

captions_path = '/Users/reuben/Downloads/annotations/captions_train2014.json'


captions = open(captions_path,'r')

captions = json.loads(captions.read())

# print([x for x in captions])
sentence = (captions['annotations'][0]['caption'])
number = captions['annotations'][0]['image_id']
number = ('0000000000000'+str(number))[-12:]
print(number)
image_path = '/Users/reuben/Downloads/train2014/COCO_train2014_'+number+'.jpg'
print(preprocess_captions([str(sentence)]))
