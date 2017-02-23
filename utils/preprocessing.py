from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def repeat_imgs(imgs, captions):
    # TODO: don't recompute caption_seqs! Need to improve API
    caption_seqs = [text_to_word_sequence(c) for c in captions]
    partial_cap_lengths = [len(seq)-1 for seq in caption_seqs]
    imgs_rep = np.zeros((sum(partial_cap_lengths), 224, 224, 3))
    start = 0
    for i, pc_len in enumerate(partial_cap_lengths):
        imgs_rep[start:start+pc_len, :] = imgs[i]
        start += pc_len
    return np.asarray(imgs_rep)


def preprocess_captions(captions):
    caption_seqs = [text_to_word_sequence(c) for c in captions]
    unique = unique_words(caption_seqs)

    word_to_idx = {}
    idx_to_word = {}
    for i, word in enumerate(unique):
        # Start indices at 1 since 0 will represent padding
        word_to_idx[word] = i+1
        idx_to_word[i+1] = word

    partial_caps, next_words = partial_captions_and_next_words(caption_seqs, word_to_idx)

    # Convert next_words into a binary matrix. Use `vocab_size`+1 columns so that we can
    # use regular indexing to access the 1-based words.
    vocab_size = len(unique)
    next_words_binary = np.zeros((len(partial_caps), vocab_size+1))
    for i,nxt in enumerate(next_words):
        next_words_binary[i,nxt] = 1.0

    return unique, word_to_idx, idx_to_word, partial_caps, next_words_binary


def unique_words(caption_seqs):
    unique = set()
    for seq in caption_seqs:
        for word in seq:
            unique.add(word)

    return list(unique)


def partial_captions_and_next_words(caption_seqs, word_to_idx):
    max_caption_len = max([len(seq) for seq in caption_seqs])
    partial_caps = []
    next_words = []
    for seq in caption_seqs:
        for i, word in enumerate(seq[:-1]):
            partial_caps.append([word_to_idx[w] for w in seq[:i+1]])
            next_words.append(word_to_idx[seq[i+1]])

    # Pad sequences with 0's such that they all have length 'max_caption_len-1'. We subtract 1 b/c the
    # last word of a caption will never be included in any partial caption
    partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_caption_len-1, padding='post')
    return partial_caps, next_words


def get_captions():
    return 0

if __name__ == '__main__':
    captions = ['a cat with fur', 'the dog has teeth']
    unique, word_to_idx, idx_to_word, partial_caps, next_words = preprocess_captions(captions)
    print 'Partial Captions: \n', partial_caps
    print 'Next Words: ', next_words
    print 'Words indices: ', idx_to_word