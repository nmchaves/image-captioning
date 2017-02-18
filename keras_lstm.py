# import sys
# sys.path.append('../fchollet-deep-learning-models')

# import numpy as np
# from imagenet_utils import decode_predictions, preprocess_input

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import TimeDistributed
from keras.layers import GRU
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()[:1000]
# text = "hello etc etc etc "
print('corpus length:', len(text))

img = np.load("keras_vgg_19/block4_pool_features.npy")
word_text = text.split(' ')

print(word_text)

words = sorted(list(set(word_text)))
word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

vocab_length = len(words)

maxlen = 40
step = 3
sentences = []
next_words = []
for i in range(0, len(word_text) - maxlen, step):
    sentences.append(word_text[i: i + maxlen])
    next_words.append(word_text[i + maxlen])
print('nb sequences:', len(sentences))

# print(sentences)

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, vocab_length), dtype=np.bool)
y = np.zeros((len(sentences), vocab_length), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1

# print("X",X,"y",y)

# language_model = Sequential()
# language_model.add(Embedding(128, 256, input_length=vocab_length))
# language_model.add(GRU(output_dim=128, return_sequences=False))
# # language_model.add(TimeDistributed(Dense(vocab_length)))
# language_model.add(Dense(vocab_length))
# language_model.add(Activation('softmax'))

language_model = Sequential()
# language_model.add(Embedding(128, 256, input_length=vocab_length))
language_model.add(GRU(output_dim=128,input_dim=100352, return_sequences=True))
# language_model.add(TimeDistributed(Dense(vocab_length)))
language_model.add(TimeDistributed(Dense(vocab_length)))
language_model.add(Activation('softmax'))





optimizer = RMSprop(lr=0.01)
language_model.compile(loss='categorical_crossentropy', optimizer=optimizer)
# # language_model.add(TimeDistributed(Dense(128)))

def vectorization(x):
    out = np.zeros((30,maxlen))
    for t, y in enumerate(x):
        for i,z in enumerate(y):
            w = word_indices[word]
            out[t, i] = w
    # out[0,word_indices[word]] = 1
    return out

a = "the the the".split(' ')
# print(vectorization(a))


# print(np.ndarray.reshape(img,(40,128)).shape)
# what i want to do: have a forward prop that takes a tensor of shape (batch_size,caption_length,image_size) and returns a tensor of
# (batch_size,caption_length,word)

# X = np.zeros((30,109))
# y = np.zeros((30,109))

# language_model.fit(X, y, batch_size=128, nb_epoch=1)

b= np.zeros((30,40,100352))
img = np.ndarray.flatten(img)
c = np.stack([img]*maxlen)

pred = (language_model.predict(b))

# # print(pred.shape())
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
# print(indices_word[sample(pred[0])])
# print(pred)

# print(sample(pred))
# chars = sorted(list(set(text)))
# print('total chars:', len(chars))
# char_indices = dict((c, i) for i, c in enumerate(chars))
# indices_char = dict((i, c) for i, c in enumerate(chars))

# # cut the text in semi-redundant sequences of maxlen characters
# maxlen = 40
# step = 3
# sentences = []
# next_chars = []
# for i in range(0, len(text) - maxlen, step):
#     sentences.append(text[i: i + maxlen])
#     next_chars.append(text[i + maxlen])
# print('nb sequences:', len(sentences))

# # print(sentences)

# print('Vectorization...')
# X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
# y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
# for i, sentence in enumerate(sentences):
#     for t, char in enumerate(sentence):
#         X[i, t, char_indices[char]] = 1
#     y[i, char_indices[next_chars[i]]] = 1

# # print("X",X,"y",y)


# # build the model: a single LSTM
# print('Build model...')
# model = Sequential()
# model.add(LSTM(128, input_shape=(maxlen, len(chars)),return_sequences=False))
# model.add(Dense(len(chars)))
# model.add(Activation('softmax'))

# optimizer = RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# def sample(preds, temperature=1.0):
#     # helper function to sample an index from a probability array
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)



# # train the model, output generated text after each iteration

# # print(model.predict(np.zeros(1,maxlen,len(chars))))

# # for iteration in range(1):
# #     # print()
# #     # print('-' * 50)
# #     # print('Iteration', iteration)
# #     # model.fit(X, y, batch_size=128, nb_epoch=1)

# #     # start_index = random.randint(0, len(text) - maxlen - 1)

# #     # for diversity in [0.2, 0.5, 1.0, 1.2]:
# #     #     print()
# #     #     print('----- diversity:', diversity)

# #     #     generated = ''
# #     #     sentence = text[start_index: start_index + maxlen]
# #     #     generated += sentence
# #     #     print('----- Generating with seed: "' + sentence + '"')
# #     #     sys.stdout.write(generated)

# #         # for i in range(400):
# #     x = np.zeros((1, maxlen, len(chars)))
# #     # for t, char in enumerate(sentence):
# #     #     x[0, t, char_indices[char]] = 1.

# #     preds = model.predict(x, verbose=0)[0]
# #     next_index = sample(preds)
# #     next_char = indices_char[next_index]
# #     print(next_char)
#     # generated += next_char
#     # sentence = sentence[1:] + next_char

#     # sys.stdout.write(next_char)
#     # sys.stdout.flush()
#     # print(preds)
#         # print()