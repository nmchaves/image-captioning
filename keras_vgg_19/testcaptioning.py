from keras.models import Sequential, Model
from keras.layers import Dense, Activation, \
    Embedding, TimeDistributed, GRU, RepeatVector, Merge
from keras.applications import VGG19
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
from keras.preprocessing import sequence
import pickle

with open('savedoc', 'rb') as handle:
    data = pickle.load(handle)

# img = data[0][0][0]
# cap = data[0][0][1][0].reshape(1,16)
# vocab_size = data[2]
# y = np.zeros((1,vocab_size))
# y[0][data[1][0]] = 1

# print(y.shape)

# print(cap.shape)

# # ideal
X,y,vocab_size,idx_to_word = data  # ideally, Xand y should be tensors, right? well, X should be two tensors
images = X[0] #shape (batch_size,224,224,3)
captions = X[1] # (batch_size,16)
next_words = y # vocab_size 

# print(captions),"CAPTIONS"
# image = images[0]
# caption = captions[0]
# next_word = next_words[0]

# print(images.shape,captions.shape)

# vocab_size = 7

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def preprocess_img_and_text(images, sentences):
    images = [preprocess_image(image) for image in images]
    images = np.asarray(images)


    words = [s.split() for s in sentences]
    unique = []
    for word in words:
        unique.extend(word)
    unique = list(set(unique))
    word_index = {}
    index_word = {}
    for i,word in enumerate(unique):
        word_index[word] = i
        index_word[i] = word

    partial_captions = []
    for s in sentences:
        one = [word_index[txt] for txt in s.split()]
        partial_captions.append(one)

    partial_captions = sequence.pad_sequences(partial_captions, maxlen=max_caption_len,padding='post')
    next_words = np.zeros((1,vocab_size))
    for i,s in enumerate(sentences):
        text = s.split()
        x = [word_index[txt] for txt in text]
        x = np.asarray(x)
        next_words[i,x] = 1

    return images, partial_captions, next_words


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

if __name__ == '__main__':
    max_caption_len = 10
    vocab_size = vocab_size #10000

    initial_model = VGG19(weights="imagenet", include_top=True)
    flat = initial_model.get_layer('flatten').output
    flat_repeated = RepeatVector(max_caption_len)(flat)
    image_model = Model(initial_model.inputs, flat_repeated)

    # next, let's define a RNN model that encodes sequences of words
    # into sequences of 128-dimensional word vectors.
    language_model = Sequential()
    language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
    language_model.add(GRU(output_dim=128, return_sequences=True))
    language_model.add(TimeDistributed(Dense(128)))

    # let's repeat the image vector to turn it into a sequence.
    #image_model.add(RepeatVector(max_caption_len))

    # the output of both models will be tensors of shape (samples, max_caption_len, 128).
    # let's concatenate these 2 vector sequences.
    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
    #model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
    # let's encode this vector sequence into a single vector
    model.add(GRU(256, return_sequences=False))
    # which will be used to compute a probability
    # distribution over what the next word in the caption should be!
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # sentences = ["START A cat with white fur END"]
    # new_image_path = ['cat.jpg']
    # images_2, partial_captions, next_words = preprocess_img_and_text(images_path, sentences)

    new_image = preprocess_image('cat.jpg')

    # print "next",next_words.shape
    # print "shape partial_captions", partial_captions.shape
    # print "partial captions 1",partial_captions[0]
    # print image.shape
    # print "shape of captions",captions.shape

    # "images" is a numpy float array of shape (nb_samples, nb_channels=3, width, height).
    # "captions" is a numpy integer array of shape (nb_samples, max_caption_len)
    # containing word index sequences representing partial captions.
    # "next_words" is a numpy float array of shape (nb_samples, vocab_size)
    # containing a categorical encoding (0s and 1s) of the next word in the corresponding
    # partial caption.
    # model.fit([np.zeros((17,224,224,3)),np.zeros((17,10))], np.zeros((17,15)), batch_size=2, nb_epoch=5)
    # print captions.shape

    captions = np.zeros((17,10))
    captions[0][0] = 10
    model.fit(X, y, batch_size=10, nb_epoch=5)
    result = model.predict(new_image,X[1][0].reshape([1,10]) ])
    # print result   
    out = sample(result[0])
    print(idx_to_word[out])