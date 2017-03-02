import sys
sys.path.append('../') # needed for Azure VM to see utils directory

from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Activation, \
    Embedding, TimeDistributed, GRU, RepeatVector, Merge
from keras.applications import VGG19
# from keras.preprocessing.text import text_to_word_sequence
# from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
# from keras.preprocessing import sequence
import pickle
from keras_vgg19_features import predict_image
# from utils.general import *
# from utils 
# from utils.preprocessing import preprocess_image, repeat_imgs
# refexp_filename='../google_refexp_dataset_release/google_refexp_train_201511_coco_aligned.json'
# coco_filename='../external/coco/annotations/instances_train2014.json'
# datasetDir = '../external/coco/'
# datasetType = 'images/train2014/'

#     # Create Refexp instance.
# refexp = Refexp(refexp_filename, coco_filename)



#put in preprocessing
def get_image(id,path):
    #possibly pad id with 0s
    return np.load(path+id+'.npy')

def words_to_caption(cap,word_to_idx):
    out = np.zeros((1,49))
    if cap != []:
        for i,x in enumerate(cap):
            out[0][i] = word_to_idx[x]
    return out



def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

if __name__ == '__main__':

    with open('savedoc', 'rb') as handle:
        data = pickle.load(handle)

    X, y, word_to_idx, idx_to_word = data
    vocab_size = len(word_to_idx)
    image_ids = X[0] #shape (batch_size,224,224,3)
    
    print("SANITY CHECK",idx_to_word[word_to_idx['the']],word_to_idx[idx_to_word[1]])

    print([idx_to_word[x] if x!=0 else "null" for x in X[1][0]])
    print(idx_to_word[y[0]])

    images = []
    for image_id in image_ids:
        number = str(('_0000000000000'+str(image_id))[-12:])

        try:
            x = get_image(number,path='../external/coco/processed/')
        except IOError:
            x = predict_image(str(image_id))

        images.append(x)


    X[0] = np.asarray(images).transpose((1,0,2))[0]
    partial_captions = X[1] # (batch_size,16)
    next_words = y # vocab_size
    new_next_words = []
    for x in next_words:
      # print x
      a = np.zeros(vocab_size)
      a[x-1] = 1
      new_next_words.append(a)
    next_words = np.asarray(new_next_words)
    y = next_words

    max_caption_len = partial_captions.shape[1]

    #MODEL
    image_model = Sequential()
    image_model.add(Dense(128, input_dim=25088))


    language_model = Sequential()
    language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
    language_model.add(GRU(output_dim=128, return_sequences=True))
    language_model.add(TimeDistributed(Dense(128)))

    image_model.add(RepeatVector(max_caption_len))

    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
    model.add(GRU(256, return_sequences=False))

    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')



    print(X[0].shape,X[1].shape,y.shape,"SHAPES")

    #view X[:5]
    # print(X[1])
    # for y in X[1]:
    print(X[1].shape)
    for i,n in enumerate(X[1]):
        print([idx_to_word[x] if x!=0 else "null" for x in n],"FIRST CAPTION")
        out = np.argmax(y[i])
        print(idx_to_word[out],"NEXT WORD")
    # print(X[1].shape)
    # for m in y:

    model.fit([X[0],X[1]],y, batch_size=10, nb_epoch=10)
    model.save("modelweights")
    # model = load_model("modelweights")



    new_image = X[0][0].reshape((1,len(X[0][1])))
    new_image = np.zeros(shape=new_image.shape)


    cap = "everything the".split()
    # cap = 
    print(words_to_caption(cap,word_to_idx))
    result = model.predict([new_image, words_to_caption(cap,word_to_idx)])
    print(result[0][np.argmax(result[0])],"PROB DIST")
    print(result.shape)
    result = idx_to_word[np.argmax(result[0])]
    print(result)

    # while len(cap) < 10: 
    #     result = model.predict([new_image, words_to_caption(cap,word_to_idx)])
    #     m = max(result[0])
    #     # print(result)
    #     out = idx_to_word[[i for i, j in enumerate(result[0]) if j == m][0]]
    #     # out = idx_to_word[sample(result[0])]
    #     cap.append(out)
    #     print(cap)


        
    
