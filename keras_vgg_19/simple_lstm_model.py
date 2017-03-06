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
import copy

from cnn_preprocessing import predict_image
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


def words_to_caption(cap, word_to_idx, max_caption_len):
    out = np.zeros((1,max_caption_len))
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

    # print([idx_to_word[n] if n!=0 else "null" for n in y],"NEXT_WORDS")

    
    # print("SANITY CHECK",idx_to_word[word_to_idx['the']],word_to_idx[idx_to_word[1]])

    # print([idx_to_word[x] if x!=0 else "null" for x in X[1][0]])
    # print(idx_to_word[y[0]])

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
    max_caption_len = partial_captions.shape[1]

    next_words = y # vocab_size
    new_next_words = []
    for x in next_words:
      # print x
      a = np.zeros(vocab_size)
      a[x] = 1
      new_next_words.append(a)
    next_words = np.asarray(new_next_words)
    y = next_words

    #MODEL
    # image_model = Sequential()
    # image_model.add(Dense(128, input_dim=25088))


    language_model = Sequential()
    language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
    language_model.add(GRU(output_dim=128, return_sequences=False))
    # language_model.add(TimeDistributed(Dense(128),name="lang"))

    # language_model.add(GRU(256, return_sequences=False,name='thing'))

    language_model.add(Dense(vocab_size,name='thing2'))
    language_model.add(Activation('softmax',name='soft'))

    language_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')




    # for i,n in enumerate(X[1]):
    #     # print([idx_to_word[x] if x!=0 else "null" for x in n],"FIRST CAPTION")
    #     out = np.argmax(y[i])
    #     if out!=0:
    #         print(idx_to_word[out],"NEXT WORD")
    # print(X[1].shape)
    # for m in y:

    # language_model.fit(X[1],y, batch_size=10, nb_epoch=60)
    # language_model.save("lstmmodelweights")
    language_model = load_model("lstmmodelweights")

    new_image = X[0][0].reshape((1,len(X[0][1])))

    #FIRST TEST

    # intermediate_layer_model = Model(input=language_model.input,
    #                              output=language_model.get_layer("soft").output)
    # # cap = "vegetables market".split()
    # intermediate_output = intermediate_layer_model.predict(np.zeros((1,50)))

    # # cap = "carrots and potatoes at a crowded outdoor market carrots and potatoes at a crowded outdoor market carrots and potatoes at a crowded outdoor market".split()
    # intermediate_output2 = intermediate_layer_model.predict(np.ones((1,50)))

    # print(np.array_equal(intermediate_output,intermediate_output2),"INTER OUT1")


    #SECOND TEST


    # cap = "vegetables market".split()
    cap = "piles crowded piles crowded piles crowded piles crowded piles crowded piles crowded".split()
    processed_cap = words_to_caption(cap,word_to_idx, max_caption_len)
    print(processed_cap)
    result = language_model.predict(processed_cap)
    print(result[0][np.argmax(result[0])],"PROB DIST",np.argmax(result[0]))
    print(idx_to_word[np.argmax(result[0])])


    cap2 = "vegetables market".split()
    # new_image2 = X[0][0].reshape((1,len(X[0][1])))
    processed_cap2 = words_to_caption(cap2,word_to_idx, max_caption_len)
    print(processed_cap2)
    # cap2 = "piles crowded piles crowded piles crowded piles crowded piles crowded piles crowded".split()
    result2 = language_model.predict(processed_cap2)
    print(result2[0][np.argmax(result2[0])],"PROB DIST",np.argmax(result2[0]))
    print(idx_to_word[np.argmax(result2[0])])

    print(np.array_equal(result,result2),"FINAL OUT")
    # # print(intermediate_output)
    # # print(intermediate_output2)
    # # print(np.array_equal(intermediate_output,intermediate_output2),"OUT1")

    new_image = X[0][0].reshape((1,len(X[0][0])))
    result = []
    cap = []
    while len(cap) < max_caption_len:
        old_result = copy.deepcopy(result)
        result = language_model.predict(words_to_caption(cap,word_to_idx,max_caption_len))
        print("EQUAL?",np.array_equal(old_result,result))
        m = max(result[0])
        # print(result)
        out = idx_to_word[[i for i, j in enumerate(result[0]) if j == m][0]]
        # out = idx_to_word[sample(result[0])]
        cap.append(out)
        print(cap)


    # cap = "piles crowded piles crowded piles crowded piles crowded piles crowded piles crowded".split()
    # print(result[0][np.argmax(result[0])],"PROB DIST")
    # cap = "vegetables market".split()
    # new_image = X[0][0].reshape((1,len(X[0][48])))
    # result2 = model.predict([new_image, words_to_caption(cap,word_to_idx, max_caption_len)])
    # print(result2[0][np.argmax(result2[0])],"PROB DIST",np.argmax(result2[0]))
    # result3 = model.predict([new_image, np.zeros((1,50))])
    # print(result3[0][np.argmax(result3[0])],"PROB DIST",np.argmax(result3[0]))

    # print(np.array_equal(result,result3),"OUT2")

    #checks: try training, try different images and full vs top equal

    # result = model.predict([new_image, words_to_caption(cap,word_to_idx, max_caption_len)])

    
    # # # print(words_to_caption(cap,word_to_idx, max_caption_len))
    # result2 = model.predict([new_image, words_to_caption(cap,word_to_idx, max_caption_len)])
    # print(np.array_equal(result,result2))



    # print(result[0][np.argmax(result[0])],"PROB DIST")
    # result = idx_to_word[np.argmax(result[0])]
    # print(result)



    # # new_image = np.zeros(shape=new_image.shape)
    # # cap = "vegetables market".split()
    # # inp = np.zeros((1,50))

    # # # cap = 

    # # # print(words_to_caption(cap,word_to_idx))
    # # result = model.predict([new_image, inp])
    # # print("EQUAL?",np.array_equal(result,result2))
    # result = idx_to_word[np.argmax(result[0])]
    # # print(result)

    # new_image = X[0][0].reshape((1,len(X[0][1])))
    # result = []
    # cap = []
    # while len(cap) < max_caption_len:
    #     old_result = copy.deepcopy(result)
    #     result = model.predict([new_image, words_to_caption(cap,word_to_idx,max_caption_len)])
    #     print("EQUAL?",np.array_equal(old_result,result))
    #     m = max(result[0])
    #     # print(result)
    #     out = idx_to_word[[i for i, j in enumerate(result[0]) if j == m][0]]
    #     # out = idx_to_word[sample(result[0])]
    #     cap.append(out)
        # print(cap)


        
    
