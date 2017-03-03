import sys
sys.path.append('../') # needed for Azure VM to see utils directory

from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Activation, \
    Embedding, TimeDistributed, GRU, RepeatVector, Merge
from keras.applications import VGG19
# from keras.preprocessing.text import text_to_word_sequence
# from keras.preprocessing import image
import numpy as np
import pickle
from utils.preprocessing import preprocess_captioned_images
import argparse
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

    default_num_images = 50
    data_path = 'savedoc'

    parser = argparse.ArgumentParser(description='Preprocess image captions if necessary.')
    parser.add_argument("-p", "--preprocess", default=False,
                        type=bool, help='whether to use preprocessing')
    parser.add_argument("-n", "--num_images", default=default_num_images,
                        type=int, help='number of images to preprocess')
    args = parser.parse_args()
    num_images = args.num_images
    if args.preprocess:
        preprocess_captioned_images(number_of_items=num_images, category_name='person', result_path=data_path)

    with open(data_path, 'rb') as handle:
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
    image_model = Sequential()
    image_model.add(Dense(128, input_dim=25088))


    language_model = Sequential()
    language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
    language_model.add(GRU(output_dim=128, return_sequences=True))
    language_model.add(TimeDistributed(Dense(128),name="lang"))

    image_model.add(RepeatVector(max_caption_len))

    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1,name='foo'))
    model.add(GRU(256, return_sequences=False))

    model.add(Dense(vocab_size))
    model.add(Activation('softmax',name='soft'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')




    # for i,n in enumerate(X[1]):
    #     # print([idx_to_word[x] if x!=0 else "null" for x in n],"FIRST CAPTION")
    #     out = np.argmax(y[i])
    #     if out!=0:
    #         print(idx_to_word[out],"NEXT WORD")
    # print(X[1].shape)
    # for m in y:

    # model.fit([X[0],X[1]],y, batch_size=10, nb_epoch=10)
    # model.save("modelweights")
    model = load_model("modelweights")
    # intermediate_layer_model = Model(input=model.input,
    #                              output=model.get_layer("soft").output)
    # # new = "LAYER",model.get_layer(name='lang').output
    # intermediate_output = intermediate_layer_model.predict([new_image,np.zeros((1,50))])

    # intermediate_output2 = intermediate_layer_model.predict([new_image,np.ones((1,50))])

    # print(intermediate_output)
    # print(intermediate_output2)
    # print(np.array_equal(intermediate_output,intermediate_output2))




    # cap = "vegetables market".split()
    # # cap = 
    # # print(words_to_caption(cap,word_to_idx, max_caption_len))
    # result = model.predict([new_image, words_to_caption(cap,word_to_idx, max_caption_len)])
    # print(result[0][np.argmax(result[0])],"PROB DIST")
    # result = idx_to_word[np.argmax(result[0])]
    # print(result)



    # # cap = "piles crowded".split()
    # new_image = X[0][0].reshape((1,len(X[0][1])))
    # # new_image = np.zeros(shape=new_image.shape)
    # # cap = "vegetables market".split()
    # cap = "carrots and potatoes at a crowded outdoor market carrots and potatoes at a crowded outdoor market carrots and potatoes at a crowded outdoor market".split()
    # # inp = np.zeros((1,50))

    # # # cap = 

    # # # print(words_to_caption(cap,word_to_idx))
    # # result = model.predict([new_image, inp])
    # result2 = model.predict([new_image, words_to_caption(cap,word_to_idx, max_caption_len)])
    # # print("EQUAL?",np.array_equal(result,result2))
    # print(result[0][np.argmax(result[0])],"PROB DIST")
    # result = idx_to_word[np.argmax(result[0])]
    # # print(result)

    new_image = X[0][0].reshape((1,len(X[0][1])))
    cap = []
    while len(cap) < max_caption_len:
        result = model.predict([new_image, words_to_caption(cap,word_to_idx,max_caption_len)])
        m = max(result[0])
        # print(result)
        out = idx_to_word[[i for i, j in enumerate(result[0]) if j == m][0]]
        # out = idx_to_word[sample(result[0])]
        cap.append(out)
        print(cap)


        
    
