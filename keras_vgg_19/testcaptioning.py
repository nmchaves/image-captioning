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
from utils.preprocessing import preprocess_captioned_images, STOP_TOKEN
import argparse
from cnn_preprocessing import predict_image


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

    default_num_imgs = 50
    data_path = 'preprocess_data'
    coco_dir = '../external/coco'

    # Parse program arguments
    parser = argparse.ArgumentParser(description='Preprocess image captions if necessary.')
    parser.add_argument("-p", "--preprocess", default=False,
                        type=bool, help='If true, apply preprocessing')
    parser.add_argument("-n", "--num_imgs", default=default_num_imgs,
                        type=int, help='Number of images to preprocess')
    parser.add_argument("-t", "--train", default=False,
                        type=bool, help='If true, train the model. Else, load the saved model')

    args = parser.parse_args()
    train = args.train
    num_imgs = args.num_imgs

    # Preprocess the data if necessary
    if args.preprocess:
        preprocess_captioned_images(num_imgs_to_sample=num_imgs, coco_dir=coco_dir,
                                    category_name='person', out_file=data_path)

    with open(data_path, 'rb') as handle:
        data = pickle.load(handle)

    X, y, word_to_idx, idx_to_word = data
    vocab_size = len(word_to_idx)
    image_ids = X[0] #shape (batch_size,224,224,3)

    # print([idx_to_word[n] if n!=0 else "null" for n in y],"NEXT_WORDS")
    # print([idx_to_word[x] if x!=0 else "null" for x in X[1][0]])
    # print(idx_to_word[y[0]])

    images = []
    for image_id in image_ids:
        number = str(('_0000000000000'+str(image_id))[-12:])

        try:
            x = get_image(number,path=coco_dir+'/processed/')
        except IOError:
            x = predict_image(str(image_id))

        images.append(x)


    X[0] = np.asarray(images).transpose((1,0,2))[0]
    partial_captions = X[1]
    max_caption_len = partial_captions.shape[1]

    next_words = y # vocab_size
    new_next_words = []
    for x in next_words:
        a = np.zeros(vocab_size)
        a[x] = 1
        new_next_words.append(a)
    next_words = np.asarray(new_next_words)
    y = next_words

    # Model
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

    if args.train:
        model.fit([X[0],X[1]],y, batch_size=10, nb_epoch=5)
        model.save("modelweights")
    else:
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
    cap = ["vegetables"]
    while len(cap) < max_caption_len:
        result = model.predict([new_image, words_to_caption(cap,word_to_idx,max_caption_len)])
        m = max(result[0])
        # print(result)
        out = idx_to_word[[i for i, j in enumerate(result[0]) if j == m][0]]
        # out = idx_to_word[sample(result[0])]
        cap.append(out)
        print(cap)
        # if out == STOP_TOKEN:
        #     break
