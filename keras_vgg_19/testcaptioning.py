import sys
sys.path.append('../') # needed for Azure VM to see utils directory

from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Activation, \
    Embedding, TimeDistributed, GRU, RepeatVector, Merge
from keras.applications import VGG19
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
from keras.preprocessing import sequence
import pickle
from keras_vgg19_features import predict_image
# from utils.preprocessing import preprocess_image, repeat_imgs
# refexp_filename='../google_refexp_dataset_release/google_refexp_train_201511_coco_aligned.json'
# coco_filename='../external/coco/annotations/instances_train2014.json'
# datasetDir = '../external/coco/'
# datasetType = 'images/train2014/'

#     # Create Refexp instance.
# refexp = Refexp(refexp_filename, coco_filename)

#todo
#write a function "decode_text" that takes lists of embedded words and returns words

#todo
# load the partial captions along with ids, and next words, from separate h5py files, along with max_caption_length

#todo
#go through and substitute each id for an image looked up from preprocessed directory (see vgg_run)

#todo
#X = blah
# y = blah

#train and save

#further down the line
    #write char lstm model
    #pragmatic training with referring expressions
    #consider using visual genome
    #






#put in preprocessing
def get_image(id,path):
    #possibly pad id with 0s
    return np.load(path+id+'.npy')



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

    # # ideal
    X, y, word_to_idx, idx_to_word = data
    vocab_size = len(word_to_idx)
    image_ids = X[0] #shape (batch_size,224,224,3)
    # print(image_ids,"STOP")
    # print(len(image_ids)), "LENGTH OF IMAGE_IDS"
    images = []
    # print(image_ids[0]),"IMAGE ID"
    for image_id in image_ids:
        number = str(('_0000000000000'+str(image_id))[-12:])

        try:
            x = get_image(number,path='../external/coco/processed/')
        except IOError:
            # print(str(image_id)[-6:])
            x = predict_image(str(image_id))

        images.append(x)
            # x = image.load_img(datasetDir+datasetType+"_"+number, target_size=(224, 224))
            # x = image.img_to_array(x)
            # x = np.expand_dims(x, axis=0)
            # x = preprocess_input(x)

            # output = model.predict(x)
            # images.append(x)
    # images.append(get_image('000000131074',path='../external/coco/processed/COCO_train2014_'))
    # images = [get_image('000000131074',path='../external/coco/processed/COCO_train2014_')[0]]*17

    # print(len(images),len(images[0]))

    X[0] = np.asarray(images).transpose((1,0,2))[0]
    # print(X[0].shape,"SHAPE")

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

    # initial_model = VGG19(weights="imagenet", include_top=True)
    # flat = initial_model.get_layer('flatten').output
    # flat_repeated = RepeatVector(max_caption_len)(flat)
    # image_model = Model(initial_model.inputs, flat_repeated)
    # images = 




    #MODEL
    image_model = Sequential()
    image_model.add(Dense(256, input_dim=25088))
    image_model.add(RepeatVector(max_caption_len))

    # next, let's define a RNN model that encodes sequences of words
    # into sequences of 128-dimensional word vectors.
    language_model = Sequential()
    # TODO: check if better way to handle off by 1 error with vocab_size
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


    #what is this for?
    # imgs_rep = repeat_imgs(images, captions)
    # model.fit([imgs_rep, partial_captions], next_words, batch_size=10, nb_epoch=5)

    print(X[0].shape,X[1].shape,y.shape,"SHAPETYSHAPE")

    # model.fit([X[0],X[1]],y, batch_size=10, nb_epoch=1)
    # model.save("modelweights")
    # del model
    model = load_model("modelweights")

    # print result   
    # out = sample(result[0])

    # init = np.zeros(len(captions[0])) 
    # print(word_to_idx)
    #sampling loop
    # print(list_of_words_to_caption(['', 'view', 'all', 'of', 'empty', 'of', 'of', 'empty', 'of', 'bathroom', 'decorated'],word_to_idx,len(captions[0])))
    new_image = X[0][0].reshape((1,len(X[0][0])))
    gen = []
    # cap = "the the the"
    # out = np.zeros((1,49))
    # for i,x in enumerate(cap.split()):
    #     out[0][i] = word_to_idx[x]
    # cap = out
    # cap = np.asarray([word_to_idx[x] for x in cap.split()])
    cap = np.zeros((1,49))
    while len(gen) < 10: 
        result = model.predict([new_image, cap])
        out = idx_to_word[sample(result[0])]
        gen.append(out)
        print(gen)

        
    
