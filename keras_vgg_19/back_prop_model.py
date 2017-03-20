import sys
sys.path.append('../') # needed for Azure VM to see utils directory

import tensorflow as tf
from keras.regularizers import l2, activity_l2
from keras.callbacks import EarlyStopping,ReduceLROnPlateau, CSVLogger
from os import path, makedirs, listdir, remove

from keras.optimizers import Adam,SGD
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Activation, \
    Embedding, TimeDistributed, GRU, RepeatVector, Merge, Masking, LSTM, Dropout, Input, merge
from keras.applications import VGG19
# from keras.preprocessing.text import text_to_word_sequence
# from keras.preprocessing import image
import numpy as np
import pickle
from utils.preprocessing import preprocess_captioned_images, load_dicts, load_refexp_dicts, STOP_TOKEN, preprocess_refexp_images
import argparse
from cnn_preprocessing import predict_image
import copy
import tensorflow as tf

csv_logger = CSVLogger('training.log')

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                  patience=0, min_lr=0.0001)

# todo: keras streaming, variable length sequence, dynamic data
#put in preprocessing
def get_image(id,path):
    #possibly pad id with 0s
    return np.load(path+id+'.npy')


def words_to_caption(cap, word_to_idx, max_caption_len):
    out = np.zeros((1,max_caption_len-1))
    if cap != []:
        for i,x in enumerate(cap):
            out[0][i] = word_to_idx[x]
    return out

def switch(prob):
    idxs = np.argsort(prob)
    new_prob = copy.deepcopy(prob)
    new_prob[idxs[0]] = new_prob[idxs[1]]
    new_prob[idxs[1]] = new_prob[idxs[0]]
    return new_prob

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# function takes probabilites for all images - first row is for target image
# outputs new probabilites for target image, relative to other images
# i'm not sure this is a good method to use...
#def relative_probs(all_preds):
    #all_preds = np.asarray(all_preds).astype('float64')
    #total_preds = np.sum(all_preds,axis=0)
    # division by zero
    
 #   return tf.sub(tf.log(all_preds[0]),tf.log(all_preds[1]))
    #return np.divide(all_preds[0],total_preds)


def load_stream(stream_num, stream_size, preprocess, max_caption_len, word_to_idx):
    # Preprocess the data if necessary
    if preprocess:
        preprocess_refexp_images(stream_num=stream_num, stream_size=stream_size, word_to_idx=word_to_idx,
                                    max_cap_len=max_caption_len, coco_dir=coco_dir,
                                    out_file=data_path)

    with open(data_path, 'rb') as handle:
        X, next_words = pickle.load(handle)

    if False:

        new_X1 = []
        new_X0 = []
        new_y = []
        for i,x in enumerate(X[1]):
            if x[-5] == 0 and x[-6] != 0:
                new_X1.append(x[:-5])
                new_X0.append(X[0][i])
                new_y.append(next_words[i])
        next_words = np.asarray(new_y)
        X[0] = np.asarray(new_X0)
        X[1] = np.asarray(new_X1)

    print([([idx_to_word[x] for x in X[1][p]],idx_to_word[next_words[p]]) for (p,q) in enumerate(X[1][:50])])

    image_ids = X[0]
    #bounding_boxes = [x[1] for x in image_ids]
    #image_ids = [x[0] for x in image_ids]
    partial_captions = X[1]
    vocab_size = len(word_to_idx)



    # Load the CNN feature representation of each image
    images = []
    classes = []
    alt_images = []
    alt_classes = []
    for image_id in image_ids:
        number = str(('_0000000000000'+str(image_id[0]))[-12:])

        try:
            x_whole = get_image(number,path='/extra'+'/processed_flatten/')
            class_whole = get_image(number,path='/extra'+'/processed_predictions/')
            x_region = get_image(number,path='/extra'+'/processed_flatten_b/')
            class_region = get_image(number,path='/extra'+'/processed_predictions_b/')

            images.append(x_region)
            classes.append(class_region)
            alt_images.append(x_whole)
            alt_classes.append(class_whole)
        except IOError:
            x = predict_image(image_id[0],image_id[1])
            images.append(x[4])
            classes.append(x[5])
            alt_images.append(x[1])
            alt_classes.append(x[2])

    images = np.asarray(images).transpose((1,0,2))[0]
    alt_images = np.asarray(alt_images).transpose((1,0,2))[0]
    classes = np.asarray(classes).transpose((1,0,2))[0]
    alt_classes = np.asarray(alt_classes).transpose((1,0,2))[0]
    


    # Convert next words to one hot vectors
    new_next_words = []
    for x in next_words:
        a = np.zeros(vocab_size)
        a[x] = 1
        new_next_words.append(a)
    next_words_one_hot = np.asarray(new_next_words)

    return classes, images, alt_classes, alt_images, partial_captions, next_words_one_hot, \
        vocab_size, idx_to_word, word_to_idx


# Get all the model weight files in the directory that contains the model weights
def get_saved_model_files(model_weights_dir):
    return [s for s in listdir(model_weights_dir) if path.isfile(path.join(model_weights_dir, s)) and is_saved_model_file(s)]


def is_saved_model_file(fname):
    fname_split = fname.split('_')
    return fname_split[0:2] == ['modelweights', 'stream']


def largest_stream_index(model_filenames):
    return int(sorted(model_filenames, key=lambda ss: int(ss.split('_')[-1]), reverse=True)[0][-1])


def load_last_saved_model(model_weights_dir):
    saved_models = get_saved_model_files(model_weights_dir)

    # Return the model with the largest stream index (the index should be after the last '_' of the filename)
    #last_model_fname = largest_stream_index(saved_models)
    last_model_fname = 'modelweights_stream_' + str(largest_stream_index(saved_models))
    print(model_weights_dir + '/' + last_model_fname)
    return load_model(model_weights_dir + '/' + last_model_fname)


def configure_model_weights_dir(model_weights_dir, train):
    if not path.exists(model_weights_dir):
        if not train:
            print 'The directory ', model_weights_dir, ' does not exist, so there are no models to load. Either ' \
                                        'set -t to True or specify an existing directory that contains model weights.'
            exit(1)

        # Create the directory for saving model weights after training
        makedirs(model_weights_dir)
    elif train:
        # Directory already exists, but we need to store the new trained models there and overwrite the old models

        # TODO: Check that it's ok to delete this directory
        # e.g. make sure that the dir is not a parent dir

        saved_models = get_saved_model_files(model_weights_dir)
        if len(saved_models) > 0:
            # At least 1 saved model file already exists in this dir
            most_recent_stream_num = largest_stream_index(saved_models)

            print 'A directory named', model_weights_dir, 'already exists, and it contains a model file.\n' \
                'The last saved stream # was', str(most_recent_stream_num) + '\n' \
                'Would you like to continue training where you left off? (Y/N): '

            use_existing_stream = raw_input()
            if use_existing_stream == 'Y':
                return most_recent_stream_num
            else:
                # User wants to start fresh. Make sure that the user agrees to delete the old model files
                print 'In order to retrain from the beginning, we must delete the previously saved model(s) ' \
                    'in this directory. Do you want to delete these old model(s)? (Y/N):'

                user_del_response = raw_input()
                if user_del_response == 'Y':
                    print 'Deleting old model(s) in the directory ', model_weights_dir
                    for sm in saved_models:
                        remove(path.join(model_weights_dir, sm))
                else:
                    print 'Exiting. Please run again with a different directory for the model weights.'
                    exit(0)


if __name__ == '__main__':

    default_num_partial_caps = 50
    data_path = 'preprocess_data'
    coco_dir = '../external/coco'

    # Parse program arguments
    parser = argparse.ArgumentParser(description='Preprocess image captions if necessary.')
    parser.add_argument("-p", "--preprocess", default=False,
                        type=bool, help='If true, apply preprocessing')
    parser.add_argument("-n", "--num_partial_caps", default=default_num_partial_caps,
                        type=int, help='Number of partial captions to use.')
    parser.add_argument("-t", "--train", default=False,
                        type=bool, help='If true, train the model. Else, load the saved model')
    parser.add_argument("-m", "--max_cap_len", default=15,
                        type=int, help='Maximum caption length. ~95% of captions have length <= 15')
    parser.add_argument("-s", "--stream_size", default=100,
                        type=int, help='Stream size')
    parser.add_argument("-dir", "--model_weights_dir", type=str)

    args = parser.parse_args()
    train = args.train
    num_partial_caps = args.num_partial_caps
    preproc = args.preprocess
    max_caption_len = args.max_cap_len
    stream_size = args.stream_size
    num_streams = num_partial_caps / stream_size
    model_weights_dir = args.model_weights_dir

    last_saved_stream_num = configure_model_weights_dir(model_weights_dir, train)

    word_to_idx, idx_to_word = load_refexp_dicts()
    vocab_size = len(word_to_idx)

    EMBEDDING_DIM = 300
    embeddings_index = {}
    f = open('glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()




    embedding_matrix = np.zeros((len(word_to_idx) + 1, EMBEDDING_DIM))
    for word, i in word_to_idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

#    def relative_probs(all_preds):
    #all_preds = np.asarray(all_preds).astype('float64')
    #total_preds = np.sum(all_preds,axis=0)
    # division by zero
    
 #       return tf.sub(tf.log(all_preds[0]),tf.log(all_preds[1]))
    #return np.divide(all_preds[0],total_preds)

    #num_class_features = 1000 # dimensionality of CNN output
    #class_model = Sequential()
    #image_model.add(Dense(512, input_dim=num_img_features, activation='tanh',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #class_model.add(Dense(64, input_dim=num_class_features, activation='tanh')) 

    #num_img_features = 25088 # dimensionality of CNN output
    #image_model = Sequential()
    #image_model.add(Dense(512, input_dim=num_img_features, activation='tanh',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #image_model.add(Dense(128, input_dim=num_img_features, activation='tanh')) 
    #image_model.add(Dropout(0.3))
    #language_model = Sequential()
    #dummy = np.zeros(max_caption_len-1)
    #language_model.add(Masking(mask_value=0.0, input_shape=dummy.shape))
    ##language_model.add(Masking(mask_value=0.0, input_shape=(partial_captions[0].shape)))
    ##language_model.add(Embedding(vocab_size, 512, input_length=max_caption_len-1))
    #language_model.add(Embedding(vocab_size+1, 300, input_length=max_caption_len-1,weights=[embedding_matrix],trainable=True))
    #language_model.add(LSTM(output_dim=512, return_sequences=True,dropout_U=0.2,dropout_W=0.2))
    #language_model.add(TimeDistributed(Dense(512,activation='tanh'),name="lang"))
    #language_model.add(TimeDistributed(Dropout(0.3)))
    #image_model.add(RepeatVector(max_caption_len-1))
    #class_model.add(RepeatVector(max_caption_len-1))
    #image_model.add(RepeatVector(1))
    #model1 = Sequential()
    #model1.add(Merge([class_model,image_model, language_model], mode='concat', concat_axis=-1,name='foo'))
    #model1.add(LSTM(512, return_sequences=False,dropout_U=0.2,dropout_W=0.2))

    #model1.add(Dense(vocab_size))
    #model1.add(Activation('softmax',name='soft'))

    #class_model2 = Sequential()
    #image_model.add(Dense(512, input_dim=num_img_features, activation='tanh',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
   # class_model2.add(Dense(64, input_dim=num_class_features, activation='tanh')) 

    #num_img_features = 25088 # dimensionality of CNN output
  #  image_model2 = Sequential()
 #   #image_model.add(Dense(512, input_dim=num_img_features, activation='tanh',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
#    image_model2.add(Dense(128, input_dim=num_img_features, activation='tanh')) 
   # image_model2.add(Dropout(0.3))
  #  language_model2 = Sequential()
 #   dummy = np.zeros(max_caption_len-1)
#    language_model2.add(Masking(mask_value=0.0, input_shape=dummy.shape))
    #language_model.add(Masking(mask_value=0.0, input_shape=(partial_captions[0].shape)))
    #language_model.add(Embedding(vocab_size, 512, input_length=max_caption_len-1))
    #language_model2.add(Embedding(vocab_size+1, 300, input_length=max_caption_len-1,weights=[embedding_matrix],trainable=True))
    #language_model2.add(LSTM(output_dim=512, return_sequences=True,dropout_U=0.2,dropout_W=0.2))
   # language_model2.add(TimeDistributed(Dense(512,activation='tanh'),name="lang"))
  #  language_model2.add(TimeDistributed(Dropout(0.3)))
 #   image_model2.add(RepeatVector(max_caption_len-1))
    ##class_model2.add(RepeatVector(max_caption_len-1))
    #image_model.add(RepeatVector(1))
    #model2 = Sequential()
    #model2.add(Merge([class_model2,image_model2, language_model2], mode='concat', concat_axis=-1,name='foo'))
    #model2.add(LSTM(512, return_sequences=False,dropout_U=0.2,dropout_W=0.2))

    #model2.add(Dense(vocab_size))
    #model2.add(Activation('softmax',name='soft'))

    #model = Sequential()
    #prag_model.add(Merge([model,model2],mode=lambda x: relative_probs(np.asarray(x)),concat_axis=-1))
    #model.add(Merge([model1,model2],mode=lambda x : tf.log(x[0]) + tf.log(np.divide(x[0],x[1])),concat_axis=-1, output_shape=lambda x: x[0]))
    #model.add(Dense(vocab_size,W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))

#    dropout_param = 0.5
 #   recurrent_dropout_param = 0.2
 #   num_class_features = 1000 # dimensionality of CNN output
 #   num_img_features = 25088 # dimensionality of CNN output

  #  dummy = np.zeros(max_caption_len-1)

 #   image_model_input = Input(shape=(num_img_features,))
    #image_model = Dense(512, activation='tanh')(image_model_input)
    #image_model = RepeatVector(max_caption_len-1)(image_model)

    #class_model_input = Input(shape=(num_class_features,))
    #class_model = Dense(64, input_dim=num_class_features, activation='tanh')(class_model_input)
    #class_model = RepeatVector(max_caption_len-1)(class_model)

    #language_model_input = Input(shape=dummy.shape)
    #language_model = Embedding(vocab_size+1, 300, input_length=max_caption_len-1,weights=[embedding_matrix],trainable=True)(language_model_input)
    #language_model = LSTM(output_dim=512, return_sequences=True,dropout_U=0.2,dropout_W=0.2)(language_model)
    #language_model = TimeDistributed(Dense(512,activation='tanh'),name="lang")(language_model)
    #language_model = TimeDistributed(Dropout(0.3))(language_model)
    # merge_model = Input(inputs=[class_model_input,image_model_input,language_model_input])
    #merge_model = Merge([class_model_input,image_model_input,language_model_input],mode='concat',concat_axis=-1)
    #predictions = Dense(vocab_size,activation='softmax')(merge_model)
    # merge_model = Merge([class_model,image_model, language_model], mode='concat', concat_axis=-1,name='foo'))
    
    #model = Model(inputs=[class_model_input, image_model_input, language_model_input],outputs=predictions)

    #opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
   # class_model = Sequential()
#image_model.add(Dense(512, input_dim=num_img_features, activation='tanh',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #class_model.add(Dense(64, input_dim=num_class_features, activation='tanh'))
    #class_model.add(Dropout(dropout_param)) 

    #num_img_features = 25088 # dimensionality of CNN output
    #image_model = Sequential()
#image_model.add(Dense(512, input_dim=num_img_features, activation='tanh',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #image_model.add(Dense(512, input_dim=num_img_features, activation='tanh'))
    #image_model.add(Dropout(dropout_param))
    #language_model = Sequential()
    #dummy = np.zeros(max_caption_len)
    #language_model.add(Masking(mask_value=0.0, input_shape=dummy.shape))
#language_model.add(Masking(mask_value=0.0, input_shape=(partial_captions[0].shape)))
#language_model.add(Embedding(vocab_size, 512, input_length=max_caption_len-1))
    #language_model.add(Embedding(vocab_size+1, 300, input_length=max_caption_len-1))
# language_model.add(LSTM(output_dim=512, return_sequences=True,dropout_U=0.2,dropout_W=0.2))
    #language_model.add(TimeDistributed(Dense(512,activation='tanh'),name="lang"))
    #language_model.add(TimeDistributed(Dropout(dropout_param)))
    #image_model.add(RepeatVector(max_caption_len-1))
    #class_model.add(RepeatVector(max_caption_len-1))
#image_model.add(RepeatVector(1))
    #model = Sequential()
    #model.add(Merge([class_model,image_model, language_model], mode='concat', concat_axis=-1,name='foo'))
    #model.add(LSTM(512, return_sequences=True,dropout_U=recurrent_dropout_param,dropout_W=recurrent_dropout_param))
# model.add(LSTM(512, return_sequences=True,dropout_U=recurrent_dropout_param,dropout_W=recurrent_dropout_param))
    #model.add(LSTM(512, return_sequences=False,dropout_U=recurrent_dropout_param,dropout_W=recurrent_dropout_param))
#model.add(Dense(vocab_size,W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #model.add(Dense(vocab_size))
#model.add(Dense(512, input_dim=num_img_features, activation='tanh'))
    #model.add(Activation('softmax',name='soft'))


    dropout_param = 0.4
    recurrent_dropout_param = 0.0
    num_class_features = 1000 # dimensionality of CNN output
    class_model = Sequential()
    #image_model.add(Dense(512, input_dim=num_img_features, activation='tanh',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    class_model.add(Dense(64, input_dim=num_class_features, activation='relu'))
    class_model.add(Dropout(dropout_param)) 

    num_img_features = 25088 # dimensionality of CNN output
    image_model = Sequential()
    #image_model.add(Dense(512, input_dim=num_img_features, activation='tanh',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    image_model.add(Dense(400, input_dim=num_img_features, activation='relu')) 
    image_model.add(Dropout(dropout_param))
    language_model = Sequential()
    dummy = np.zeros(max_caption_len-1)
    language_model.add(Masking(mask_value=0.0, input_shape=dummy.shape))
    #language_model.add(Masking(mask_value=0.0, input_shape=(partial_captions[0].shape)))
    #language_model.add(Embedding(vocab_size, 512, input_length=max_caption_len-1))
    language_model.add(Embedding(vocab_size+1, 300, input_length=max_caption_len-1,weights=[embedding_matrix],trainable=False))
    #language_model.add(LSTM(output_dim=512, return_sequences=True,dropout_U=0.2,dropout_W=0.2))
    language_model.add(TimeDistributed(Dense(512,activation='tanh'),name="lang"))
    language_model.add(TimeDistributed(Dropout(dropout_param)))
    # language_model.add(LSTM(output_dim=512, return_sequences=True,dropout_U=0.2,dropout_W=0.2))
    #language_model.add(TimeDistributed(Dense(512,activation='relu'),name="lang"))
    #language_model.add(TimeDistributed(Dropout(dropout_param)))
    image_model.add(RepeatVector(max_caption_len-1))
    class_model.add(RepeatVector(max_caption_len-1))
    #image_model.add(RepeatVector(1))
    model = Sequential()
    model.add(Merge([class_model,image_model, language_model], mode='concat', concat_axis=-1,name='foo'))
    model.add(LSTM(512, return_sequences=True,dropout_U=recurrent_dropout_param,dropout_W=recurrent_dropout_param))
    # model.add(LSTM(512, return_sequences=True,dropout_U=recurrent_dropout_param,dropout_W=recurrent_dropout_param))
    model.add(LSTM(512, return_sequences=False,dropout_U=recurrent_dropout_param,dropout_W=recurrent_dropout_param))
    #model.add(Dense(vocab_size,W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(dropout_param))
    model.add(Dense(vocab_size))
#model.add(Dense(512, input_dim=num_img_features, activation='tanh'))
    model.add(Activation('softmax',name='soft'))

# first_inp = Input([])
    image_model_input1 = Input(shape=(num_img_features,))
    class_model_input1 = Input(shape=(num_class_features,))
    language_model_input1 = Input(shape=dummy.shape)

    image_model_input2 = Input(shape=(num_img_features,))
    class_model_input2 = Input(shape=(num_class_features,))
    language_model_input2 = Input(shape=dummy.shape)

# model1 = Model(input=[class_model_input1, image_model_input1, language_model_input1],output=predictions)
    model1 = model([class_model_input1, image_model_input1, language_model_input1])
    model2 = model([class_model_input2, image_model_input2, language_model_input2])


#    bayes_pred = merge([model1,model2],mode=lambda x: tf.sub(tf.log(x[0]),tf.log(x[1])),concat_axis=-1,output_shape=lambda x:x[0])
#    bayes_pred = merge([model1,model2],mode=lambda x: x[1],output_shape=lambda x:x[0])
    bayes_pred = merge([model1,model2],mode=lambda x: tf.exp(tf.sub(tf.log(x[0]),tf.log(((x[1]+x[0]))))),output_shape=lambda x:x[0])
#    bayes_pred = Dense(vocab_size)(bayes_pred)
# merge_model...

    final_model = Model(input=[class_model_input1, image_model_input1, language_model_input1,class_model_input2, image_model_input2, language_model_input2],output=bayes_pred)
    #final_model.add(Dense(vocab_size))
# # model = Input([])
# model_a = merge_model(a)
# model_b = merge_model(b)
# merged_vector = Merge([encoded_a, encoded_b], 'mode' = lambda x : relative_probs(np.asarray(x)) , axis=-1)
# # merged_vector = 
# # predictions = Dense(vocab_size, activation='softmax')(merged_vector)
# model = Model(inputs=[image_model_input_a, class_model_input_a,partial_caption, image_model_input_b, class_model_input_b,partial_caption], outputs=merged_vector)

#    sgd = SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
    final_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    images = None
    if args.train:
        # Check if we should start from some previously saved stream
        if not isinstance(last_saved_stream_num, int):
            last_saved_stream_num = 0

        for cur_stream_num in range(last_saved_stream_num+1, num_streams+1):
            print "Stream #: ", cur_stream_num, '/', num_streams
            classes, images, alt_classes,alt_images, partial_captions, next_words_one_hot, \
            vocab_size, idx_to_word, word_to_idx = load_stream(stream_num=cur_stream_num, stream_size=stream_size, preprocess=preproc,
                                                              max_caption_len=max_caption_len, word_to_idx=word_to_idx)

	    #alt_classes =  np.asarray([switch(x) for x in classes])
	    print(np.array_equal(classes,alt_classes),np.array_equal(images,alt_images),'CHECK')
#	    print True in [np.array_equal(x[0],x[1]) for x in zip([y for y in classes],[y for y in alt_classes])]
            early_stopping = EarlyStopping(monitor='val_loss', patience=1)
            final_model.fit([classes,images, partial_captions,alt_classes,alt_images, partial_captions], next_words_one_hot, batch_size=100, nb_epoch=6,validation_split=0.2,callbacks=[early_stopping,reduce_lr,csv_logger])    #model.save('modelweights_stream_' + str(i))
            #model.fit([images, partial_captions], next_words_one_hot, batch_size=100, nb_epoch=2)

            final_model.save(model_weights_dir + '/modelweights_stream_' + str(cur_stream_num))

            # Delete any of the older streams
            saved_models = get_saved_model_files(model_weights_dir)

#	    early_stopping = EarlyStopping(monitor='val_loss', patience=0)
   #         model.fit([classes,images, partial_captions,alt_classes,images, partial_captions], next_words_one_hot, batch_size=200, nb_epoch=3,validation_split=0.2,callbacks=[early_stopping])
  #          #model.save('modelweights_stream_' + str(i))
 #           #model.fit([images, partial_captions], next_words_one_hot, batch_size=100, nb_epoch=2)
#            model.save(model_weights_dir + '/modelweights_stream_' + str(i)

            for sm in saved_models:
                sm_stream_num = int(sm.split('_')[2])
  
                if sm_stream_num < cur_stream_num:
                    # delete the old stream
                    print 'Deleting saved model for stream', sm_stream_num, ' since we have a newer model now.'
                    remove(path.join(model_weights_dir, sm))


    else:
        #model = load_model("modelweights_test1/modelweights_stream_4")

        # Load the last stream that was saved
        final_model = load_last_saved_model(model_weights_dir)

    
#    intermediate_layer_model = Model(inputs=model.get_input_at(0),
 #                                outputs=model.output)


    def image_grab(id):


        #word_to_idx, idx_to_word = load_refexp_dicts()

        #classes, images, alt_classes,alt_images, partial_captions, next_words_one_hot, \
          #  vocab_size, idx_to_word, word_to_idx = load_stream(stream_num=cur_stream_num, stream_size=stream_size, preprocess=preproc,
         #                                                     max_caption_len=max_caption_len, word_to_idx=word_to_idx)


        #return images[:1],classes[:1],alt_images[:1],alt_classes[:1]
         try:
             new_image = get_image(id,path='/extra'+'/processed_flatten/')
         except IOError:
             new_image = predict_image(id)[1]
         try:
             new_class = get_image(id,path='/extra'+'/processed_predictions/')
         except IOError:
             new_class = predict_image(id)[2]
         return new_class,new_image

    def trained_pragmatic_speaker(target,distractor):
	new_class,new_image = image_grab(target)
	distractor_class,distractor_image = image_grab(distractor)
        #new_image,new_class, distractor_image,distractor_class = image_grab(target)
        cap = ['$START$']
        while len(cap) < max_caption_len:
            result = final_model.predict([new_class,new_image, words_to_caption(cap,word_to_idx,max_caption_len),distractor_class,new_image, words_to_caption(cap,word_to_idx,max_caption_len)])
            out = idx_to_word[np.argmax(result[0])]
            cap.append(out)

            print(result[0])
            #intermediate_output = intermediate_layer_model.predict([new_class,new_image, words_to_caption(cap,word_to_idx,max_caption_len)])
            #print intermediate_output[0]

        return cap

    print(trained_pragmatic_speaker('000000000431','000000000436'))
