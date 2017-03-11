import sys
sys.path.append('../') # needed for Azure VM to see utils directory

from keras.regularizers import l2, activity_l2
from keras.callbacks import EarlyStopping
from os import path, makedirs, listdir
from shutil import rmtree
from keras.optimizers import Adam
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Activation, \
    Embedding, TimeDistributed, GRU, RepeatVector, Merge, Masking, LSTM, Dropout
from keras.applications import VGG19
# from keras.preprocessing.text import text_to_word_sequence
# from keras.preprocessing import image
import numpy as np
import pickle
from utils.preprocessing import preprocess_captioned_images, load_dicts, STOP_TOKEN
import argparse
from cnn_preprocessing import predict_image

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
def relative_probs(all_preds):
    all_preds = np.asarray(all_preds).astype('float64')
    total_preds = np.sum(all_preds,axis=0)
    # division by zero
    return np.divide(all_preds[0],total_preds)


def load_stream(stream_num, stream_size, preprocess, max_caption_len, word_to_idx):
    # Preprocess the data if necessary
    if preprocess:
        preprocess_captioned_images(stream_num=stream_num, stream_size=stream_size, word_to_idx=word_to_idx,
                                    max_cap_len=max_caption_len, coco_dir=coco_dir, category_name='person',
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
    partial_captions = X[1]
    vocab_size = len(word_to_idx)



    # Load the CNN feature representation of each image
    images = []
    for image_id in image_ids:
        number = str(('_0000000000000'+str(image_id))[-12:])

        try:
            x = get_image(number,path=coco_dir+'/processed/')
        except IOError:
            x = predict_image(str(image_id))

        images.append(x)

    images = np.asarray(images).transpose((1,0,2))[0]

    # Convert next words to one hot vectors
    new_next_words = []
    for x in next_words:
        a = np.zeros(vocab_size)
        a[x] = 1
        new_next_words.append(a)
    next_words_one_hot = np.asarray(new_next_words)

    return images, partial_captions, next_words_one_hot, \
        vocab_size, idx_to_word, word_to_idx


def load_last_saved_model(model_weights_dir):
    # Get all the files in the directory that contains the model weights
    saved_models = [s for s in listdir(model_weights_dir) if path.isfile(path.join(model_weights_dir, s))]

    # Return the model with the largest stream index (the index should be the last char of the filename)
    last_model_fname = sorted(saved_models, key=lambda ss: ss[-1], reverse=True)[0]
    return load_model(model_weights_dir + '/' + last_model_fname)


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

    if path.exists(model_weights_dir):
        # TODO: Check that it's ok to delete this directory
        # e.g. make sure that the dir is not a parent dir

        # Make sure that the user agrees to delete this directory
        print 'Caution! ', model_weights_dir, ' already exists!' \
            ' Do you want to delete all of its contents? (Y/N): '
        user_response = raw_input()
        if user_response == 'Y':
            print 'Deleting contents of directory', model_weights_dir
            rmtree(model_weights_dir)
        else:
            print 'Exiting. Please run again with a different directory for the model weights.'
            exit(0)

    # Create the directory for saving model weights
    makedirs(model_weights_dir)

    word_to_idx, idx_to_word = load_dicts()
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



    # Define the Model
    num_img_features = 4096 # dimensionality of CNN output
    image_model = Sequential()
    #image_model.add(Dense(512, input_dim=num_img_features, activation='tanh',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    image_model.add(Dense(128, input_dim=num_img_features, activation='tanh')) 
    image_model.add(Dropout(0.4))
    language_model = Sequential()
    dummy = np.zeros(max_caption_len-1)
    language_model.add(Masking(mask_value=0.0, input_shape=dummy.shape))
    #language_model.add(Masking(mask_value=0.0, input_shape=(partial_captions[0].shape)))
    #language_model.add(Embedding(vocab_size, 512, input_length=max_caption_len-1))
    language_model.add(Embedding(vocab_size+1, 300, input_length=max_caption_len-1,weights=[embedding_matrix],trainable=False))
    language_model.add(LSTM(output_dim=128, return_sequences=True,dropout_U=0.4,dropout_W=0.4))
    language_model.add(TimeDistributed(Dense(128,activation='tanh'),name="lang"))
    language_model.add(TimeDistributed(Dropout(0.4)))
    image_model.add(RepeatVector(max_caption_len-1))
    #image_model.add(RepeatVector(1))
    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1,name='foo'))
    model.add(LSTM(128, return_sequences=False,dropout_U=0.4,dropout_W=0.4))

    #model.add(Dense(vocab_size,W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Dense(vocab_size))
#model.add(Dense(512, input_dim=num_img_features, activation='tanh'))
    model.add(Activation('softmax',name='soft'))

    #opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    images = None
    if args.train:

        for i in range(num_streams):
            print "Stream #: ", i+1, '/', num_streams
            images, partial_captions, next_words_one_hot, \
            vocab_size, idx_to_word, word_to_idx = load_stream(stream_num=i+1, stream_size=stream_size, preprocess=preproc,
                                                              max_caption_len=max_caption_len, word_to_idx=word_to_idx)

	    early_stopping = EarlyStopping(monitor='val_loss', patience=1)
            model.fit([images, partial_captions], next_words_one_hot, batch_size=20, nb_epoch=3,validation_split=0.2,callbacks=[early_stopping])
            #model.save('modelweights_stream_' + str(i))
            #model.fit([images, partial_captions], next_words_one_hot, batch_size=100, nb_epoch=2)
            model.save(model_weights_dir + '/modelweights_stream_' + str(i))


    else:
        #model = load_model("modelweights_test1/modelweights_stream_4")

        # Load the last stream that was saved
        model = load_last_saved_model(model_weights_dir)

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

    #new_image = images[0].reshape((1, num_img_features))
    try:
        new_image = get_image('000000000036',path=coco_dir+'/processed/')
    except IOError:
        new_image = predict_image('000000000036')
    #new_image = np.zeros((1,num_img_features))
    cap = ['$START$']
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

    try:
        new_image2 = get_image('000000000731',path=coco_dir+'/processed/')
    except IOError:
        new_image2 = predict_image('000000000731')
    #new_image = np.zeros((1,num_img_features))
    cap = ['$START$']
    while len(cap) < max_caption_len:
        result = model.predict([new_image2, words_to_caption(cap,word_to_idx,max_caption_len)])
        m = max(result[0])
        # print(result)
        out = idx_to_word[[i for i, j in enumerate(result[0]) if j == m][0]]
        # out = idx_to_word[sample(result[0])]
        cap.append(out)
        print(cap)
        # if out == STOP_TOKEN:
        #     break
    try:
        new_image2 = get_image('000000000731',path=coco_dir+'/processed/')
    except IOError:
        new_image2 = predict_image('000000000731')
    #new_image = np.zeros((1,num_img_features))
    cap = ['$START$']
    while len(cap) < max_caption_len:
	result = model.predict([new_image, words_to_caption(cap,word_to_idx,max_caption_len)])[0]
        result2 = model.predict([new_image2, words_to_caption(cap,word_to_idx,max_caption_len)])[0]
	#inp = np.asarray([result,result2])
	#inp = relative_probs(inp)
	lam = 0.2
	elem_div = np.divide(result,result2)
	inp = (lam * np.log(result)) + ((1-lam) * np.log(elem_div) )
	print(inp)
	out = idx_to_word[np.argmax(inp)]
        #m = max(inp)
        # print(result)
        #out = idx_to_word[[i for i, j in enumerate(result[0]) if j == m][0]]
        # out = idx_to_word[sample(result[0])]
        cap.append(out)
        print(cap)
        # if out == STOP_TOKEN:
        #     break

    cap_number = 8
    branch_number = 4
   
    sents =[ (['$START$'],0)] * 8
    #sent_probs = [0] * 8
    lam = 0.5
    print(cap)
    while len(sents[0]) < max_caption_len:
	new_sents = [(0,0)]*32
	for i,row in enumerate(sents):
	    result = model.predict([new_image, words_to_caption(row[0],word_to_idx,max_caption_len)])[0]
	    result2 = model.predict([new_image2, words_to_caption(row[0],word_to_idx,max_caption_len)])[0]
            inp =  np.log(np.divide(result, result2 ** (1 - lam)))
	    top4idx = np.argsort(inp)[0:4]
	    for j in range(4):
		new_sents[i*4+j] = (row[0] + [top4idx[j]], row[1] + inp[top4idx[j]])
	sents = sorted(new_sents,key=lambda x: x[1])[:8]
	print sents
	    
        #result = np.asarray([model.predict([new_image, words_to_caption(sent[0],word_to_idx,max_caption_len)])[0] for sent in sents])
	#result2 = np.asarray([model.predict([new_image2, words_to_caption(sent[0],word_to_idx,max_caption_len)])[0] for sent in sents])
	#inp =  np.log(np.divide(result, result2 ** (1 - lam)))
	#result2 = model.predict([new_image2, words_to_caption(cap,word_to_idx,max_caption_len)])[0]
        #inp = np.asarray([result,result2])
        #inp = relative_probs(inp)
        #elem_div = np.divide(result,result2)
        #inp = (lam * np.log(result)) + ((1-lam) * np.log(elem_div) )
	#for i,row in enumerate(inp):
	    #top_n
        #print(inp)
        #out = idx_to_word[np.argmax(inp)]
        #m = max(inp)
        # print(result)
        #out = idx_to_word[[i for i, j in enumerate(result[0]) if j == m][0]]
        # out = idx_to_word[sample(result[0])]
        #cap.append(out)
        #print(cap)
        # if out == STOP_TOKEN:
        #     break


