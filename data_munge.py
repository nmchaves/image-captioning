import json
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
from keras.preprocessing import sequence
captions_path = '/Users/reuben/Downloads/annotations/captions_train2014.json'


captions = open(captions_path,'r')

captions = json.loads(captions.read())

# print([x for x in captions])
sentence = (captions['annotations'][0]['caption'])
number = captions['annotations'][0]['image_id']
number = ('0000000000000'+str(number))[-12:]
print(number)
image_path = '/Users/reuben/Downloads/train2014/COCO_train2014_'+number+'.jpg'

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# print(preprocess_image(image_path))
max_caption_len = 16
vocab_size = 10
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

print(preprocess_img_and_text([image_path],[sentence]))

# first: takes first image and gives its caption


# todo: returns a pickled list of numpy tensors of annotations, and a list of numpy tensors of images 