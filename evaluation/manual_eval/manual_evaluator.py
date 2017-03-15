from numpy.random import binomial
import skimage.io as io
import matplotlib.pyplot as plt
from os import path

def test_samples():
    img = io.imread(path.join('', 'man_img.png'))
    img_distracter = io.imread(path.join('', 'horse_img.png'))
    plt.figure()
    plt.imshow(I)
    img = ''
    img_distracter = ''

    yield ['img_0123', img, 'some generate caption', img_distracter]
    #yield ['todo: yield an id for the sample (the image or caption, not sure which)', 'todo: img here', 'todo: caption here', 'todo: distractor img here']


def display_sample(img, generated_caption, img_distracter):
    print generated_caption
    # todo: bernoulli rand then show img on left or right
    true_img_pos = binomial(1, 0.5) + 1
    if true_img_pos == 1:
    else:



    return true_img_pos


def process_user_input(input_str):
    if input_str == '1' or input_str == '2':
        return int(input_str)
    else:
        print ''
        return None


if __name__ == '__main__':
    print 'Time to evaluate performance! You\'ll see images appear on the ' \
        'screen along with a caption. Tell us which image you think the caption was written for.'

    print 'Where would you like to save your results? (enter filename):'
    out_file = raw_input()

    sample_ids_seen = []
    user_inputs = []
    user_input_correctness = []

    for i, (sample_id, img, generated_caption, img_distracter) in enumerate(test_samples()):
        true_img_position = display_sample(img, generated_caption, img_distracter)

        print 20*'='
        print 'Example', i+1

        display_sample(img, generated_caption, img_distracter)

        print 'This caption is for image 1 or image 2: '
        user_in = process_user_input(raw_input())
        while user_in is None:
            print 'Please enter 1 or 2.'
            user_in = process_user_input(raw_input())

        user_inputs.append(user_in)
        user_input_correctness.append(1 if user_in == true_img_pos else 0)
        sample_ids_seen.append(sample_id)


