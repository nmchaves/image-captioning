from numpy.random import binomial
import skimage.io as io
import matplotlib.pyplot as plt
from os import path
import pandas as pd
import numpy as np

POS_LEFT = 1
POS_RIGHT = 2


def test_samples():
    img = io.imread(path.join('', 'man_img.png'))
    img_distracter = io.imread(path.join('', 'horse_img.png'))

    yield ['img_0123', img, 'some generated caption', img_distracter]
    yield ['img_0123', img, 'another generated caption', img_distracter]
    #yield ['todo: yield an id for the sample (the image or caption, not sure which)', 'todo: img here', 'todo: caption here', 'todo: distractor img here']


def display_sample(img, generated_caption, img_distracter):

    true_img_pos = binomial(1, 0.5) + 1

    plt.figure(1, figsize=(8, 6))

    plt.suptitle(generated_caption, ha='center', fontsize=20)

    plt.subplot(1, 2, 1)
    plt.imshow(img if true_img_pos == 1 else img_distracter)
    plt.title(str(POS_LEFT))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img if true_img_pos == 2 else img_distracter)
    plt.title(str(POS_RIGHT))
    plt.axis('off')

    plt.draw()
    return true_img_pos


def save_results(sample_ids_seen, user_inputs, user_input_correctness, out_file):
    df = pd.DataFrame(data=np.asarray([sample_ids_seen, user_inputs, user_input_correctness]).T,
                      columns=['sample_ids', 'user_inputs', 'user_input_correctness'])
    df.to_csv(out_file)


if __name__ == '__main__':
    # Use interactive plotting mode so that we can interact with
    # the console while showing images
    plt.ion()

    print 'Time to evaluate performance! You\'ll see images appear on the ' \
        'screen along with a caption. Tell us which image you think the caption was written for.\n' \
        'Type q to quit at any time (your partial results will be saved).'

    print 'Where would you like to save your results? (enter filename):'
    out_file = raw_input()

    sample_ids_seen = []
    user_inputs = []
    user_input_correctness = []

    for i, (sample_id, img, generated_caption, img_distracter) in enumerate(test_samples()):

        print 20*'='
        print 'Example', i+1

        true_img_pos = display_sample(img, generated_caption, img_distracter)

        while True:
            print 'Image 1 or Image 2 (q to quit): '
            user_in = raw_input()
            if user_in == '1' or user_in == '2':
                user_in = int(user_in)
                break
            elif user_in == 'q':
                print 'Quitting early. Saving partial results.'
                save_results(sample_ids_seen, user_inputs, user_input_correctness, out_file)
                exit(0)

        user_inputs.append(user_in)
        user_input_correctness.append(1 if user_in == true_img_pos else 0)
        sample_ids_seen.append(sample_id)

    save_results(sample_ids_seen, user_inputs, user_input_correctness, out_file)
    plt.ioff()