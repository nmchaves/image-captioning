import skimage.io as io
import matplotlib.pyplot as plt
from os import path
import pandas as pd
import numpy as np
from random import randint
import argparse

POS_LEFT = 1
POS_RIGHT = 2

PRAG_CAP_IDX = 1  # pragmatic captions
BLINE_CAP_IDX = 2  # baseline captions
GND_TR_CAP_IDX = 3  # ground truth captions

def test_samples():
    img = io.imread(path.join('', 'man_img.png'))
    img_distracter = io.imread(path.join('', 'horse_img.png'))

    idx_to_caption1 = {
        PRAG_CAP_IDX: 'pragmatic caption...',
        BLINE_CAP_IDX: 'baseline caption...',
        GND_TR_CAP_IDX: 'ground truth caption...'
    }
    yield ['img_0123', img, idx_to_caption1, img_distracter]

    idx_to_caption2 = {
        PRAG_CAP_IDX: 'pragmatic caption...',
        BLINE_CAP_IDX: 'baseline caption...',
        GND_TR_CAP_IDX: 'ground truth caption...'
    }
    yield ['img_0123', img, idx_to_caption2, img_distracter]
    #yield ['todo: yield an id for the sample (the image or caption, not sure which)', 'todo: img here', 'todo: caption here', 'todo: distractor img here']


def display_sample(img, caption, img_distracter):

    true_img_pos = randint(1,2)

    plt.figure(1, figsize=(8, 6))

    plt.suptitle(caption, ha='center', va='center', fontsize=20)

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


def rand_caption(idx_to_caption):
    n_caption_types = len(idx_to_caption)
    rnd_caption_idx = randint(1, n_caption_types)
    return idx_to_caption[rnd_caption_idx], rnd_caption_idx


def to_DataFrame(sample_ids_seen, cap_indices, user_inputs, user_input_correctness):
    df = pd.DataFrame(data=np.asarray([cap_indices, user_inputs, user_input_correctness]).T,
                        columns=['caption_indices', 'user_inputs', 'user_input_correctness'])
    df['sample_ids'] = sample_ids_seen  # add string values separately
    return df


def save_results(sample_ids_seen, cap_indices, user_inputs, user_input_correctness, out_file):
    df = to_DataFrame(sample_ids_seen, cap_indices, user_inputs, user_input_correctness)
    df.to_csv(out_file)
    return df


def user_accuracy(df):
    acc = {}

    for idx in [PRAG_CAP_IDX, BLINE_CAP_IDX, GND_TR_CAP_IDX]:
        print 'idx: ', idx
        examples = df.loc[df['caption_indices'] == idx]
        print 'ex: ', examples
        n_examples = examples.shape[0]
        print 'n_examples', n_examples
        if n_examples > 0:
            n_correct = np.sum(examples['user_input_correctness'])
            acc[idx] = 1.0 * n_correct / n_examples

    return acc


if __name__ == '__main__':
    # Use interactive plotting mode so that we can interact with
    # the console while showing images
    plt.ion()

    '''
    # Parse program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pragmatic", default=False, action='store_true',
                        help='If true, include pragmatic captions.')
    parser.add_argument("-b", "--baseline", default=False, action='store_true',
                        help='If true, include baseline captions.')
    parser.add_argument("-g", "--gnd_truth", default=False, action='store_true',
                        help='If true, include ground truth captions.')
    args = parser.parse_args()
    '''

    print 'Time to evaluate performance! You\'ll see images appear on the ' \
        'screen along with a caption. Tell us which image you think the caption was written for.\n' \
        'Type q to quit at any time (your partial results will be saved).'

    print 'Where would you like to save your results? (enter filename):'
    out_file = raw_input()

    sample_ids_seen = []
    user_inputs = []
    user_input_correctness = []
    cap_indices = []

    for i, (sample_id, img, idx_to_caption, img_distracter) in enumerate(test_samples()):

        print 20*'='
        print 'Example', i+1

        if i == 0:
            has_prag_caps = PRAG_CAP_IDX in idx_to_caption
            has_bline_caps = BLINE_CAP_IDX in idx_to_caption
            has_gnd_tr_caps = GND_TR_CAP_IDX in idx_to_caption

        cap_to_show, cap_idx = rand_caption(idx_to_caption)
        true_img_pos = display_sample(img, cap_to_show, img_distracter)

        while True:
            print 'Image 1 or Image 2 (q to quit): '
            user_in = raw_input()
            if user_in == '1' or user_in == '2':
                user_in = int(user_in)
                break
            elif user_in == 'q':
                print 'Quitting early. Saving partial results.'
                save_results(sample_ids_seen, cap_indices, user_inputs, user_input_correctness, out_file)
                exit(0)

        user_inputs.append(user_in)
        user_input_correctness.append(1 if user_in == true_img_pos else 0)
        sample_ids_seen.append(sample_id)
        cap_indices.append(cap_idx)

    df = save_results(sample_ids_seen, cap_indices, user_inputs, user_input_correctness, out_file)
    print df
    print user_accuracy(df)

    plt.ioff()
