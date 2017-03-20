import skimage.io as io
import matplotlib.pyplot as plt
from os import path
import pandas as pd
import numpy as np
from random import randint
import argparse
from collections import defaultdict

POS_LEFT = 1
POS_RIGHT = 2

PRAG_CAP_IDX = 1  # pragmatic captions
BLINE_CAP_IDX = 2  # baseline captions
GND_TR_CAP_IDX = 3  # ground truth captions

idx_to_cap_type = {
    PRAG_CAP_IDX: 'Pragmatic',
    BLINE_CAP_IDX: 'Baseline',
    GND_TR_CAP_IDX: 'Ground Truth Label'
}


# TODO: load the real data
def load_test_samples(start_idx):
    img = io.imread(path.join('', 'man_img.png'))
    img_distracter = io.imread(path.join('', 'horse_img.png'))

    for i in range(start_idx, 10):
        # TODO: load the real data
        idx_to_caption = {
            PRAG_CAP_IDX: 'pragmatic caption...',
            BLINE_CAP_IDX: 'baseline caption...',
            GND_TR_CAP_IDX: 'ground truth caption...'
        }
        yield ['img id here', img, idx_to_caption, img_distracter]


def display_sample(sample_id, img, caption, img_distracter):
    print 'Sample ID:', sample_id

    true_img_pos = randint(1, 2)

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


def lists_to_data_frame(sample_ids_seen, cap_indices, user_inputs, user_input_correctness):
    df = pd.DataFrame(data=np.asarray([cap_indices, user_inputs, user_input_correctness]).T,
                        columns=['caption_indices', 'user_inputs', 'user_input_correctness'])
    df['sample_ids'] = sample_ids_seen  # add string values separately
    return df


def data_frame_to_lists(df):
    return list(df['sample_ids'].values), \
           list(df['caption_indices'].values), \
           list(df['user_inputs'].values), \
           list(df['user_input_correctness'].values)


def save_results(sample_ids_seen, cap_indices, user_inputs, user_input_correctness, out_file):
    df = lists_to_data_frame(sample_ids_seen, cap_indices, user_inputs, user_input_correctness)
    df.to_csv(out_file)
    return df


def df_to_results_dict(df):
    results = {}

    for idx in [PRAG_CAP_IDX, BLINE_CAP_IDX, GND_TR_CAP_IDX]:
        examples = df.loc[df['caption_indices'] == idx]
        n_examples = examples.shape[0]
        if n_examples > 0:
            n_correct = np.sum(examples['user_input_correctness'])
            results[idx] = (n_correct, n_examples)

    return results


def print_results(result_dict):
    total_examples = 0

    print 20*"="
    print 'Your accuracy scores for each type of caption (1.0 is max score):'
    for (cap_type_idx, (n_correct, n_examples)) in result_dict.iteritems():
        total_examples += n_examples
        acc = 1.0 * n_correct / n_examples
        print idx_to_cap_type[cap_type_idx] + ': ' + str(acc)

    print '(There were', total_examples, 'test examples in total.)'


def load_results(filename):
    results_df = pd.read_csv(filename)
    return df_to_results_dict(results_df)


def merge_results(filenames):
    result_sets = [load_results(f) for f in filenames]

    merged = defaultdict(lambda: (0, 0))
    for rs in result_sets:
        for (cap_type_idx, (n_correct, n_examples)) in rs.iteritems():
            merged[cap_type_idx] = tuple(map(sum, zip(merged[cap_type_idx], (n_correct, n_examples))))
    return merged


if __name__ == '__main__':
    # Use interactive plotting mode so that we can interact with the console while showing images
    plt.ion()

    # Parse program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--synthesize", default=False, action='store_true',
                        help='If true, don\'t show images. Just synthesize the results.')
    parser.add_argument("-r", "--results", nargs='+', default=[],
                        help='If synthesizing, this specifies where to load the results from.')
    parser.add_argument("-res", "--resume", type=str, default='',
                        help='The partial results file to use if you want to resume a partially completed run.')
    args = parser.parse_args()

    if args.synthesize:
        filenames = args.results
        print 'Merging results for files:', ', '.join(filenames)
        results = merge_results(filenames)
        print_results(results)
        exit(0)

    print 'Time to evaluate performance! You\'ll see images appear on the ' \
        'screen along with a caption. Tell us which image you think the caption was written for.\n' \
        'Type q to quit at any time (your partial results will be saved).'

    print 'Where would you like to save your results when you\'re done? (enter filename):'
    out_file = raw_input()

    start_idx = 0
    if args.resume:
        sample_ids_seen, cap_indices, user_inputs, user_input_correctness = data_frame_to_lists(pd.read_csv(args.resume))
        start_idx = len(sample_ids_seen)
        print 'Continuing from where you left off (you previously saw', start_idx, 'examples).'
    else:
        sample_ids_seen = []
        cap_indices = []
        user_inputs = []
        user_input_correctness = []

    for i, (sample_id, img, idx_to_caption, img_distracter) in enumerate(load_test_samples(start_idx)):

        print 20*'='
        print 'Example', i+start_idx+1

        cap_to_show, cap_idx = rand_caption(idx_to_caption)
        true_img_pos = display_sample(sample_id, img, cap_to_show, img_distracter)

        user_quit = False  # flag to tell if user quits

        while True:
            print 'Image 1 or Image 2 (q to quit): '
            user_in = raw_input()
            if user_in == '1' or user_in == '2':
                user_in = int(user_in)
                break
            elif user_in == 'q':
                print 'Quitting early. Saving partial results.'
                user_quit = True
                break

        if user_quit:
            break

        user_inputs.append(user_in)
        user_input_correctness.append(1 if user_in == true_img_pos else 0)
        sample_ids_seen.append(sample_id)
        cap_indices.append(cap_idx)

    df = save_results(sample_ids_seen, cap_indices, user_inputs, user_input_correctness, out_file)
    results = df_to_results_dict(df)
    print_results(results)

    plt.ioff()
