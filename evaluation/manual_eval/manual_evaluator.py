import skimage.io as io
import matplotlib.pyplot as plt
from os import path, listdir
import pandas as pd
import numpy as np
from random import randint
import argparse
from collections import defaultdict
import pickle
from keras.preprocessing.text import text_to_word_sequence

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


def get_child_dirs(dir):
    return [name for name in listdir(dir) if path.isdir(path.join(dir, name))]


def sorted_eval_example_dirs(data_dir):
    dir_names = get_child_dirs(data_dir)
    return sorted(dir_names, key=lambda name: name.split('_')[1])


def load_test_example(data_dir, example_dir):
    with open(path.join(data_dir, example_dir, 'ids_dict')) as f:
        ids_dict = pickle.load(f)
        target_id = ids_dict['target']
        distractor_id = ids_dict['distractor']

    with open(path.join(data_dir, example_dir, 'caption_dict')) as f:
        idx_to_caption = pickle.load(f)

    img = io.imread(path.join(data_dir, example_dir, 'img_target.png'))
    img_distractor = io.imread(path.join(data_dir, example_dir, 'img_distractor.png'))

    return target_id, img, idx_to_caption, distractor_id, img_distractor


def load_test_examples(start_idx=0):
    # TODO: change this to the non-mock directory once the data is ready
    data_dir = 'manual_eval_data_mock'

    test_example_dirs = sorted_eval_example_dirs(data_dir)
    for ex_dir in test_example_dirs[start_idx:]:
        yield load_test_example(data_dir, ex_dir)


def display_2_images(img1, img2, suptitle, figsize=(12, 10)):
    plt.figure(1, figsize=figsize)

    if suptitle:
        plt.suptitle(suptitle, fontsize=20, y=0.8)

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(str(POS_LEFT))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title(str(POS_RIGHT))
    plt.axis('off')

    plt.tight_layout()
    plt.draw()


def display_sample(target_id, img, caption, distractor_id, img_distractor):
    # Convert caption to all lower case with no punctuation
    caption = ' '.join(text_to_word_sequence(caption))

    true_img_pos = randint(1, 2)
    img1 = img if true_img_pos == 1 else img_distractor
    img2 = img if true_img_pos == 2 else img_distractor

    display_2_images(img1, img2, caption)

    return true_img_pos


def rand_caption(idx_to_caption):
    n_caption_types = len(idx_to_caption)
    rnd_caption_idx = randint(1, n_caption_types)
    return idx_to_caption[rnd_caption_idx], rnd_caption_idx


def lists_to_data_frame(target_ids, distractor_ids, cap_indices, user_inputs, user_input_correctness):
    df = pd.DataFrame(data=np.asarray([cap_indices, user_inputs, user_input_correctness]).T,
                        columns=['caption_indices', 'user_inputs', 'user_input_correctness'])
    # add string values separately to prevent type conversions
    df['target_ids'] = target_ids
    df['distractor_ids'] = distractor_ids
    return df


def data_frame_to_lists(df):
    return list(df['target_ids'].values), \
           list(df['distractor_ids'].values), \
           list(df['caption_indices'].values), \
           list(df['user_inputs'].values), \
           list(df['user_input_correctness'].values)


def save_results(target_ids, distractor_ids, cap_indices, user_inputs, user_input_correctness, out_file):
    df = lists_to_data_frame(target_ids, distractor_ids, cap_indices, user_inputs, user_input_correctness)
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
    parser.add_argument("-f", "--files", nargs='+', default=[],
                        help='If synthesizing, this specifies which files to load the results from.')
    parser.add_argument("-r", "--resume", type=str, default='',
                        help='The partial results file to use if you want to resume a partially completed run.')
    args = parser.parse_args()

    if args.synthesize:
        filenames = args.files
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
        target_ids, distractor_ids, cap_indices, user_inputs, user_input_correctness = data_frame_to_lists(pd.read_csv(args.resume))
        start_idx = len(target_ids)
        print 'Continuing from where you left off (you previously saw', start_idx, 'examples).'
    else:
        target_ids = []
        distractor_ids = []
        cap_indices = []
        user_inputs = []
        user_input_correctness = []

    for i, (target_id, img, idx_to_caption, distractor_id, img_distractor) in enumerate(load_test_examples(start_idx)):

        print 20*'='
        print 'Example', i+start_idx+1

        cap_to_show, cap_idx = rand_caption(idx_to_caption)
        true_img_pos = display_sample(target_id, img, cap_to_show, distractor_id, img_distractor)

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
        target_ids.append(target_id)
        distractor_ids.append(distractor_id)
        cap_indices.append(cap_idx)

    df = save_results(target_ids, distractor_ids, cap_indices, user_inputs, user_input_correctness, out_file)
    results = df_to_results_dict(df)
    print_results(results)

    plt.ioff()
