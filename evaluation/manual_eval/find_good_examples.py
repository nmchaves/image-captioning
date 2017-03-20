from manual_evaluator import load_test_examples, display_2_images
import matplotlib.pyplot as plt

from manual_evaluator import PRAG_CAP_IDX, BLINE_CAP_IDX, GND_TR_CAP_IDX

if __name__ == '__main__':
    plt.ion()
    for i, (target_id, img, idx_to_caption, distractor_id, img_distractor) in enumerate(load_test_examples()):
        display_2_images(img, img_distractor, '', figsize=(10, 8))
        print 'Target id: ', target_id
        print 'Baseline caption:', idx_to_caption[BLINE_CAP_IDX]
        print 'Pragmatic caption:', idx_to_caption[PRAG_CAP_IDX]
        print 'Press ENTER to continue...'
        user_in = raw_input()
    plt.ioff()
