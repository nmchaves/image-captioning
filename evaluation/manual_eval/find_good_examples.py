from manual_evaluator import load_test_examples, display_2_images
import matplotlib.pyplot as plt

from manual_evaluator import idx_to_cap_type, PRAG_CAP_IDX, BLINE_CAP_IDX, \
    GND_TR_CAP_IDX, PRAG_CAP2_IDX, BEAM_CAP_IDX, BEAM_CAP2_IDX

if __name__ == '__main__':
    plt.ion()
    for i, (target_id, img, idx_to_caption, distractor_id, img_distractor) in enumerate(load_test_examples()):
        display_2_images(img, img_distractor, '', figsize=(8, 6))
        print 'Target id: ', target_id
        for idx in [GND_TR_CAP_IDX, PRAG_CAP_IDX, BLINE_CAP_IDX, PRAG_CAP2_IDX, BEAM_CAP_IDX, BEAM_CAP2_IDX]:
            print idx_to_cap_type[idx], ':', idx_to_caption[idx]
        print 'Press ENTER to continue...'
        user_in = raw_input()
    plt.ioff()
