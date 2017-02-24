import sys
sys.path.append('google_refexp_py_lib')
sys.path.append('external/coco/PythonAPI')

from refexp import Refexp
import numpy as np
from utils.preprocessing import preprocess_captions, preprocess_image, repeat_imgs
from pycocotools.coco import COCO
import cPickle as pickle

if __name__ == '__main__':
    # Specify datasets path.
    refexp_filename='google_refexp_dataset_release/google_refexp_train_201511_coco_aligned.json'
    coco_filename='external/coco/annotations/instances_train2014.json'
    datasetDir = 'external/coco/'
    datasetType = 'train2014'

    # Create Refexp instance.
    refexp = Refexp(refexp_filename, coco_filename)

    # Get image ids of all images containing human beings
    categoryIds = refexp.getCatIds(catNms=['person'])
    imgIds = refexp.getImgIds(catIds=categoryIds)

    # Select 2 random images
    nImages = 2
    randImgIndices = np.random.choice(np.arange(0, len(imgIds)), size=(nImages, 1), replace=True)
    randImgIds = [imgIds[idx] for idx in randImgIndices]

    # TODO: double-check that you don't need to access element 0 of each coco img
    coco_imgs = refexp.loadImgs(randImgIds)

    # The actual images as numpy arrays
    images = [preprocess_image('%s/images/%s/%s' % (datasetDir, datasetType, img['file_name'])) for img in coco_imgs]
    images = np.squeeze(np.asarray(images))

    # initialize COCO api for caption annotations
    capsAnnFile = '%s/annotations/captions_%s.json'%(datasetDir, datasetType)
    coco_caps = COCO(capsAnnFile)
    '''
    annIds = coco_caps.getAnnIds(imgIds=randImgIds)
    anns = coco_caps.loadAnns(annIds)
    coco_caps.showAnns(anns)

    # For now, just take 1 caption from each image, even though that caption
    # corresponds to a specific region of the image
    captions = []
    for ann in anns:
        # Convert caption from unicode string to regular string
        captions.append(str(ann['caption']))
    '''
    captions = []
    for id in randImgIds:
        annId = coco_caps.getAnnIds(imgIds=id)
        anns = coco_caps.loadAnns(annId)

        # For now, just take 1 caption from each image, even though that caption
        # corresponds to a specific region of the image
        ann = anns[0]

        # Convert caption from unicode string to regular string
        captions.append(str(ann['caption']))

    print 'Captions: ', captions

    unique, word_to_idx, idx_to_word, partial_caps, next_words = preprocess_captions(captions)
    vocab_size = len(unique)

    # Save data
    X = [images, np.asarray(partial_caps)]
    y = np.asarray(next_words)
    out = X,y,captions,vocab_size,idx_to_word, word_to_idx
    pickle.dump(out, open( "savedoc", "r+" ))


    '''
    # Get the annotation (ie the true labels) for each image
    annotIds = refexp.getAnnIds(imgIds=randImgIds) # [...(=id for id in randImgIds]
    ##ann = refexp.loadAnns(annotIds[0])[0] # 14763
    anns = refexp.loadAnns(annotIds)

    for ann in anns:
        ref_exp = refexp.dataset['refexps'][ann['refexp_ids'][0]]['raw']
    # todo: do for each
    # get referring expression
    refexp.dataset['refexps'][ann['refexp_ids'][0]]['raw']
    '''