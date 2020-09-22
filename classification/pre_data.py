import os
# import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import skimage.io
import skimage.transform
import skimage.color
import skimage
import IPython
from pycocotools.coco import COCO

set_name='2017'
root_dir = '/home/shuxuang/dataset/coco2017'
coco      = COCO(os.path.join(root_dir, 'annotations', 'instances_' + set_name + '.json'))
image_ids = coco.getImgIds()

def load_image(image_id):
    image_info = coco.loadImgs(image_id)[0]
    path       = os.path.join(root_dir, set_name, image_info['file_name'])
    img = skimage.io.imread(path)

    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    # return img.astype(np.float32)/255.0, image_info['le_name']
    return img.astype(np.float32), image_info['file_name']

def load_classes():
    # load class names (name -> label)
    categories = coco.loadCats(coco.getCatIds())
    categories.sort(key=lambda x: x['id'])

    classes             = {}
    coco_labels         = {}
    coco_labels_inverse = {}
    for c in categories:
        coco_labels[len(classes)] = c['id']
        coco_labels_inverse[c['id']] = len(classes)
        classes[c['name']] = len(classes)

    # also load the reverse (label -> name)
    labels = {}
    for key, value in classes.items():
        labels[value] = key
    return classes, coco_labels, coco_labels_inverse, labels

_, _, coco_labels_inverse, _ = load_classes()

def coco_label_to_label(coco_label, coco_labels_inverse):
    return coco_labels_inverse[coco_label]

def load_annotations(image_id, coco_labels_inverse):
    # get ground truth annotations
    annotations_ids = coco.getAnnIds(imgIds=image_id, iscrowd=False)
    annotations     = np.zeros((0, 5))

    # some images appear to miss annotations (like image with id 257034)
    if len(annotations_ids) == 0:
        return annotations

    # parse annotations
    coco_annotations = coco.loadAnns(annotations_ids)
    for idx, a in enumerate(coco_annotations):

        # some annotations have basically no width / height, skip them
        if a['bbox'][2] < 1 or a['bbox'][3] < 1:
            continue

        annotation        = np.zeros((1, 5))
        annotation[0, :4] = a['bbox']
        annotation[0, 4]  = coco_label_to_label(a['category_id'], coco_labels_inverse)
        annotations       = np.append(annotations, annotation, axis=0)

    # transform from [x, y, w, h] to [x1, y1, x2, y2]
    annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
    annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

    return annotations



def crop_and_save_patches(output_dir):
    file1 = open(os.path.join(output_dir, "%s.txt" % set_name), "a")
    n_imgs = len(image_ids)
    n_objs = 0
    for idx in image_ids:
        # image_id = ids[index]
        image, image_name = load_image(idx)
        annots = load_annotations(idx, coco_labels_inverse)
        # sample = {'img': img, 'annot': annot}
        image_name_ = image_name.split('.')[0]

        # IPython.embed()
        for (i,annot) in enumerate(annots):
            # IPython.embed()
            n_objs = n_objs +1 
            img = image[int(annot[1]):int(annot[3]), int(annot[0]):int(annot[2]), :]
            # print(img.shape)
            label = int(annot[4])
            img_name = str(label) + '_' + image_name_ + '_' + str(i) + '.jpg'
            print(img_name)
            Image.fromarray(img.astype(np.uint8)).save(os.path.join(output_dir, img_name))
            file1.write(img_name+'\n')  
    file1.close()
    print('%d objects from %d images' % (n_objs, n_imgs))

# cocoval2017: val2017: 36334 objects from 5k images
# train2017: 849902 objects from 118287 images
## coco2017zip: 66594
## coco2017-cls: 66621 /home/shuxuang/data/coco2017-cls.tar.gz 

if __name__ == '__main__':
    
    output_dir = os.path.join('/home/shuxuang/data/coco2017-cls/', "%s" % set_name)
    # output_dir = './demos/coco2017-cls/'
    import errno
    if output_dir:
        try:
            os.makedirs(output_dir
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
    crop_and_save_patches(output_dir)