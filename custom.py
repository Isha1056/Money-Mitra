import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
import imgaug

ROOT_DIR = "/mnt/c//Users/ishag/Downloads/Mask-R-CNN-using-Tensorflow2-main/Mask-R-CNN-using-Tensorflow2-main"

sys.path.append(ROOT_DIR)  
from mrcnn.config import Config
from mrcnn import model as modellib, utils

COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5"
DEFAULT_LOGS_DIR = "logs/"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def ann_to_mask_via(ann, height, width):
  """
  Convert geometries specific to via tool to mask
  :return: binary mask (numpy 2D array)
  """
  mask = np.zeros([height, width], dtype=np.uint8)
  rr, cc = ann_to_geometry_via(ann)
  if rr is not None and cc is not None:
    rr[rr > mask.shape[0]-1] = mask.shape[0]-1
    cc[cc > mask.shape[1]-1] = mask.shape[1]-1
    mask[rr, cc] = 1
  return mask


def ann_to_geometry_via(ann):
  """
  Load Different Geometry types specific to via tool
  Ref: http://scikit-image.org/docs/0.8.0/api/skimage.draw.html
  Ref: http://scikit-image.org/docs/0.14.x/api/skimage.draw.html#skimage.draw.line
  """
  ## rr, cc = 0, 0
  rr = np.zeros([0, 0, len(ann)],dtype=np.uint8)
  cc = np.zeros([0, 0, len(ann)],dtype=np.uint8)
  
  if ann['name'] == 'polygon':
    rr, cc = skimage.draw.polygon(ann['all_points_y'], ann['all_points_x'])
  elif ann['name'] == 'rect':
    rr, cc = skimage.draw.rectangle((ann['y'], ann['x']), extent=(ann['height'], ann['width']))
  elif ann['name'] == 'circle':
    rr, cc = skimage.draw.circle(ann['cy'], ann['cx'],ann['r'])
  elif ann['name'] == 'ellipse':
    rr, cc = skimage.draw.ellipse(ann['cy'], ann['cx'],ann['ry'],ann['rx'])
  elif ann['name'] == 'polyline':
    points = list(zip(ann['all_points_x'],ann['all_points_y']))
    rr,cc = polyline2coords(points)
  else:
    ## TBD: raise error
    print("Annotation Geometry Not Yet Supported")
    print("ann_to_mask_via: ann['name']: {}".format(ann['name']))
  return rr,cc


def polyline2coords(points):
    """
    Reference:
    https://www.programcreek.com/python/example/94226/skimage.draw.line

    Return row and column coordinates for a polyline.

    >>> rr, cc = polyline2coords([(0, 0), (2, 2), (2, 4)])
    >>> list(rr)
    [0, 1, 2, 2, 3, 4]
    >>> list(cc)
    [0, 1, 2, 2, 2, 2]

    :param list of tuple points: Polyline in format [(x1,y1), (x2,y2), ...] 
    :return: tuple with row and column coordinates in numpy arrays
    :rtype: tuple of numpy array
    """
    coords = []
    for i in range(len(points) - 1):
        xy = list(map(int, points[i] + points[i + 1]))
        coords.append(skimage.draw.line(xy[1], xy[0], xy[3], xy[2]))
    return [np.hstack(c) for c in zip(*coords)]




class CustomConfig(Config):
    NAME = "object"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2
    STEPS_PER_EPOCH = 10
    DETECTION_MIN_CONFIDENCE = 0.9
    LEARNING_RATE = 0.001

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        self.add_class("object", 1, "500_front")
        self.add_class("object", 2, "500_back")


     
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        annotations1 = json.load(open('./dataset/'+subset+'/train.json'))
        # print(annotations1)
        annotations = list(annotations1['_via_img_metadata'].values())  

        annotations = [a for a in annotations if a['regions']]
        
        for a in annotations:
            a_regions = []
            m_region = len(a['regions'])
            for i in range(m_region):
                a_regions.append(a['regions'][i])
            polygons = [r['shape_attributes'] for r in a_regions] 
            objects = [s['region_attributes']['name'] for s in a_regions]
            #print("objects:",objects)
            name_dict = {"500_front": 1,"500_back": 2}

            # key = tuple(name_dict)
            num_ids = [name_dict[i] for i in objects]
     
            # num_ids = [int(n['Event']) for n in objects]
            #print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  
                image_id=a['filename'],  
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            if rr is not None and cc is not None:
                rr[rr > mask.shape[0]-1] = mask.shape[0]-1
                cc[cc > mask.shape[1]-1] = mask.shape[1]-1
            mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
def train(model):
    dataset_train = CustomDataset()
    dataset_train.load_custom("./dataset", "train")
    dataset_train.prepare()

    dataset_val = CustomDataset()
    dataset_val.load_custom("./dataset", "val")
    dataset_val.prepare()

    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
                # learning_rate=config.LEARNING_RATE,
                # epochs=250,
                # layers='heads')
                
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=500,
                layers='heads', #layers='all', 
                augmentation = imgaug.augmenters.Sequential([ 
                imgaug.augmenters.Fliplr(1), 
                imgaug.augmenters.Flipud(1), 
                imgaug.augmenters.Affine(rotate=(-45, 45)), 
                imgaug.augmenters.Affine(rotate=(-90, 90)), 
                imgaug.augmenters.Affine(scale=(0.5, 1.5)),
                imgaug.augmenters.Crop(px=(0, 10)),
                imgaug.augmenters.Grayscale(alpha=(0.0, 1.0)),
                imgaug.augmenters.AddToHueAndSaturation((-20, 20)), 
                imgaug.augmenters.Add((-10, 10), per_channel=0.5), 
                imgaug.augmenters.Invert(0.05, per_channel=True), 
                imgaug.augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), 
                
                ])
            )
				

# augmentation = imgaug.Sometimes(5/6,aug.OneOf(
                                            # [
                                            # imgaug.augmenters.Fliplr(1), 
                                            # imgaug.augmenters.Flipud(1), 
                                            # imgaug.augmenters.Affine(rotate=(-45, 45)), 
                                            # imgaug.augmenters.Affine(rotate=(-90, 90)), 
                                            # imgaug.augmenters.Affine(scale=(0.5, 1.5))
                                             # ]
                                        # ) 
                                   # )
                                    

    
				
config = CustomConfig()



model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)


weights_path = COCO_WEIGHTS_PATH
if not os.path.exists(weights_path):
  utils.download_trained_weights(weights_path)

#weights_path = model.find_last()
  
model.keras_model.load_weights(weights_path, by_name=True, skip_mismatch=True)

'''

from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="training", model_dir=DEFAULT_LOGS_DIR, config=config)

weights_path = COCO_WEIGHTS_PATH
if not os.path.exists(weights_path):
  utils.download_trained_weights(weights_path)

#weights_path = model.find_last()


import tensorflow.compat.v1 as tfc

tfc.keras.Model.load_weights(model.keras_model, weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
'''



train(model)