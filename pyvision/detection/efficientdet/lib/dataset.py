import os 
import torch 
import numpy as np  

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

import cv2 

class CustomDataset(Dataset):

    def __init__(self, root_dir, img_dir="images", set_name="train2017", transform=None):

        self.root_dir = root_dir
        self.img_dir = img_dir
        self.set_name = set_name
        self.transform  =  transform

        self.coco_tool = COCO(os.path.join(self.root_dir, 'annotations', 'instances_'+self.set_name+'.json'))
        self.image_ids = self.coco_tool.getImgIds() 

        self.load_classes() 
    
    def load_classes(self):

        categories = self.coco_tool.loadCats(self.coco_tool.getCatIds())
        categories.sort(key = lambda x: x["id"])
        
        # load name -> label
        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {} 
        for category in categories:
            self.coco_labels[len(self.classes)] = category['id']
            self.coco_labels_inverse[category['id']] = len(self.classes)
            self.classes[category['name']] = len(self.classes)
        
        # load label -> name
        self.labels = {} 
        for key, value in self.classes.items():
            self.labels[value] = key 
        

    def load_image(self, idx):

        img_info = self.coco_tool.loadImgs(self.image_ids[idx])[0]
        img_path = os.path.join(
            self.root_dir, self.img_dir, self.set_name, img_info['file_name']
        )
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0 

        return img 

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]
    
    def num_classes(self):
        return len(self.classes)

    def load_annotations(self, idx):

        anno_ids = self.coco_tool.getAnnIds(
            imgIds=self.image_ids[idx], iscrowd=False
        )
        annotations = np.zeros((0, 5))

        # if some images miss annotations 
        if len(anno_ids) == 0:
            return annotations
        
        # parsing the annotations here 
        coco_annotations = self.coco_tool.loadAnns(anno_ids)
        for idx, a in enumerate(coco_annotations):

            # skip the annotations that have no height/width
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue 
            
            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)
        
        # transform [x, y, w, h] -> [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations    
    

    def __len__(self):
        return len(self.image_ids)

    
    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        
        data = {
            "img": img, 
            "annot": annot
        }
        
        if self.transform:
            data = self.transform(data)

        return data


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, common_size=512):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((common_size, common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}

