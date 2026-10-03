"""
Dataset skeleton for Oxford-IIIT Pet.
the dataset can be downloaded from https://www.robots.ox.ac.uk/~vgg/data/pets/
"""

from torch.utils.data import Dataset
import os
from PIL import Image
import torch 
import numpy as np
import xml.etree.ElementTree as ET

class OxfordIIITPetDataset(Dataset):
    """
    Oxford-IIIT Pet multi-task dataset loader skeleton.
    Args:
        root_dir (str): Root directory of the dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root_dir, split='trainval', transform = None):

        # store paths to image, masks, bboxes and the transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        self.xml_dir = os.path.join(root_dir, 'annotations', 'xmls')
        self.transform = transform

        # determine which annotation file to read based on the split
        if split == 'trainval':
            annotations_file = os.path.join(root_dir, 'annotations', 'list.txt')
        elif split == 'test':
            annotations_file = os.path.join(root_dir, 'annotations', 'test.txt')
        else:
            raise ValueError("Invalid split. Expected 'trainval' or 'test'.")

        # read and clean the annotations
        self.annotations = []
        with open(annotations_file, 'r') as f:
            for l in f:
                l = l.strip()
                if l and not l.startswith('#'):
                    name = l.split()[0]
                    xml_path = os.path.join(self.xml_dir, name + '.xml')
                    if os.path.exists(xml_path):
                        self.annotations.append(l)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """ 
        Function to get a sample from the dataset. It should return a tuple (image, label, bbox, mask) where:
        - image: the input image as a tensor
        - label: the class label as a tensor
        - bbox: the bounding box coordinates as a tensor (normalized to [0, 1])
        - mask: the segmentation mask as a tensor

        """

        # parse the annotation line to get the image name and label
        data = self.annotations[idx].split()

        image_name = data[0] + ".jpg"
        mask_name =  data[0] + ".png"
        xml_name = data[0] + ".xml"
        xml_path = os.path.join(self.xml_dir, xml_name)

        # load the image and convert to RGB
        image = Image.open(os.path.join(self.images_dir, image_name)).convert('RGB')
        
        # convert to numpy arrays for transformations and tensor conversion
        image = np.array(image)
    
        # lable 0-based
        label = int(data[1]) - 1

        # parse the XML to get bounding box coordinates
        tree = ET.parse(xml_path)
        root = tree.getroot()

        obj = root.find('.//object') 
        if obj is None:
            raise ValueError(f"No object found in XML file: {xml_path}")

        bndbox = obj.find('bndbox')

        x_min = int(bndbox.find('xmin').text)
        y_min = int(bndbox.find('ymin').text)
        x_max = int(bndbox.find('xmax').text)
        y_max = int(bndbox.find('ymax').text)
        
        # # convert to COCO format [x_min, y_min, w, h]
        # xc = (x_min + x_max) / 2
        # yc = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        bbox = [x_min, y_min, w, h] 

        # load the mask and convert to binary (0 for background & border, 1 for pet)
        mask = Image.open(os.path.join(self.masks_dir, mask_name)).convert('L')
        mask = np.array(mask)

        mask = (mask - 1).astype(np.int64)  # values are  1, 2, 3 -> convert to 0, 1, 2 (0 for background, 1 for pet, 2 for border)

        # apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask, bboxes=[bbox])
            image = augmented['image']
            mask = augmented['mask']
            bbox = list(augmented['bboxes'][0])
        
        # convert mask to tensor if needed and ensure it's of type long for loss functions like CrossEntropyLoss
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)
        else:
            mask = mask.long()

        # convert image to tensor and permute to CxHxW format if needed
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) # convert to CxHxW 
        else:
            image = image


        #  convert bbox to tensor
        bbox = torch.tensor(bbox, dtype=torch.float32)

        # convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label, bbox, mask