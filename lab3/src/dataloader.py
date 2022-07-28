

from matplotlib import transforms
import pandas as pd
from torch.utils import data
import numpy as np
import tqdm
from PIL import Image
from torchvision import transforms, models
import torch
import os
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import recall_score
from tqdm import tqdm


def getData(mode):
    if mode == 'train':
        img = pd.read_csv(os.getcwd() + '/data/train_img.csv')
        label = pd.read_csv(os.getcwd() + '/data/train_label.csv')
#         img = pd.read_csv('../data/train_img.csv')
#         label = pd.read_csv('../data/train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv(os.getcwd() + '/data/test_img.csv')
        label = pd.read_csv(os.getcwd() + '/data/test_label.csv')
#         img = pd.read_csv('../data/test_img.csv')
#         label = pd.read_csv('../data/test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


# +
class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))
        self.img_list = list();
        
        

        
        for i in tqdm(range(len(self.img_name))):
            path = self.root + self.img_name[i] + ".jpeg"
            img = Image.open(path)

#             m_img = torch.mean(img, [1, 2])
#             std_img = torch.std(img, [1, 2])

            
            self.img_list.append(img)

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
            step1. Get the image path from 'self.img_name' and load it.
                hint : path = root + self.img_name[index] + '.jpeg'
            step2. Get the ground truth label from self.label
                
            step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                
                In the testing phase, if you have a normalization process during the training phase, you only need 
                to normalize the data. 
                
                hints : Convert the pixel value to [0, 1]
                        Transpose the image shape from [H, W, C] to [C, H, W]
                        
            step4. Return processed image and label
        """
        
        label = self.label[index];

        img = self.img_list[index]
        img_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize((0.3749, 0.2602, 0.1857),
                                 (0.2526, 0.1780, 0.1291)),
        ]
        )
        img = img_transform(img)

        return img, label
