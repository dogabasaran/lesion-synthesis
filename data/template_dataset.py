"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
import numpy as np
import nibabel as nib
from data.image_folder import make_dataset
import torch
import os.path
import random
import util.util as util
import elasticdeform
import random
# from PIL import Image

'''
Write a method for testing so that we don't do data augmentation for that
'''


class TemplateDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # parser.set_defaults(max_dataset_size=100, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        
        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        input_nc = self.opt.output_nc# if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc# if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        if A_path.endswith('.npz') and B_path.endswith('.npz'):
            A_img = np.load(A_path)["arr_0"]
            B_img = np.load(B_path)["arr_0"]

        elif A_path.endswith('.nii.gz') and B_path.endswith('.nii.gz'):
            A_img = nib.load(A_path).get_fdata()
            B_img = nib.load(B_path).get_fdata()

        A = self.augment(A_img)
        B = self.augment(B_img)
        
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}        # path = 'temp'    # needs to be a string

    def __len__(self):
        """Return the total number of images."""
        return max(self.A_size, self.B_size)

    def augment(self, img):
        img = self._fliplr(img)
        img = self._elastic_deform(img)
        img = img[None, :]
        return torch.from_numpy(img.copy()).float()

    def _fliplr(self, img, p=.5):
        if np.random.rand() > p:
            return np.fliplr(img)
        return img

    def _elastic_deform(self, img, p=.5):
        if np.random.rand() > p:
            rand_sig = random.randint(1,2)
            return elasticdeform.deform_random_grid(img, sigma=rand_sig, points=3)
        return img