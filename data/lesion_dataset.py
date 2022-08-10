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
from skimage.filters import gaussian


class LesionDataset(BaseDataset):
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
        self.phase = opt.phase
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        #NOTE: added directories for lesion
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')  # create a path '/path/to/data/trainC'
        
        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))    # load images from '/path/to/data/trainC'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.C_size = len(self.C_paths)  # get the size of dataset C
        input_nc = self.opt.output_nc    # get the number of channels of input image
        output_nc = self.opt.input_nc    # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.transform_C = get_transform(self.opt, grayscale=(output_nc == 1))


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, C, A_paths, B_paths and C_paths
            A (tensor)       -- an image in the A domain
            B (tensor)       -- an image in the B domain
            C (tensor)       -- an image in the C domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
            C_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
            index_C = index % self.C_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
            index_C = random.randint(0, self.C_size - 1)
        B_path = self.B_paths[index_B]
        C_path = self.C_paths[index_C]

        if A_path.endswith('.npz') and B_path.endswith('.npz'):
            A_img = np.load(A_path)["arr_0"]
            B_img = np.load(B_path)["arr_0"]
            C_img = np.load(C_path)["arr_0"]

        elif A_path.endswith('.nii.gz') and B_path.endswith('.nii.gz'):
            A_img = nib.load(A_path).get_fdata().squeeze()
            B_img = nib.load(B_path).get_fdata().squeeze()
            C_img = nib.load(C_path).get_fdata().squeeze()

        A = self.augment(A_img, dataset="A", phase=self.phase)
        B = self.augment(B_img, dataset="B", phase=self.phase)
        C = self.augment(C_img, dataset="C", phase=self.phase)        
        return {'A': A, 'B': B, 'C': C, 'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path}        # path = 'temp'    # needs to be a string

    def __len__(self):
        """Return the total number of images."""
        return max(self.A_size, self.B_size)


    def smooth(self, img, sigma=1):
        return gaussian(img, sigma=sigma)

    def augment(self, img, dataset, phase="train"):

        if phase == "train":
            if dataset=="A" or dataset=="B":
                img = self._normalise01(img)    
                img = self._elastic_deform(img)
                img = np.rot90(img)
            if dataset=="C":
                img = self.smooth(img, sigma=2)
            img = self._fliplr(img)

        return torch.from_numpy(img[None,:].copy()).float()

    def _normalise(self, img):
        return (img - img.mean())/(img.std() + .001)

    def _normalise01(self, img):
        return (img - img.min())/(img.max() - img.min())

    def _fliplr(self, img, p=.5):
        if np.random.rand() > p:
            return np.fliplr(img)
        return img

    def _elastic_deform(self, img, p=.5):
        if np.random.rand() > p:
            rand_sig = random.randint(1,2)
            return elasticdeform.deform_random_grid(img, sigma=rand_sig, points=3)
        return img
