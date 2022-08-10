from data.base_dataset import BaseDataset, get_transform
import numpy as np
import nibabel as nib
from data.image_folder import make_dataset
import torch
import os.path


class LesionTestBDataset(BaseDataset):
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
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')                 # create a path '/path/to/data/trainB'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.B_size = len(self.B_paths)  # get the size of dataset B
        input_nc = self.opt.output_nc    # get the number of channels of input image
        output_nc = self.opt.input_nc    # get the number of channels of output image
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
        B_path = self.B_paths[index % self.B_size]
        B_img = nib.load(B_path).get_fdata().squeeze()
        B = self.augment(B_img, dataset="B", phase=self.phase)

        return {'B': B, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images."""
        return self.B_size

    def augment(self, img, dataset, phase):
        if phase == "test":        
            if dataset=="A" or dataset=="B":
                img = self._normalise01(img)
                img = np.rot90(img, k=1)
            if dataset=="C":
                img = np.rot90(img, k=3)

        return torch.from_numpy(img[None,:].copy()).float()

    def _normalise(self, img):
        return (img - img.mean())/(img.std() + .001)

    def _normalise01(self, img):
        return (img - img.min())/(img.max() - img.min())