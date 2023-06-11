import os

from torch import index_copy
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
#from tps_transformation import tps_transform
import numpy as np
import torch
import torchvision.transforms as transforms

class tpsdataset(BaseDataset):

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')  # create a path '/path/to/data/trainC'
        self.dir_D = os.path.join(opt.dataroot, opt.phase + 'D')  # create a path '/path/to/data/trainD'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))    # load images from '/path/to/data/trainC'
        self.D_paths = sorted(make_dataset(self.dir_D, opt.max_dataset_size))    # load images from '/path/to/data/trainD'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.C_size = len(self.C_paths)  # get the size of dataset C
        self.D_size = len(self.D_paths)  # get the size of dataset D

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 3))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 3))
        self.transform_C = get_transform(self.opt, grayscale=(output_nc == 3))
        self.transform_D = get_transform(self.opt, grayscale=(output_nc == 3))
        # self.transform_A = get_transform(self.opt)#, grayscale=(input_nc == 1))
        # self.transform_B = get_transform(self.opt)#, grayscale=(output_nc == 1))
        # self.transform_C = get_transform(self.opt)#, grayscale=(output_nc == 1))
        # self.transform_D = get_transform(self.opt)#, grayscale=(output_nc == 1))


    def __getitem__(self, index):
        domain_list = ['A_paths','B_paths','C_paths','D_paths']
        domain_choice = random.randint(0,3)
        if domain_list[domain_choice] == 'A_paths':
            A_index = index % self.A_size
            A_path = self.A_paths[A_index]
            B_path = self.A_paths[random.randint(0, self.A_size - 1)]
            num_rand = random.randint(0,2)
            if num_rand == 0:
                C_path = self.B_paths[A_index]
            elif num_rand == 1:
                C_path = self.C_paths[A_index]
            elif num_rand == 2:
                C_path = self.D_paths[A_index]
        elif domain_list[domain_choice] == 'B_paths':
            A_index = index % self.A_size
            A_path = self.B_paths[A_index]
            B_path = self.B_paths[random.randint(0, self.A_size - 1)]
            num_rand = random.randint(0,2)
            if num_rand == 0:
                C_path = self.A_paths[A_index]
            elif num_rand == 1:
                C_path = self.C_paths[A_index]
            elif num_rand == 2:
                C_path = self.D_paths[A_index]
        elif domain_list[domain_choice] == 'C_paths':
            A_index = index % self.A_size
            A_path = self.C_paths[A_index]
            B_path = self.C_paths[random.randint(0, self.A_size - 1)]
            num_rand = random.randint(0,2)
            if num_rand == 0:
                C_path = self.B_paths[A_index]
            elif num_rand == 1:
                C_path = self.A_paths[A_index]
            elif num_rand == 2:
                C_path = self.D_paths[A_index]
        elif domain_list[domain_choice] == 'D_paths':
            A_index = index % self.A_size
            A_path = self.D_paths[A_index]
            B_path = self.D_paths[random.randint(0, self.A_size - 1)]
            num_rand = random.randint(0,2)
            if num_rand == 0:
                C_path = self.B_paths[A_index]
            elif num_rand == 1:
                C_path = self.C_paths[A_index]
            elif num_rand == 2:
                C_path = self.A_paths[A_index]
        

        A_img = Image.open(A_path).convert('L')
        B_img = Image.open(B_path).convert('L')
        C_img = Image.open(C_path).convert('L')

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        C = self.transform_B(C_img)




        return {'A': A, 'B': B, 'C': C, 'A_paths': A_path, 'B_paths': B_path , 'C_paths': C_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
