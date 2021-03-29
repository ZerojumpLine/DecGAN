import os.path
from data.base_dataset import BaseDataset, get_transform, get_transform2
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np
import random
import scipy.io as sio


class UnalignedDataset_test(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform = get_transform(opt)
        self.transformless = get_transform2(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = np.transpose(np.array(A_img), (2, 0, 1))
        B_img = np.transpose(np.array(B_img), (2, 0, 1))
        A = torch.FloatTensor(A_img) / 255.0
        B = torch.FloatTensor(B_img) / 255.0

        # Comment out this 2 lines if you want to train G_Dec
        A = self.transformless(A)
        B = self.transformless(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        # X = np.zeros([2,A.shape[1],A.shape[2]])
        # X[0,:,:] = C
        # X[1,:,:] = D
        # X = torch.FloatTensor(X)
        return {'A': A, 'B': B,  # 'X':X, #'E':E,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'


class UnalignedDataset_Dec(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
        self.dir_D = os.path.join(opt.dataroot, opt.phase + 'D')
        self.dir_E = os.path.join(opt.dataroot, opt.phase + 'E')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.C_paths = make_dataset(self.dir_C)
        self.D_paths = make_dataset(self.dir_D)
        self.E_paths = make_dataset(self.dir_E)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.C_paths = sorted(self.C_paths)
        self.D_paths = sorted(self.D_paths)
        self.E_paths = sorted(self.E_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.C_size = len(self.C_paths)
        self.D_size = len(self.D_paths)
        self.E_size = len(self.E_paths)

        self.transform = get_transform(opt)
        self.transformless = get_transform2(opt)

    def __getitem__(self, index):
        # A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        C_path = self.C_paths[index % self.C_size]
        D_path = self.D_paths[index % self.D_size]
        E_path = self.E_paths[index % self.E_size]
        if self.opt.serial_batches:
            index_A = index % self.A_size
        else:
            index_A = random.randint(0, self.A_size - 1)
        A_path = self.A_paths[index_A]

        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        C_img = Image.open(C_path).convert('RGB')
        D_img = Image.open(D_path).convert('RGB')
        E_img = Image.open(E_path).convert('RGB')

        A_img = np.transpose(np.array(A_img), (2, 0, 1))
        B_img = np.transpose(np.array(B_img), (2, 0, 1))
        C_img = np.transpose(np.array(C_img), (2, 0, 1))
        D_img = np.transpose(np.array(D_img), (2, 0, 1))
        E_img = np.transpose(np.array(E_img), (2, 0, 1))

        A = torch.FloatTensor(A_img) / 255.0
        B = torch.FloatTensor(B_img) / 255.0
        C = torch.FloatTensor(C_img) / 255.0
        D = torch.FloatTensor(D_img) / 255.0
        E = torch.FloatTensor(E_img) / 255.0

        # Comment out this 5 lines if you want to train G_Dec
        A = self.transformless(A)
        B = self.transformless(B)
        C = self.transformless(C)
        D = self.transformless(D)
        E = self.transformless(E)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc > 0:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
            tmp = C[0, ...] * 0.299 + C[1, ...] * 0.587 + C[2, ...] * 0.114
            C = tmp.unsqueeze(0)
            tmp = D[0, ...] * 0.299 + D[1, ...] * 0.587 + D[2, ...] * 0.114
            D = tmp.unsqueeze(0)
            tmp = E[0, ...] * 0.299 + E[1, ...] * 0.587 + E[2, ...] * 0.114
            E = tmp.unsqueeze(0)
        X = np.zeros([3, A.shape[1], A.shape[2]])
        X[0, :, :] = C
        X[1, :, :] = D
        X[2, :, :] = E
        X = torch.FloatTensor(X)
        return {'A': A, 'B': B, 'X': X, 'E': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
