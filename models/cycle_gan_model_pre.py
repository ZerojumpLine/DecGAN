import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['decB']# 'decB'
        visual_names_A = ['real_A']
        # visual_names_B = ['real_B', 'decB_bone', 'decB_lung', 'decB_other', 'decBc', 'real_bone', 'real_lung', 'real_other', 'real_Deccom']
        visual_names_B = ['real_B', 'decB_bone', 'decB_lung', 'decB_other', 'decBc']

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['S_B']
        else:  # during test time, only load Gs
            self.model_names = ['S_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netS_B = networks.define_G(opt.input_nc, 3,
                                        opt.ngf, 'unet_32', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)


        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionDec = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netS_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
            # print('continue training!')
        #initial S_B
        # self.load_networks_S_B('CP111')
        self.print_networks(opt.verbose)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        # input_A = input['A' if AtoB else 'B']
        # input_B = input['B' if AtoB else 'A'] #1,1,255,255
        input_A = input['A']
        input_B = input['B'] #1,1,255,255

		
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)	
        self.input_A = input_A
        self.input_B = input_B

        if self.opt.phase == 'train':
            input_X = input['X']  # 1,2,255,255
            input_E = input['E']
            if len(self.gpu_ids) > 0:
                input_X = input_X.cuda(self.gpu_ids[0], async=True)
                input_E = input_E.cuda(self.gpu_ids[0], async=True)
            self.input_X = input_X
            self.input_E = input_E

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_Dec = Variable(self.input_X)
        self.real_Deccom = Variable(self.input_E)

    def test(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

        self.decB, self.decBc = self.netS_B(self.real_B)

        self.decB_bone = self.decB[:, 0, :, :]
        self.decB_bone.contiguous()
        self.decB_bone = self.decB_bone.view([self.decB.shape[0], 1, self.decB.shape[2], self.decB.shape[3]])
        self.decB_lung = self.decB[:, 1, :, :]
        self.decB_lung.contiguous()
        self.decB_lung = self.decB_lung.view([self.decB.shape[0], 1, self.decB.shape[2], self.decB.shape[3]])
        self.decB_other = self.decB[:, 2, :, :]
        self.decB_other.contiguous()
        self.decB_other = self.decB_other.view([self.decB.shape[0], 1, self.decB.shape[2], self.decB.shape[3]])


    def backward_G(self):
        self.decB, self.decBc = self.netS_B(self.real_B)

        self.decB_bone = self.decB[:,0,:,:]
        self.decB_bone.contiguous()
        self.decB_bone = self.decB_bone.view([self.decB.shape[0],1,self.decB.shape[2],self.decB.shape[3]])
        self.decB_lung = self.decB[:,1,:,:]
        self.decB_lung.contiguous()
        self.decB_lung = self.decB_lung.view([self.decB.shape[0],1,self.decB.shape[2],self.decB.shape[3]])
        self.decB_other = self.decB[:, 2, :, :]
        self.decB_other.contiguous()
        self.decB_other = self.decB_other.view([self.decB.shape[0], 1, self.decB.shape[2], self.decB.shape[3]])

        self.real_bone = self.real_Dec[:, 0, :, :]
        self.real_bone.contiguous()
        self.real_bone = self.real_bone.view([self.decB.shape[0], 1, self.decB.shape[2], self.decB.shape[3]])
        self.real_lung = self.real_Dec[:, 1, :, :]
        self.real_lung.contiguous()
        self.real_lung = self.real_lung.view([self.decB.shape[0], 1, self.decB.shape[2], self.decB.shape[3]])
        self.real_other = self.real_Dec[:, 2, :, :]
        self.real_other.contiguous()
        self.real_other = self.real_other.view([self.decB.shape[0], 1, self.decB.shape[2], self.decB.shape[3]])

        self.loss_decB = self.criterionDec(self.decB.view(-1), self.real_Dec.view(-1).float()) + \
                         self.criterionDec(self.decBc.view(-1), self.real_Deccom.view(-1).float())

		
        # combined loss
        self.loss_G = self.loss_decB
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()