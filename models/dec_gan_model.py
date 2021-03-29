import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import scipy.io as sio
import numpy as np


class DecGANModel(BaseModel):
    def name(self):
        return 'DecGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['cycle_A', 'D_B', 'G_B', 'cycle_B', 'cyclemask']  #
        # visual_names_A = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B', 'postdecA', 'postdecAlung',
        #                   'postdecA_other', 'real_bone', 'real_lung', 'real_other', 'rec_Asup', 'maskbone', 'maskreal_A',
        #                   'maskrec_Asup']
        # visual_names_A = ['real_A', 'fake_B', 'rec_A', 'posrec_A', 'postdecA', 'postdecAlung', 'postdecA_other']
        visual_names_A = ['posrec_A']
        visual_names_B = []

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_Dec', 'G_Dec']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B', 'G_Dec']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type,
                                        self.gpu_ids)
        self.netG_B = networks.define_G(3, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type,
                                        self.gpu_ids)
        self.netG_Dec = networks.define_G(opt.input_nc, 3,
                                        opt.ngf, 'unet_32', opt.norm, not opt.no_dropout, opt.init_type,
                                        self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_Dec = networks.define_D(3, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.postdecAt_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_Dec.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
            # print('continue training!')
        # initial G_Dec
        self.load_networks_G_Dec('latest_net_G_Dec')
        self.print_networks(opt.verbose)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        # input_A = input['A' if AtoB else 'B']
        # input_B = input['B' if AtoB else 'A'] #1,1,255,255
        input_A = input['A']
        input_B = input['B']  # 1,1,255,255

        if self.opt.phase == 'train':
            input_X = input['X']  # 1,2,255,255
            input_E = input['E']
            if len(self.gpu_ids) > 0:
                input_X = input_X.cuda(self.gpu_ids[0])
                input_E = input_E.cuda(self.gpu_ids[0])
            self.input_X = input_X
            self.input_E = input_E

        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0])
            input_B = input_B.cuda(self.gpu_ids[0])
        self.input_A = input_A
        self.input_B = input_B

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_Dec = Variable(self.input_X)

    def test(self, alpha_bone, alpha_lung, alpha_other):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A(self.real_A)

        self.prefake_B = (self.fake_B + 1.0) / 2.0
        self.decA, self.decAc = self.netG_Dec(self.prefake_B)

        self.decA_bone = self.decA[:, 0, :, :]
        self.decA_bone.contiguous()
        self.decA_bone = self.decA_bone.view([self.decA.shape[0], 1, self.decA.shape[2], self.decA.shape[3]])
        self.decA_lung = self.decA[:, 1, :, :]
        self.decA_lung.contiguous()
        self.decA_lung = self.decA_lung.view([self.decA.shape[0], 1, self.decA.shape[2], self.decA.shape[3]])

        self.decA_other = self.prefake_B - self.decA_bone - self.decA_lung

        self.postdecA = (self.decA_bone - 0.5) / 0.5
        self.postdecAlung = (self.decA_lung - 0.5) / 0.5
        self.postdecA_other = (self.decA_other - 0.5) / 0.5

        self.postdecAt = np.zeros([self.decA.shape[0], 3, self.decA.shape[2], self.decA.shape[3]])
        self.postdecAt = torch.FloatTensor(self.postdecAt)
        self.postdecAt = self.postdecAt.cuda(self.gpu_ids[0])
        self.postdecAt = Variable(self.postdecAt)
        self.postdecAt[:, 0, :, :] = self.postdecA
        self.postdecAt[:, 1, :, :] = self.postdecAlung
        self.postdecAt[:, 2, :, :] = self.postdecA_other

        self.rec_A = self.netG_B(self.postdecAt)

        self.postdecAtp = np.zeros([self.decA.shape[0], 3, self.decA.shape[2], self.decA.shape[3]])
        self.postdecAtp = torch.FloatTensor(self.postdecAtp)
        self.postdecAtp = self.postdecAtp.cuda(self.gpu_ids[0])
        self.postdecAtp = Variable(self.postdecAtp)
        self.postdecAtp[:, 0, :, :] = self.postdecA * alpha_bone
        self.postdecAtp[:, 1, :, :] = self.postdecAlung * alpha_lung
        self.postdecAtp[:, 2, :, :] = self.postdecA_other * alpha_other

        self.posrec_A = self.netG_B(self.postdecAtp)
        #
        # tmz = self.decAo.data
        # tmp = tmz[0].cpu().float().numpy()
        # sio.savemat('./tmp.mat', {'tmp': tmp})

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_Dec(self):
        postdecAt = self.postdecAt_pool.query(self.postdecAt)
        self.loss_D_Dec = self.backward_D_basic(self.netD_Dec, self.real_Dec, postdecAt)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.fake_B = self.netG_A(self.real_A)
        # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # Forward cycle loss
        self.prefake_B = (self.fake_B + 1.0) / 2.0
        self.decA, self.decAc = self.netG_Dec(self.prefake_B)

        self.decA_bone = self.decA[:, 0, :, :]
        self.decA_bone.contiguous()
        self.decA_bone = self.decA_bone.view([self.decA.shape[0], 1, self.decA.shape[2], self.decA.shape[3]])
        self.decA_lung = self.decA[:, 1, :, :]
        self.decA_lung.contiguous()
        self.decA_lung = self.decA_lung.view([self.decA.shape[0], 1, self.decA.shape[2], self.decA.shape[3]])

        self.decA_other = self.prefake_B - self.decA_bone - self.decA_lung

        self.postdecA = (self.decA_bone - 0.5) / 0.5
        self.postdecAlung = (self.decA_lung - 0.5) / 0.5
        self.postdecA_other = (self.decA_other - 0.5) / 0.5

        self.postdecAt = np.zeros([self.decA.shape[0], 3, self.decA.shape[2], self.decA.shape[3]])
        self.postdecAt = torch.FloatTensor(self.postdecAt)
        self.postdecAt = self.postdecAt.cuda(self.gpu_ids[0])
        self.postdecAt = Variable(self.postdecAt)
        self.postdecAt[:, 0, :, :] = self.postdecA
        self.postdecAt[:, 1, :, :] = self.postdecAlung
        self.postdecAt[:, 2, :, :] = self.postdecA_other

        self.real_bone = self.real_Dec[:, 0, :, :]
        self.real_bone.contiguous()
        self.real_bone = self.real_bone.view([self.fake_B.shape[0], 1, self.fake_B.shape[2], self.fake_B.shape[3]])
        self.real_lung = self.real_Dec[:, 1, :, :]
        self.real_lung.contiguous()
        self.real_lung = self.real_lung.view([self.fake_B.shape[0], 1, self.fake_B.shape[2], self.fake_B.shape[3]])
        self.real_other = self.real_Dec[:, 2, :, :]
        self.real_other.contiguous()
        self.real_other = self.real_other.view([self.fake_B.shape[0], 1, self.fake_B.shape[2], self.fake_B.shape[3]])

        self.rec_A = self.netG_B(self.postdecAt)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        # GAN feature loss
        # self.loss_G_Dec = self.criterionGAN(self.netD_Dec(self.postdecAt), True)

        # GAN loss D_B(G_B(B))
        self.fake_A = self.netG_B(self.real_Dec)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Backward cycle loss
        self.rec_B = self.netG_A(self.fake_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # mask loss
        self.postdecAtp = np.zeros([self.decA.shape[0], 3, self.decA.shape[2], self.decA.shape[3]])
        self.postdecAtp = torch.FloatTensor(self.postdecAtp)
        self.postdecAtp = self.postdecAtp.cuda(self.gpu_ids[0])
        self.postdecAtp = Variable(self.postdecAtp)
        # self.postdecAtp[:, 0, :, :] = self.postdecA
        self.postdecAtp[:, 1, :, :] = self.postdecAlung
        self.postdecAtp[:, 2, :, :] = self.postdecA_other
        self.rec_Asup = self.netG_B(self.postdecAtp)
        self.maskbonet = 1 - self.decA_bone  # 0 ~ 1 soft mask
        self.maskbol = self.maskbonet > 0.95
        self.maskbonef = (self.maskbonet - 0.95) * 20
        self.maskbone = self.maskbonef.float() * self.maskbol.float()
        self.maskreal_A = self.real_A * self.maskbone
        maskreal_A = self.maskreal_A.detach()
        self.maskrec_Asup = self.rec_Asup * self.maskbone
        self.loss_cyclemask = self.criterionCycle(self.maskrec_Asup, maskreal_A) * 5

        # tmz = self.maskreal_A.data
        # tmp = tmz[0].cpu().float().numpy()
        # sio.savemat('./tmp.mat', {'tmp': tmp})

        # combined loss
        self.loss_G = self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_cyclemask
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.optimizer_D.zero_grad()
        self.backward_D_B()
        self.optimizer_D.step()
