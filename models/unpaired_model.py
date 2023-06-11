import torch
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool
from .perceptual import VGGPerceptualLoss,VGGstyleLoss
import numpy as np

class UnpairedModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='ref_unpair_cbam_cat',netG2='ref_unpair_recon', dataset_mode='unaligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1_1','G_Rec','G_line','D_real', 'D_fake'] 
        self.visual_names = ['real_A', 'content_output', 'real_B']

        if self.isTrain:
            self.model_names = ['G_A','G_B', 'D']
        else:  # during test time, only load G
            self.model_names = ['G_A','G_B']
        # define networks (both generator and discriminator)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG2, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(1, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.styletps = networks.define_styletps(init_weights_='./checkpoints/contrastive_pretrained.pth', gpu_ids_=self.gpu_ids,shape=False)
            self.HED = networks.define_HED(init_weights_='./checkpoints/network-bsds500.pytorch', gpu_ids_=self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1_1 = torch.nn.L1Loss()
            self.criterionL1_2 = torch.nn.L1Loss()
            self.criterionL1_3 = torch.nn.L1Loss()
            self.per_loss_1 = VGGPerceptualLoss().to(self.device)
            self.per_loss_2 = VGGPerceptualLoss().to(self.device)
            self.per_loss_3 = VGGPerceptualLoss().to(self.device)

            self.optimizer_GA = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_GB = torch.optim.Adam(self.netG_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_GA)
            self.optimizers.append(self.optimizer_GB)

            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']



    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.content_output = self.netG_A(self.real_A,self.real_B)
        self.rec_output = self.netG_B(self.content_output,self.content_output)


    def update_process(self, epoch, total_epoch):
        self.epoch_count = epoch
        self.epoch_count_total = total_epoch
    def backward_D(self):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = self.netD(self.content_output.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        pred_fake = self.netD(self.content_output)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.content_output_line=self.HED(self.real_A)
        self.rec_output_line=self.HED(self.rec_output)
        self.t1, self.t2, _=self.styletps(self.content_output, self.real_B,self.real_B)


        decay_lambda = 5 - ((self.epoch_count*4.5)/self.epoch_count_total)
        self.loss_G_L1_1 = self.criterionL1_1(self.t1, self.t2) *10 
        self.loss_G_Rec = self.per_loss_2(self.real_A, self.rec_output) *decay_lambda 
        self.loss_G_line = self.per_loss_3(self.content_output_line, self.rec_output_line) *decay_lambda 


        self.loss_G = self.loss_G_GAN + self.loss_G_L1_1 + self.loss_G_Rec + self.loss_G_line
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()      # calculate gradients for backward_D_unsuper
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_GA.zero_grad()        # set G's gradients to zero
        self.optimizer_GB.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_GA.step()             # udpate G's weights
        self.optimizer_GB.step()             # udpate G's weights
