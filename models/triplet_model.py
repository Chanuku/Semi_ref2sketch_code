import torch
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool


class TripletModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='triplet', dataset_mode='triplet')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):

        BaseModel.__init__(self, opt)

        self.loss_names = ['G_triplet']
        self.visual_names = ['x','y']

        if self.isTrain:
            self.model_names = ['G']
        else: 
            self.model_names = ['G']
        self.netG = networks.define_G(1, 1, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            
            self.triplet = torch.nn.TripletMarginLoss(margin=3.0)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_C = input['C'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']



    def forward(self):
        self.x,self.y,self.z = self.netG(self.real_A,self.real_B,self.real_C)


    def backward_G(self):
        self.loss_G_triplet_1 = self.triplet(self.x,self.y,self.z)
        self.loss_G_triplet = self.loss_G_triplet_1

        self.loss_G = self.loss_G_triplet
        self.loss_G.backward()

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()        
        self.backward_G()                  
        self.optimizer_G.step()             
