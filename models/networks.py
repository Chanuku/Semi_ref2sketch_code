import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from models.layer import *


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'ref_unpair_cbam_cat':
        net = ref_unpair(input_nc, output_nc, ngf, norm ='inorm',status='ref_unpair_cbam_cat')
    elif netG == 'ref_unpair_recon':
        net = ref_unpair(input_nc, output_nc, ngf, norm ='inorm', status='ref_unpair_recon')
    elif netG == 'triplet':
        net = triplet(input_nc, output_nc, ngf, norm ='inorm')

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)





class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        eps = 1e-5	
        mean_x = torch.mean(x, dim=[2,3])
        mean_y = torch.mean(y, dim=[2,3])

        std_x = torch.std(x, dim=[2,3])
        std_y = torch.std(y, dim=[2,3])

        mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)
        mean_y = mean_y.unsqueeze(-1).unsqueeze(-1)

        std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps
        std_y = std_y.unsqueeze(-1).unsqueeze(-1) + eps

        out = (x - mean_x)/ std_x * std_y + mean_y


        return out

class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()

        self.moduleVggOne = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleVggTwo = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleVggThr = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleVggFou = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleVggFiv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.moduleScoreOne = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreTwo = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreThr = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFou = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFiv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.moduleCombine = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, tensorInput):

        tensorBlue = (tensorInput[:, 2:3, :, :] * 255.0) - 104.00698793
        tensorGreen = (tensorInput[:, 1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (tensorInput[:, 0:1, :, :] * 255.0) - 122.67891434
        tensorInput = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 1)
        
        tensorVggOne = self.moduleVggOne(tensorInput)
        tensorVggTwo = self.moduleVggTwo(tensorVggOne)
        tensorVggThr = self.moduleVggThr(tensorVggTwo)
        tensorVggFou = self.moduleVggFou(tensorVggThr)
        tensorVggFiv = self.moduleVggFiv(tensorVggFou)
        
        tensorScoreOne = self.moduleScoreOne(tensorVggOne)
        tensorScoreTwo = self.moduleScoreTwo(tensorVggTwo)
        tensorScoreThr = self.moduleScoreThr(tensorVggThr)
        tensorScoreFou = self.moduleScoreFou(tensorVggFou)
        tensorScoreFiv = self.moduleScoreFiv(tensorVggFiv)
        
        tensorScoreOne = nn.functional.interpolate(input=tensorScoreOne, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreTwo = nn.functional.interpolate(input=tensorScoreTwo, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreThr = nn.functional.interpolate(input=tensorScoreThr, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreFou = nn.functional.interpolate(input=tensorScoreFou, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreFiv = nn.functional.interpolate(input=tensorScoreFiv, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)

        return self.moduleCombine(torch.cat([ tensorScoreOne, tensorScoreTwo, tensorScoreThr, tensorScoreFou, tensorScoreFiv ], 1))
        #return self.moduleCombine(torch.cat([ tensorScoreOne, tensorScoreTwo, tensorScoreThr, tensorScoreOne, tensorScoreTwo ], 1))

        #return torch.sigmoid(tensorScoreOne),torch.sigmoid(tensorScoreTwo),torch.sigmoid(tensorScoreThr),torch.sigmoid(tensorScoreFou),torch.sigmoid(tensorScoreFiv),self.moduleCombine(torch.cat([ tensorScoreOne, tensorScoreTwo, tensorScoreThr, tensorScoreFou, tensorScoreFiv ], 1))
        #return torch.sigmoid(tensorScoreTwo)



def define_HED(init_weights_, gpu_ids_=[]):
    net = HED()

    if len(gpu_ids_) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids_[0])
        net = torch.nn.DataParallel(net, gpu_ids_)  # multi-GPUs
    
    if not init_weights_ == None:
        device = torch.device('cuda:{}'.format(gpu_ids_[0])) if gpu_ids_ else torch.device('cpu')
        print('Loading model from: %s'%init_weights_)
        state_dict = torch.load(init_weights_, map_location=str(device))
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(state_dict)
        else:
            net.load_state_dict(state_dict)
        print('load the weights successfully')

    return net


def define_styletps(init_weights_, gpu_ids_=[],shape=False):
    net = None
    if shape == False:
        net = triplet()
    if len(gpu_ids_) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids_[0])
        net = torch.nn.DataParallel(net, gpu_ids_)  # multi-GPUs

    if not init_weights_ == None:
        device = torch.device('cuda:{}'.format(gpu_ids_[0])) if gpu_ids_ else torch.device('cpu')
        print('Loading model from: %s'%init_weights_)
        state_dict = torch.load(init_weights_, map_location=str(device))
    if isinstance(net, torch.nn.DataParallel):
        net.module.load_state_dict(state_dict)
    else:
        net.load_state_dict(state_dict)
    print('load the weights successfully')

    return net

class triplet(nn.Module):
    def __init__(self): #mnblk=4
        super(triplet, self).__init__()


        # self.channels = nch_in
        self.nch_in = 1
        self.nch_out = 1
        self.nch_ker = 64
        self.norm = 'bnorm'
        #self.nblk = nblk

        if self.norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True
  
        self.conv0 = CNR2d(self.nch_in, self.nch_ker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)
        self.conv1 = CNR2d(self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.conv2 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)


        self.final_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(256, 128)


    def forward(self,x,y,z):

        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final_pool(x)
        x = torch.flatten(x,1)
        x = self.linear(x)

        y = self.conv0(y)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.final_pool(y)
        y = torch.flatten(y,1)
        y = self.linear(y)

        z = self.conv0(z)
        z = self.conv1(z)
        z = self.conv2(z)
        z = self.final_pool(z)
        z = torch.flatten(z,1)
        z = self.linear(z)
            

        return x,y,z 




class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class ref_unpair(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', nblk=4, status='ref_unpair'):
        super(ref_unpair, self).__init__()

        nch_ker=64
        #self.channels = nch_in
        self.nch_in = nch_in
        self.nchs_in = 1
        self.status = status

        if self.status == 'ref_unpair_recon':
            self.nch_out = 3
            self.nch_in = 1
        else:
            self.nch_out = 1

        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk
        self.dec0 = []

        if status == 'ref_unpair_cbam_cat':
            self.cbam_c = CBAM(nch_ker*8,16,3,cbam_status="channel")
            self.cbam_s = CBAM(nch_ker*8,16,3,cbam_status="spatial")

            self.enc1_s = CNR2d(self.nchs_in, self.nch_ker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)
            self.enc2_s = CNR2d(self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
            self.enc3_s = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
            self.enc4_s = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)


        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True
        
        self.enc1_c = CNR2d(self.nch_in, self.nch_ker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)
        self.enc2_c = CNR2d(self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.enc3_c = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.enc4_c = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)


        if status == 'ref_unpair_cbam_cat':
            self.res_cat1 = ResBlock_cat(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')
            self.res_cat2 = ResBlock_cat(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')
            self.res_cat3 = ResBlock_cat(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')
            self.res_cat4 = ResBlock_cat(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')

        if self.nblk and status !='ref_unpair_cbam_cat':
            res = []
            for i in range(self.nblk):
                res += [ResBlock(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')]
            self.res1 = nn.Sequential(*res)
        
        #self.dec0 += [DECNR2d(16 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)]        
        self.dec0 += [DECNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)]
        self.dec0 += [DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)]
        self.dec0 += [DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)]
        self.dec0 += [DECNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)]
        self.dec0 += [nn.Conv2d(1 * self.nch_ker, self.nch_out, kernel_size=3, stride=1, padding=1)]

        self.dec = nn.Sequential(*self.dec0)


    def forward(self, content,style):


        content_cs = self.enc1_c(content)
        content_cs = self.enc2_c(content_cs)
        content_cs = self.enc3_c(content_cs)
        content_cs = self.enc4_c(content_cs)
        #content_cs = self.enc5_c(content_cs)


        if self.status == 'ref_unpair_cbam_cat':
            cbam_content_cs = self.cbam_s(content_cs)
            sp_content_cs = content_cs + cbam_content_cs

            style_cs = self.enc1_s(style)
            style_cs = self.enc2_s(style_cs)
            style_cs = self.enc3_s(style_cs)
            style_cs = self.enc4_s(style_cs)

            cbam_style_cs = self.cbam_c(style_cs)
            ch_style_cs = style_cs + cbam_style_cs

            content_output =self.adaptive_instance_normalization(content_cs ,style_cs)
            cbam_content_output =self.adaptive_instance_normalization(sp_content_cs ,ch_style_cs)

            content_output = self.res_cat1(content_output,cbam_content_output)
            content_output = self.res_cat2(content_output,cbam_content_output)
            content_output = self.res_cat3(content_output,cbam_content_output)
            content_output = self.res_cat4(content_output,cbam_content_output)


        else:
            content_output = content_cs

        if self.nblk and self.status !='ref_unpair_cbam_cat':
            content_cs = self.res1(content_output)


        content_output = self.dec(content_output)


        content_output = torch.tanh(content_output)

        return content_output


    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std


    def adaptive_instance_normalization(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)







def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

import math


class CBAM(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio, kernel_size,cbam_status):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.channel_attention = ChannelAttention_nopara(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention_nopara(kernel_size)
        self.status = cbam_status

    def forward(self, x):
        ## We don't use cbam in this version
        if self.status == "cbam":
            chan_att = self.channel_attention(x) 
            fp = chan_att * x
            spat_att = self.spatial_attention(fp)
            fpp = spat_att * fp

        if self.status == "spatial":
            spat_att = self.spatial_attention(x) #* s_para_1d
            fpp = spat_att * x
        if self.status == "channel":
            chan_att = self.channel_attention(x) #* c_para_1d
            fpp = chan_att * x

        return fpp #,c_wgt,s_wgt

class SpatialAttention_nopara(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention_nopara, self).__init__()
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = kernel_size, padding= int((kernel_size-1)/2))

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim = 1)
        conv = self.conv(pool)
        conv = conv.repeat(1,x.size()[1],1,1)
        att = torch.sigmoid(conv)        
        return att
    def agg_channel(self, x, pool = "max"):
        b,c,h,w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0,2,1)
        if pool == "max":
            x = F.max_pool1d(x,c)
        elif pool == "avg":
            x = F.avg_pool1d(x,c)
        x = x.permute(0,2,1)
        x = x.view(b,1,h,w)
        return x

class ChannelAttention_nopara(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention_nopara, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in/ float(self.reduction_ratio))
        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel )
        max_pool = F.max_pool2d(x, kernel)
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)
        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)
        pool_sum = avg_pool_bck + max_pool_bck
        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)
        #out = sig_pool.repeat(1,1,kernel[0], kernel[1])

        return sig_pool





