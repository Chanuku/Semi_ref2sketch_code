import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import torch


def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path,w,h, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    image_pil = image_pil.resize((w,h))
    image_pil.save(image_path)


def save_img(image_tensor,w,h ,filename):
    image_pil = tensor2im(image_tensor)

    save_image(image_pil, filename,w,h, aspect_ratio=1.0)
    print("Image saved as {}".format(filename))

def load_img(filepath):
    img = Image.open(filepath).convert('L')
    #print(img.size)
    width = img.size[0]
    height = img.size[1]
    img = img.resize((512, 512), Image.BICUBIC)
    return img,width,height


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    device = torch.device("cuda:0")
    ref_dir = "{}/testC/".format(opt.dataroot)
    ref_names =sorted(os.listdir(ref_dir))

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    transform_list = [transforms.ToTensor(),
                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                    transforms.Normalize([0.5], [0.5])]
    transform = transforms.Compose(transform_list)    
    if opt.eval:
        model.eval()
    for j in ref_names:
        #k=0
        for i, data in enumerate(dataset):
                #print(data)
                data['B_paths'] = ref_dir + j
                #print(data['B']) 
                reference,_,_ = load_img(ref_dir + j)
                style_img = transform(reference)
                data['B'] = style_img
                data['B'] = data['B'].unsqueeze(0).to(device)

                #print(style_img.shape)
                model.set_input(data)  # unpack data from data loader
                model.test()           # run inference
                visuals = model.get_current_visuals()  # get image results
                img_path = model.get_image_paths()     # get image paths
                img_path = img_path[0]

                _,w,h = load_img(img_path)
                names =os.path.split(img_path)
                #print(names[1])
                result_dir = "{}/{}/{}/{}/".format(opt.results_dir,opt.name,'dir_free',j)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                result_name = names[1]
                save_img(visuals['content_output'].cpu(),w,h,result_dir+result_name)
                #k+=1
