## Semi-supervised Reference based sketch extraction using a contrastive learning framework


This is official implementation of the paper **"Semi-supervised Reference based sketch extraction using a contrastive learning framework"**

**Chang Wook Seo**, Amirsaman Ashtari, Junyong Noh


Journal: ACM TOG\
Conference: SIGGRAPH 2023\
Project page: https://github.com/Chanuku/Semi_ref2sketch



        
## Train
    $ python train.py --name [model_name] \
                     --model unpaired \
                     --dataroot ./datasets/[datafoler_name] \

* Download the pretrained model from google drive to train and test the model (pre-trained weights for HED and contrastive learning). After download, unzip to the checkpoints folder.
https://drive.google.com/file/d/1YbddMxgIO57gSwTvYxt-C4QraM2AAgVW/view?usp=sharing
* To can change the other settings such as gpu_ids, epochs and etc by adding the arguments. Check **base_options.py** and **train_options.py** in options folder. 
* To understand hierarchy of dataset, see **Dataset directories structure** below. 


## Test
    $ python test_dir.py --name semi_unpair \
                     --model unpaired \
                     --epoch 100 \
                     --dataroot ./datasets/ref_unpair \



## Dataset
* We released the new sketch dataset which paired to color images. Please check from URL.
https://github.com/Chanuku/4skst


## Dataset directories structure
    |   \---[dataroot]
    |       +---testA
    |       |       +---test_input1.png
    |       |       +---test_input2.png
    |       +---testB
    |       |       +---test_groundtruth1.png #not necessary for testing
    |       |       +---test_groundtruth2.png #not necessary for testing
    |       +---testC
    |       |       +---style1.png
    |       |       +---style2.png
    |       +---trainA
    |       |       +---train_input1.png
    |       |       +---train_input2.png
    |       +---trainB
    |       |       +---train_groundtruth1.png
    |       |       +---train_groundtruth2.png
    
    #dataset doesn't have to be paired, model can be trained with unpaired dataset