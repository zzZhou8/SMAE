import os
import numpy as np
import torch
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from loader import get_loader
from solver import Solver

parser = argparse.ArgumentParser()

# The mode parameter determines whether the current model is in the train or test phase.
parser.add_argument('--mode', type=str, default='test')
# base_model True means you want to use your own base data to train the base model,
# False means you want to use the fine-tuned training process.
parser.add_argument('--base_model', type=bool, default=True)

#whether to save the relative intensity relationship between the data 
parser.add_argument('--Keep_relative_intensity', type=bool, default=False)
# Selection of data used for base model training
# This training data needs to be saved in npy_img//base_model_data and named in the form of base_key_word_lefts.npy
parser.add_argument('--base_key_word', type=str, default='mouse_radius')
# This training data needs to be saved in npy_img//finetune_model_data and named in the form of finetune_key_word_lefts.npy
parser.add_argument('--finetune_key_word', type=str, default='mouse_radius')
# If HSNR data does not exist as a reference, this method can only complete the preservation of data after noise reduction
parser.add_argument('--exist_target_data', type=bool, default=True)

# If you have HSNR data as a reference, you can test the noise reduction effect of all the saved models here, 
# including PSNR, SSIM, RMSE three indexes, 
# as for the calculation of physical information related to the calculation you can manually add or save the data before processing.
parser.add_argument('--Loop_test', type=bool, default=False)

#Decide whether to use data augmentation methods for training and testing during the model training and testing phase
parser.add_argument('--train_data_aug', type=bool, default=True)
parser.add_argument('--test_data_aug', type=bool, default=True)

#Where the data and the training model are stored
parser.add_argument('--saved_path', type=str, default='./npy_img/')
parser.add_argument('--save_path', type=str, default='./save/')
# Save png photos of the original, opt and target images with corresponding RMSE, PSNR and SSIM metrics when reference data is available.
parser.add_argument('--result_fig', type=bool, default=True)

parser.add_argument('--norm_range_min', type=float, default=0)
parser.add_argument('--norm_range_max', type=float, default=255)
parser.add_argument('--trunc_min', type=float, default=0)
parser.add_argument('--trunc_max', type=float, default=255)

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--print_iters', type=int, default=10)

#Start of learning rate decay epoch
parser.add_argument('--base_decay_epochs', type=int, default=20)
parser.add_argument('--finetune_decay_epochs', type=int, default=20)

# Decide on the total number of training epochs for the base model and the fine-tuned model
parser.add_argument('--base_num_epochs', type=int, default=505)
parser.add_argument('--finetune_num_epochs', type=int, default=505)
# Decide on the total number of saving epochs for the base model and the fine-tuned model
parser.add_argument('--base_save_epochs', type=int, default=20)
parser.add_argument('--finetune_save_epochs', type=int, default=5)

# base_test_epochs base_test_epochs is the number of rounds of the model you are testing, 
# and the base model of that round will be used as a starting point for fine-tuning model training
parser.add_argument('--base_test_epochs', type=int, default=260)
parser.add_argument('--finetune_test_epochs', type=int, default=25)

# Determining learning rates for base and fine-tuned models
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--finetune_lr', type=float, default=5e-5)
# Decide on the device to be used and whether it is multi-gpu parallel
parser.add_argument('--device', type=str)
parser.add_argument('--multi_gpu', type=bool, default=False)

args = parser.parse_args(args=[])

def main(args):
    
    if not os.path.exists(args.save_path):#没有就产生一个
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:

        if args.base_model:
            fig_path = os.path.join(args.save_path, 'fig_base')
        else:
            fig_path = os.path.join(args.save_path, 'fig_finetune') 

        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    print('Data in preparation')
    data_loader = get_loader(mode=args.mode,#默认为train
                             base_model=args.base_model,
                             exist_target_data=args.exist_target_data,
                             saved_path=args.saved_path,#'./npy_img/'
                             batch_size=(args.batch_size if args.mode=='train' else 1),#训练时为16，测试时为1
                             train_data_aug = args.train_data_aug,
                             test_data_aug = args.test_data_aug,
                             base_key_word=args.base_key_word,
                             finetune_key_word=args.finetune_key_word,
                             Keep_relative_intensity=args.Keep_relative_intensity
                             )
    print('Completion of data preparation')
    solver = Solver(args, data_loader)
    if args.mode == 'train':
        print('Start training')
        solver.train()
    elif args.mode == 'test':
        print('Start testing')
        solver.test()


#Locked random number seed
def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed=random.randint(0,10000)
#seed=1986
print(seed)
seed_torch(seed)
main(args)
