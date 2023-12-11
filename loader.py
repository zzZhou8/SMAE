import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from utils import Divide,Norm

#这里的测试方式还是比较属于自导自演的，因为是自己测试自己的未知，用以评估系统自身的降噪能力
class SAXS_dataset(Dataset):
    def __init__(self,mode,saved_path,train_data_aug=None,test_data_aug=None,base_model=None,exist_target_data=None,base_key_word='Nothing',finetune_key_word='Nothing',Keep_relative_intensity=False):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"#确定现在是在测试还是在训练
        
        self.mode=mode
        self.train_data_aug=train_data_aug
        self.test_data_aug=test_data_aug
        self.exist_target_data=exist_target_data

        if self.mode=='train':
            if base_model:
                print('Load the data from {}'.format(base_key_word))
                print('Using LSNR dataset pre-train the model')
                data_path=os.path.join(saved_path,'base_model_data')
                input_path=os.path.join(data_path,'{}.npy'.format(base_key_word))  
            else:
                data_path=os.path.join(saved_path,'fintune_model_data')
                if exist_target_data:
                    print('Load the data from {}'.format(finetune_key_word))
                    print('Using HSNR dataset finetuning the pre-trained model')
                    input_path=os.path.join(data_path,'{}_LSNR_sup.npy'.format(finetune_key_word))
                    target_path=os.path.join(data_path,'{}_HSNR_sup.npy'.format(finetune_key_word))
                else:
                    print('Load the data from {}'.format(finetune_key_word))
                    print('Using similar LSNR dataset finetuning the pre-trained model')
                    input_path=os.path.join(data_path,'{}.npy'.format(finetune_key_word))
        else:
            if base_model:
                print('Load the data from {}'.format(base_key_word))
                data_path=os.path.join(saved_path,'base_model_data')
                input_path=os.path.join(data_path,'{}.npy'.format(base_key_word))
                target_path=os.path.join(data_path,'{}_target.npy'.format(base_key_word))
            else:
                print('Load the data from {}'.format(finetune_key_word))
                data_path=os.path.join(saved_path,'fintune_model_data')
                input_path=os.path.join(data_path,'{}.npy'.format(finetune_key_word))
                target_path=os.path.join(data_path,'{}_target.npy'.format(finetune_key_word))
                

        if self.mode=='train':
            if exist_target_data:
                input_pics=np.load(input_path)
                target_pics=np.load(target_path)
                assert input_pics.shape==target_pics.shape

                input_left_pics,input_right_pics=Divide(input_pics,Keep_relative_intensity)
                target_left_pics,target_right_pics=Divide(target_pics,Keep_relative_intensity)

                self.left_pics=np.concatenate([input_left_pics,input_right_pics],axis=0)
                self.right_pics=np.concatenate([target_left_pics,target_right_pics],axis=0)
                if self.train_data_aug:
                    self.left_=augment_data(self.left_pics)
                    self.right_=augment_data(self.right_pics)
                    print('with data augment left input data shape is ',self.left_.shape)
                    print('with data augment right output data shape is ',self.right_.shape)
                else:
                    self.left_=self.left_pics
                    self.right_=self.right_pics
                    print('without data augment left input data shape is ',self.left_.shape)
                    print('without data augment right output data shape is ',self.right_.shape)        
            else:

                input_pics=np.load(input_path)
                self.left_pics,self.right_pics=Divide(input_pics,Keep_relative_intensity)
                print('input left data shape is ',self.left_pics.shape)
                print('input right data shape is ',self.right_pics.shape)

                if self.train_data_aug:
                    self.left_pics_flip=augment_data(self.left_pics)
                    self.right_pics_flip=augment_data(self.right_pics)
                    self.left_,self.right_=get_pair(self.left_pics_flip,self.right_pics_flip)
                    print('with data augment left input data shape is ',self.left_.shape)
                    print('with data augment right output data shape is ',self.right_.shape)
                else:
                    self.left_,self.right_=get_pair(self.left_pics,self.right_pics)
                    print('without data augment left input data shape is ',self.left_.shape)
                    print('without data augment right output data shape is ',self.right_.shape)
                    

        else: # self.mode =='test'
            
            if exist_target_data:
                input_pics=np.load(input_path)
                self.left_,self.right_=Divide(input_pics,Keep_relative_intensity)
                target_pic=np.load(target_path)
                self.target_pic=Norm(target_pic,Keep_relative_intensity)
                print('test input data shape is ',self.left_.shape)
                print('test target data shape is ',self.target_pic.shape)
            else:
                input_pics=np.load(input_path)
                self.left_,self.right_=Divide(input_pics,Keep_relative_intensity)
                print('test input data shape is ',self.left_.shape)
                print('There is no target data, so PSNR, SSIM, RMSE cannot be calculated at this time.')

    def __len__(self):

        lens = self.left_.shape[0]        
        return  lens

    def __getitem__(self, idx):

        if self.mode=='train':

            left_img=self.left_[idx]
            right_img=self.right_[idx]

            return(left_img,right_img)
        else: # mode =='test'
            
            left_img=self.left_[idx]
            right_img=self.right_[idx]
            
            if self.exist_target_data:
                target_img=self.target_pic[idx]
                return(left_img,right_img,target_img)
            else:
                return(left_img,right_img) 
        

def get_pair(input_img,target_img):
    #形成正反训练对
    assert input_img.shape == target_img.shape

    if len(input_img.shape)==3:
        patch_input_imgs=np.concatenate((input_img,target_img))
        patch_target_imgs=np.concatenate((target_img,input_img))
    else:
        patch_input_imgs=[input_img,target_img]
        patch_target_imgs=[target_img,input_img]

    return np.array(patch_input_imgs), np.array(patch_target_imgs)

#Data flipping operation
def Flip_Horizon_and_Vertical_numpy(input):
 
    input_raw=input
    horizon=np.flip(input,axis=1)
    vertical=np.flip(input,axis=0)
    center = np.flip(input)

    return input_raw,horizon,vertical,center

def augment_data(input_pics):
    output_pics=[]
    for idx in range(input_pics.shape[0]):
        input_pic=input_pics[idx,:,:]
        inputpic,horizon,vertical,center=Flip_Horizon_and_Vertical_numpy(input_pic)
        output_pics.append(inputpic)
        output_pics.append(horizon)
        output_pics.append(vertical)
        output_pics.append(center)

    return np.array(output_pics)

def get_loader(mode='train',base_model=None,exist_target_data=None,saved_path=None,batch_size=32,train_data_aug=None,test_data_aug=None,base_key_word='Nothing',finetune_key_word='Nothing',Keep_relative_intensity=False):
    dataset_=SAXS_dataset(mode,saved_path,train_data_aug,test_data_aug,base_model,exist_target_data,base_key_word,finetune_key_word,Keep_relative_intensity)
    dataloader=DataLoader(dataset=dataset_,batch_size=batch_size,shuffle=(True if mode=='train' else False))
    return dataloader


