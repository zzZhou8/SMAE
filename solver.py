import os
import math
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from utils import printProgressBar
from networks import  SEDCNN4
from measure import compute_measure
from measure import compute_measure_simple
from skimage import io

import torchvision.transforms as transforms

class Solver(object):
    def __init__(self,args,data_loader):
        self.mode=args.mode
        self.Loop_test=args.Loop_test

        self.base_model=args.base_model
        self.exist_target_data=args.exist_target_data

        self.data_loader=data_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        
        #The parameters here determine what is relevant for basic model training
        self.base_key_word=args.base_key_word
        self.base_num_epochs = args.base_num_epochs
        self.base_decay_epochs = args.base_decay_epochs
        self.base_save_epochs = args.base_save_epochs
        self.base_test_epochs = args.base_test_epochs

        #The parameters here determine what is relevant for finetune model training
        self.finetune_key_word=args.finetune_key_word
        self.finetune_num_epochs = args.finetune_num_epochs
        self.finetune_decay_epochs = args.finetune_decay_epochs
        self.finetune_save_epochs = args.finetune_save_epochs
        self.finetune_test_epochs = args.finetune_test_epochs
        
        self.print_iters = args.print_iters
        self.train_data_aug=args.train_data_aug
        self.test_data_aug=args.test_data_aug
        self.multi_gpu = args.multi_gpu
        self.save_path = args.save_path
        self.result_fig = args.result_fig
        self.exist_target_data=args.exist_target_data

        self.SEDCNN4 = SEDCNN4()
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):#多gpu调用
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.SEDCNN4= nn.DataParallel(self.SEDCNN4)
        self.SEDCNN4.to(self.device)

        self.base_lr = args.base_lr
        self.finetune_lr = args.finetune_lr
        
        self.criterion = nn.MSELoss()
        if self.base_model:
            self.optimizer = optim.Adam(self.SEDCNN4.parameters(), self.base_lr)
        else:
            self.optimizer = optim.Adam(self.SEDCNN4.parameters(), self.finetune_lr)
    
    def save_model_loss(self, epochs_ , train_losses):

        if self.base_model:
            model_path_root_raw = os.path.join(self.save_path, 'Model_base')
            model_path_root=os.path.join(model_path_root_raw, self.base_key_word)
        else:
            model_path_root_raw = os.path.join(self.save_path, 'Model_finetune')
            model_path_root=os.path.join(model_path_root_raw, self.finetune_key_word)

        if self.train_data_aug:
            model_path_root = os.path.join(model_path_root, 'with_train_data_aug')
        else:
            model_path_root = os.path.join(model_path_root, 'without_train_data_aug')

        if not os.path.exists(model_path_root):
            os.makedirs(model_path_root)
            print('Create path : {}'.format(model_path_root))

        #保存模型
        f = os.path.join(model_path_root, 'SEDCNN4_{}epochs.ckpt'.format(epochs_))#按照轮次来存的
        torch.save(self.SEDCNN4.state_dict(), f)
        #保存loss函数
        np.save(os.path.join(model_path_root, 'loss_{}_epochs.npy'.format(epochs_)), train_losses)

    def load_model(self, epochs_):

        #If the model is loaded in the training state the only possibility is to fine-tune the training.
        if self.mode == 'train':
            if self.base_model:
                print('Basic model training is in progress, there should not be a model loading process, please check the programme.')
            else:
                model_path_root_raw = os.path.join(self.save_path, 'Model_base')
                model_path_root=os.path.join(model_path_root_raw, self.base_key_word)
        #Corresponding test situation
        else:
            if self.base_model:
                model_path_root_raw = os.path.join(self.save_path, 'Model_base')
                model_path_root=os.path.join(model_path_root_raw, self.base_key_word)
            else:
                model_path_root_raw = os.path.join(self.save_path, 'Model_finetune')
                model_path_root=os.path.join(model_path_root_raw, self.finetune_key_word)
            
        if self.train_data_aug:
            model_path_root = os.path.join(model_path_root, 'with_train_data_aug')
        else:
            model_path_root = os.path.join(model_path_root, 'without_train_data_aug')

        f = os.path.join(model_path_root, 'SEDCNN4_{}epochs.ckpt'.format(epochs_))

        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f).items():
                n = k[7:]
                state_d[n] = v
            self.SEDCNN4.load_state_dict(state_d)
        else:
            self.SEDCNN4.load_state_dict(torch.load(f))

    # Learning rate reduced to half of original
    def lr_decay(self):
           
        if self.base_model:
            lr = self.base_lr * 0.5
        else:
            lr = self.finetune_lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    #Determine the path to save the Revised tif
    def save_tif(self,pred,fig_name):

        if self.base_model:
            fig_path_root = os.path.join(self.save_path, 'Revised_tif_base')
            fig_path_root = os.path.join(fig_path_root, self.base_key_word)
        else:
            fig_path_root = os.path.join(self.save_path, 'Revised_tif_finetune')
            fig_path_root = os.path.join(fig_path_root, self.finetune_key_word)
        
        if self.train_data_aug:
            fig_path_root = os.path.join(fig_path_root, 'with_train_data_aug')
        else:
            fig_path_root = os.path.join(fig_path_root, 'without_train_data_aug')

        if self.test_data_aug:
            fig_path = os.path.join(fig_path_root, 'with_test_data_aug')
        else:
            fig_path = os.path.join(fig_path_root, 'without_test_data_aug')


        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

        pred=pred.numpy()

        io.imsave(os.path.join(fig_path, 'Revised_{}.tif'.format(fig_name)),np.float32(pred))
    
    def save_fig(self,x,y,pred,fig_name,original_result,pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))

        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Noise_pics', fontsize=30)#抬头的标题
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],original_result[1],original_result[2]), fontsize=20)#标一下图片的各项参数

        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Predict_pics', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)

        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('True_pics', fontsize=30)

        if self.base_model:
            fig_path_root = os.path.join(self.save_path, 'fig_base')
            fig_path_root = os.path.join(fig_path_root, self.base_key_word)
        else:
            fig_path_root = os.path.join(self.save_path, 'fig_finetune')
            fig_path_root = os.path.join(fig_path_root, self.finetune_key_word)
        
        
        if self.train_data_aug:
            fig_path_root = os.path.join(fig_path_root, 'with_train_data_aug')
        else:
            fig_path_root = os.path.join(fig_path_root, 'without_train_data_aug')

        if self.test_data_aug:
            fig_path = os.path.join(fig_path_root, 'with_test_data_aug')
        else:
            fig_path = os.path.join(fig_path_root, 'without_test_data_aug')

        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))
        

        f.savefig(os.path.join(fig_path, 'result_{}.png'.format(fig_name)))

        plt.close()
    
    #Model training
    def train(self):
        print('现在参与训练的数据共有',len(self.data_loader),'组')
        train_losses=[]
        total_iters=0
        total_epochs=0
        start_time=time.time()

        if self.base_model:
            print('现在从头开始您的基础模型训练')
        else:
            print('现在加载保存的第{}轮基础模型并开始您的微调模型训练'.format(self.base_test_epochs))
            self.load_model(self.base_test_epochs)
        
        if self.base_model:
            train_epochs=self.base_num_epochs
        else:
            train_epochs=self.finetune_num_epochs
        for epoch in range(1, train_epochs):

            total_epochs += 1
            self.SEDCNN4.train(True)
            
            for iter_, (x,y) in enumerate(self.data_loader):
                total_iters += 1

            # Since the input is a greyscale map, it is necessary to add a channel.
                x = x.unsqueeze(1).float().to(self.device)
                y = y.unsqueeze(1).float().to(self.device)

                pred = self.SEDCNN4(x)
                loss = self.criterion(pred, y) 
                self.SEDCNN4.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                # print
                if total_iters % self.print_iters == 0:

                    if self.base_model:
                        all_train_epochs=self.base_num_epochs
                    else:
                        all_train_epochs=self.finetune_num_epochs

                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        all_train_epochs, iter_+1, 
                                                                                                        len(self.data_loader), loss.item(), 
                                                                                                        time.time() - start_time))
                if self.base_model:
                    # learning rate decay
                    if total_iters % (self.base_decay_epochs*len(self.data_loader)) == 0:
                        self.lr_decay()

                    # save model
                    if total_iters % (self.base_save_epochs*len(self.data_loader)) == 0:
                        self.save_model_loss( total_epochs, np.array(train_losses))
                else:
                    # learning rate decay
                    if total_iters % (self.finetune_decay_epochs*len(self.data_loader)) == 0:
                        self.lr_decay()

                    # save model
                    if total_iters % (self.finetune_save_epochs*len(self.data_loader)) == 0:
                        self.save_model_loss( total_epochs, np.array(train_losses))
    
    #Model testing
    def test(self):

        if self.train_data_aug == False and self.test_data_aug == True:
            print('You should not use augmentation to test data on models that have not been trained with data augmentation')
            
        else:
            print('The training and testing status of your model is now: trained using data augmentation method {},tested using data augmentation method {}'.format(self.train_data_aug,self.test_data_aug))
            if self.Loop_test:
                pred_psnr_avg_list=[]
                pred_ssim_avg_list=[]
                pred_rmse_avg_list=[]

                loop_num=int(self.test_epochs/self.save_epochs)
                print('In total, we will test the {} group of models'.format(loop_num))
                for idx in range(1,loop_num+1):
                    
                    print('Group {} of models being tested'.format(idx))

                    del self.SEDCNN4

                    self.SEDCNN4 = SEDCNN4().to(self.device)
                    test_epoch=int(self.save_epochs*idx)
                    self.load_model(test_epoch)#重新保存的模型

                    # compute PSNR, SSIM, RMSE
                    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
                    
                    with torch.no_grad():
                        for i, (x_raw,y_raw,target) in enumerate(self.data_loader):
                            shape_ = target.shape[-1]

                            if self.test_data_aug:
                                x=Flip_Horizon_and_Vertical(x_raw,inverse=False)
                                y=Flip_Horizon_and_Vertical(y_raw,inverse=False)

                                x = x.unsqueeze(1).float().to(self.device)
                                y = y.unsqueeze(1).float().to(self.device)

                                pred_left = self.SEDCNN4(x)
                                pred_right = self.SEDCNN4(y)

                                x=Flip_Horizon_and_Vertical(pred_left,inverse=True)
                                y=Flip_Horizon_and_Vertical(pred_right,inverse=True)

                                x_out=torch.mean(y,dim=0,keepdim=True)
                                y_out=torch.mean(x,dim=0)
        
                            else:
                                x = x_raw.unsqueeze(1).float().to(self.device)
                                y = y_raw.unsqueeze(1).float().to(self.device)

                                pred_left = self.SEDCNN4(x)
                                pred_right = self.SEDCNN4(y)

                                x_out = pred_right.squeeze(0)
                                y_out = pred_left.squeeze()
                            
                            transform_center=transforms.Compose([transforms.RandomHorizontalFlip(p=1),transforms.RandomVerticalFlip(p=1)])
                            x_out=transform_center(x_out)
                            x_out=x_out.squeeze()
                            pred = torch.cat([x_out,y_out],dim=-1)

                            #pred = pred.unsqueeze(0)
                            #pred = transform_center(pred)
                            #pred = pred.squeeze()

                            #这个是目标图像
                            target_pic=target.squeeze()

                            # denormalize, truncate
                            pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))
                            target_pic = self.trunc(self.denormalize_(target_pic.view(shape_, shape_).cpu().detach()))

                            data_range = self.trunc_max - self.trunc_min
                            pred_result = compute_measure_simple(target_pic.to(torch.float32), pred.to(torch.float32), data_range)
                            pred_psnr_avg += pred_result[0]
                            pred_ssim_avg += pred_result[1]
                            pred_rmse_avg += pred_result[2]
                            
                            printProgressBar(i, len(self.data_loader),
                                            prefix="Compute measurements ..",
                                            suffix='Complete', length=25)#显示一下进度

                        #均值计算：
                        pred_psnr_avg=pred_psnr_avg/len(self.data_loader)
                        pred_ssim_avg=pred_ssim_avg/len(self.data_loader)
                        pred_rmse_avg=pred_rmse_avg/len(self.data_loader)
                        delat_mean_optimized_dot3=delat_mean_optimized_dot3/len(self.data_loader)
                        delat_Var_optimized_dot3=delat_Var_optimized_dot3/len(self.data_loader)
                        delta_intensity_optimized_dot3=delta_intensity_optimized_dot3/len(self.data_loader)
                        dot3_direct_optimized_ava=dot3_direct_optimized_ava/len(self.data_loader)

                        #均值在列表中进行保存：
                        pred_psnr_avg_list.append(pred_psnr_avg)
                        pred_ssim_avg_list.append(pred_ssim_avg)
                        pred_rmse_avg_list.append(pred_rmse_avg)

                        
                        print('\n')
                        #显示一下最后的训练效果呗
                        print('对于第{}保存的模型其预测结果为：'.format(test_epoch))
                        print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f} '.format(pred_psnr_avg, pred_ssim_avg, pred_rmse_avg))

                    #最终的数据保存
                    avg_list_root='avg_list'
                    if self.train_data_aug:
                        avg_list=os.path.join(avg_list_root,'with_train_data_aug')
                    else:
                        avg_list=os.path.join(avg_list_root,'without_train_data_aug')

                    if self.test_data_aug:
                        avg_list = os.path.join(avg_list, 'with_test_data_aug')
                    else:
                        avg_list = os.path.join(avg_list, 'without_test_data_aug')


                    if not os.path.exists(avg_list):#没有就产生一个
                        os.makedirs(avg_list)
                        print('Create path : {}'.format(avg_list))
                    
                    np.save(os.path.join(avg_list,'pred_psnr_avg_list.npy'),pred_psnr_avg_list)
                    np.save(os.path.join(avg_list,'pred_ssim_avg_list.npy'),pred_ssim_avg_list)
                    np.save(os.path.join(avg_list,'pred_rmse_avg_list.npy'),pred_rmse_avg_list)

            else:#直接测试
                del self.SEDCNN4#把原本的模型进行了释放
                #load
                self.SEDCNN4 = SEDCNN4().to(self.device)

                if self.base_model:
                    self.load_model(self.base_test_epochs)#加载测试轮次的基础模型
                else:
                    self.load_model(self.finetune_test_epochs)#加载测试轮次的微调模型
                
                # compute PSNR, SSIM, RMSE
                ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
                pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
                
                with torch.no_grad():
                    for i, data_all in enumerate(self.data_loader):
                        #If there is a target, load it, if not, don't.
                        if self.exist_target_data:
                            x_raw=data_all[0]
                            y_raw=data_all[1]
                            target=data_all[2]
                        else:
                            x_raw=data_all[0]
                            y_raw=data_all[1]

                        shape_1 = x_raw.shape[1]
                        shape_2 = int(x_raw.shape[2]*2)

                        if self.test_data_aug:
                            x=Flip_Horizon_and_Vertical(x_raw,inverse=False)
                            y=Flip_Horizon_and_Vertical(y_raw,inverse=False)

                            x = x.unsqueeze(1).float().to(self.device)
                            y = y.unsqueeze(1).float().to(self.device)

                            pred_left = self.SEDCNN4(x)
                            pred_right = self.SEDCNN4(y)

                            x=Flip_Horizon_and_Vertical(pred_left,inverse=True)
                            y=Flip_Horizon_and_Vertical(pred_right,inverse=True)

                            x_out=torch.mean(y,dim=0,keepdim=True)
                            y_out=torch.mean(x,dim=0)
    
                        else:
                            x = x_raw.unsqueeze(1).float().to(self.device)
                            y = y_raw.unsqueeze(1).float().to(self.device)

                            pred_left = self.SEDCNN4(x)
                            pred_right = self.SEDCNN4(y)

                            x_out = pred_right.squeeze(0)
                            y_out = pred_left.squeeze()
                        
                        transform_center=transforms.Compose([transforms.RandomHorizontalFlip(p=1),transforms.RandomVerticalFlip(p=1)])
                        x_out=transform_center(x_out)
                        x_out=x_out.squeeze()
                        pred = torch.cat([x_out,y_out],dim=-1)

                        x_input=transform_center(x_raw)
                        x_input=x_input.squeeze()
                        y_input=y_raw.squeeze()
                        input_pic = torch.cat([x_input,y_input],dim=-1)

                        # denormalize, truncate
                        input_pic= self.trunc(self.denormalize_(input_pic.view(shape_1, shape_2).cpu().detach()))
                        pred = self.trunc(self.denormalize_(pred.view(shape_1, shape_2).cpu().detach()))

                        if self.exist_target_data:
                            target_pic=target.squeeze()
                            target_pic = self.trunc(self.denormalize_(target_pic.view(shape_1, shape_2).cpu().detach()))

                            data_range = self.trunc_max - self.trunc_min

                            original_result, pred_result = compute_measure(input_pic.to(torch.float32), target_pic.to(torch.float32), pred.to(torch.float32), data_range)
                            ori_psnr_avg += original_result[0]
                            ori_ssim_avg += original_result[1]
                            ori_rmse_avg += original_result[2]
                            pred_psnr_avg += pred_result[0]
                            pred_ssim_avg += pred_result[1]
                            pred_rmse_avg += pred_result[2]

                            # save result figure
                            if self.result_fig:
                                self.save_fig(input_pic.to(torch.float32), target_pic.to(torch.float32), pred.to(torch.float32), i, original_result, pred_result)#属于作图给大家看一眼了
                        
                        self.save_tif(pred,i)#如果是对原图做测试请用这个
                        
                        printProgressBar(i, len(self.data_loader),
                                        prefix="Compute measurements ..",
                                        suffix='Complete', length=25)#显示一下进度
                        
                    if self.exist_target_data:
                        print('\n')
                        print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), 
                                                                                                        ori_ssim_avg/len(self.data_loader), 
                                                                                                        ori_rmse_avg/len(self.data_loader)))
                        print('\n')
                        print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), 
                                                                                                            pred_ssim_avg/len(self.data_loader), 
                                                                                                            pred_rmse_avg/len(self.data_loader)))
                    else:
                        print('\n')
                        print('No target image, so no statistics output.')


def Flip_Horizon_and_Vertical(input,inverse):
 
    #如果需将其转回
    if inverse:
        input_raw=input
    else:
        input_raw=input.repeat(4,1,1)
        input_raw=input_raw.unsqueeze(1)

    transform_horizon=transforms.RandomHorizontalFlip(p=1)
    transform_vertical=transforms.RandomVerticalFlip(p=1)
    transform_center=transforms.Compose([transforms.RandomHorizontalFlip(p=1),transforms.RandomVerticalFlip(p=1)])
    raw=input_raw[0,:,:,:]
    horizon=transform_horizon(input_raw[1,:,:,:])
    vertical=transform_vertical(input_raw[2,:,:,:])
    center = transform_center(input_raw[3,:,:,:])
    #全部转化为npy数组
    output_data=torch.cat([raw,horizon,vertical,center])
    
    return output_data

