import numpy as np
import matplotlib.pyplot as plt
import torch

def See_loss(start,epochs_end):
    curve=np.load('./save/loss_{}_epochs.npy'.format(epochs_end))[start:epochs_end]
    print(curve[-1])
    x=range(start,epochs_end)
    plt.plot(x, curve, 'r', lw=1)
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train_loss"])

def printProgressBar(iteration,total,prefix='',suffix='',decimals=1,length=100,fill=' '):
    # referred from https://gist.github.com/snakers4/91fa21b9dda9d055a02ecd23f24fbc3d
    percent=('{0:.'+str(decimals)+'f}').format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '=' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()

def Flip_numpy(input):

    shape=int(input.shape[-1]/2)
    left_part=input[:,:shape]
    right_part=input[:,shape:]

    #centrosymmetry operation
    center=np.flip(left_part)

    #common mask
    common_mask=center*right_part
    common_mask[common_mask>0]=1
    common_mask[common_mask<=0]=0

    return center*common_mask,right_part*common_mask

def Divide(input_data,Keep_relative_intensity):
    B,H,W=input_data.shape
    assert H%16==0 and W%32==0
    max_all=input_data.max()
    lefts=np.zeros([B,H,int(W/2)],dtype=np.float32)
    rights=np.zeros([B,H,int(W/2)],dtype=np.float32)
    for idx in range (B):
        data=input_data[idx,:,:]
        if Keep_relative_intensity:
            data=data/max_all
        else:
            data=data/data.max()
        left,right=Flip_numpy(data)
        lefts[idx,:,:]=left
        rights[idx,:,:]=right

    return lefts,rights

def Norm(input_data,Keep_relative_intensity):
    B,H,W=input_data.shape
    assert H%16==0 and W%32==0
    max_all=input_data.max()
    outputs=np.zeros([B,H,W],dtype=np.float32)
    for idx in range (B):
        data=input_data[idx,:,:]
        if Keep_relative_intensity:
            data=data/max_all
        else:
            data=data/data.max()
        outputs[idx,:,:]=data
    return outputs
#See_loss(1000,17920)

