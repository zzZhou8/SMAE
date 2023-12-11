#%%
import numpy as np
from skimage import io
import os

key_word='hb_08'
root_path='save\Revised_SAXS_tif_base\{}\with_train_data_aug\with_test_data_aug'.format(key_word)
data_num=os.listdir(root_path)
data=np.zeros([len(data_num),256,256],dtype=np.float64)

for idx in range(len(data_num)):
    print(idx)
    tif_path=os.path.join(root_path,'Revised_SAXS_{}.tif'.format(idx))
    tif_data=io.imread(tif_path)
    tif_data=np.array(tif_data)
    data[idx,:,:]=tif_data

opt_data_save_path_root='save\opt_data'
if not os.path.exists(opt_data_save_path_root):
    os.makedirs(opt_data_save_path_root)
    print('Create path : {}'.format(opt_data_save_path_root))

opt_data_save_path=os.path.join(opt_data_save_path_root,'{}_opt.npy'.format(key_word))
np.save(opt_data_save_path,data)

# %%
