# Symmetrical Masked AutoEncoder (SMAE)
This is the code associated with the article "Denoising X-ray images by exploring their physical symmetry"

## About the Project
In this work, we combine the intrinsic __Symmetry__ of some physical experimental patterns and the __Masked AutoEncoder(MAE)__  pre-training algorithm in computer vision to create a __Symmetrical Masked AutoEncoder (SMAE)__ pre-training algorithm with __zero-shot noise reduction capability__ .The spatially-continuous small angle X-ray scattering (SAXS)/wide angle X-ray diffraction (WAXD) experiments and the temporally-continuous Atomic Pair Distribution Function (PDF) experiment are used as examples to demonstrate its excellent zero-shot noise reduction and physical information recovery ability for physical patterns with good symmetry and interference from experimental noise.In addition, we also explore its noise reduction effect after fine-tuning with paired  high-low SNR data, which still outperforms other methods, such as supervised learning algorithm; MAE pre-training algorithm  Zero-shot Noise2Noise algorithm and some traditional noise reduction methods based on physical a priori information.

## Prerequisites
Code is intended to work with: 
1. ```Python 3.8.x```
2. ```pytorch 1.9.1```
3. ```numpy 1.21.2```
4. ```skimage 1.7.1```
5. ```argparse```


__The network framework__  
Since the SMAE is a pre-training algorithm, any Unet CNN model or Transformer model can be used for its training. For convenience, I used the SEDCNN4-WAXD model from article __'A machine learning model for textured X-ray scattering and diffraction image denoising'__(DOI10.1038/s41524-023-01011-w) to implement the SMAE pre-training algorithm.

![image](https://github.com/zzZhou8/SEDCNN-for-SAXS-and-WAXD/blob/main/img/networks.png)
-----

## DATASET
### Data preparation
All of our input data is in __'.npy'__ format and the steps involved in SMAE data preprocessing are illustrated in Fig. 1 SMAE Pre-training. 

__Complete Step 0 and Step 1 yourself.__  
__Step 0:__ crops patterns into ROIs by the pattern center;   
__Step 1:__ sets the pixel values to zero in areas of no interest to users by masking; 

__The following process can be done by the Divide function in utils.__   
__Step 2:__ divides the masked pattern equally into left and right parts;   
__Step 3:__ takes either the left or the right part, then flips the pattern horizontally then vertically;   
__Step 4:__ combines the masks on the right part and on the flipped pattern as a common mask, since the masked areas on the right part and on the flipped pattern are not the same;   
__Step 5:__ applies the common mask respectively on the right part generated from Step 2 and on the flipped pattern generated from Step 3, producing masked right and masked left patterns as input and target of the Encoder-Decoder.

![image](https://github.com/zzZhou8/SMAE/blob/master/Fig_in_paper_and_code/Preprocess%20of%20SMAE%20algorithem%20and%20zero-shot%20denoising%20process.png)
## Training process
All of our input data is in __'.npy'__ format  

In the pre-training process we default to using the data augmentation method for training and testing by setting the parameters __train_data_aug__ and __test_data_aug__ to __True__, although this brings 4 times the computational cost compared to the training and testing process without data augmentation, but the effect is also significant.
### Base model pre-training process  
Before the pre-training starts, you can carry out the operations of Step 0 and Step 1 in Data preparation on your collected data according to your own needs, then save the data as a 3D npy array and name it as 'base_key_word.npy', and put it into the path of ' npy_img/base_model_data' path, our program will automatically complete the subsequent operations for you, where you can choose whether to save the relative intensity relationship between the data by setting the parameter "Keep_relative_intensity True".
```
python main.py \
    --mode train \
    --base_model True \
    --base_key_word 'base_key_word' \
    --train_data_aug True \
    --Keep_relative_intensity False \
```
### Fine-tuning using other similarity LSNR data
If a large number of similar samples are tested under the same experimental conditions, there is no need to pre-train SMAE on each sample, and it is sufficient to train the base model on one batch of data and then fine-tune the base model with other batches of LSNR data, which is done by naming the data from the other batches after they have been processed according to the requirements in the Data preparation as 'finetune_key_word.npy' and save it under the path 'npy_img\fintune_model_data', and select the parameters saved from the nth round of training of the base model to start the fine-tuning. If the result is not good, you can also follow the steps of the base model pre-training process to start training from the beginning.
```
python main.py \
    --mode train \
    --base_model False \
    --finetune_key_word 'finetune_key_word' \
    --base_test_epochs 260 \
    --train_data_aug True \
    --Keep_relative_intensity False \
```
### Fine-tuning using paired LSNR-HSNR data
If you have collected high SNR data paired with the low SNR data during the experiment, after completing the training of the base model using the low SNR data, process the low SNR data according to the requirements in the Data preparation, name the low SNR data and the high SNR data as 'finetune_key_word_LSNR_sup.npy' and 'finetune_key_word_HSNR_sup.npy' and save them in the path 'npy_img\fintune_model_data'.
```
python main.py \
    --mode train \
    --base_model False \
    --finetune_key_word 'finetune_key_word' \
    --exist_target_data True \
    --train_data_aug False \
    --base_test_epochs 260 \
    --Keep_relative_intensity False \
```

## Testing process
All of our output data is in __'.tif'__ format.
During the test you need to keep the setting of 'train_data_aug' consistent with the pre-training or fine-tuning phase, if the parameter 'train_data_aug' is True, then the parameter 'test_data_aug' can be set to either True or False, but if the parameter 'train_data_aug' is False, then the parameter 'test_data_aug' can only be set to False.

### Test base pre-trained model  
For this simplest application, where the base model is pre-trained and then tested. You can simply set the parameters as follows:  
For the parameter __base_test_epochs__, it represents the number of epochs you want to test the saved model, and we recommend this parameter to be greater than 100.   
For the parameter __exist_target_data__, if you have collected the matching high SNR data for evaluating the effectiveness of the method, if you set this parameter to True, the program will calculate the RMSE, PSNR and SSIM. of the original data and the optimised data compared with the high SNR data, and save it as a png image in the path 'save/fig'.
```
python main.py \
    --mode test \
    --base_model True \
    --base_key_word 'base_key_word' \
    --exist_target_data False \
    --train_data_aug True \
    --test_data_aug True \
    --base_test_epochs 260 \
```
### Testing model fine-tuned using other similarity LSNR data or paired LSNR-HSNR data
In this application, we will test pre-trained models that have been fine-tuned using similar LSNR data. You can simply set the parameters as follows, where __finetune_test_epochs__ indicates the number of epochs for which you want to test the saved fine-tuned model, which we recommend is less than or equal to 60  
```
python main.py \
    --mode test \
    --base_model False \
    --finetune_key_word 'finetune_key_word' \
    --exist_target_data False \
    --train_data_aug True \
    --test_data_aug True \
    --finetune_test_epochs 60 \
```
<<<<<<< HEAD
Finally we have provided a mind map for you to understand and use this code.
![image](https://github.com/zzZhou8/SMAE/blob/master/Fig_in_paper_and_code/Mind%20map%20of%20how%20to%20use%20SMAE%20code.jpg)

=======
We are providing a mind map for your understanding.
![image](https://github.com/zzZhou8/SMAE/blob/master/Fig_in_paper_and_code/Mind%20map%20of%20how%20to%20use%20SMAE%20code.jpg)
>>>>>>> ad0a4950a9604fd20ae2cc71f2f8a443ef412e4a
## Demonstrate the SMAE algorithm's zero-shot noise reduction effect and physical information recovery capability

Noise reduction results on the SAXS mapping from mouse radius (Section ‘Large amount of highly symmetrical low SNR data’), and on the WAXD mapping from cross section of human femur head (Section ‘Highly symmetrical data corrupted by external noises’) using the model trained by the SMAE algorithm and tested in zero-shot mode are shown.  
![image](https://github.com/zzZhou8/SMAE/blob/master/Fig_in_paper_and_code/Mouse%20radius%20SAXS%20and%20femoral%20head%20WAXD.png)

-----
Effectiveness of each denoising algorithm on physical information recovery of bamboo WAXD data in section “Small amount of highly symmetrical low SNR data” . A: WAXD pattern with 1-second exposure time; Symmetrical denoising result without data augmentation; Symmetrical denoising result with data augmentation; Denoising using supervised learning algorithm; Denoising using zero-shot Noise2Noise algorithm; WAXD pattern with 10-second exposure time. B: Azimuthal integration in the orange zone and gaussian fitting is performed on the green region after background subtraction.   
![image](https://github.com/zzZhou8/SMAE/blob/master/Fig_in_paper_and_code/Bamboo%20WAXD.png)

-----
(A) Illustration of anisotropic structure characterization on the MG ribbon under force loading at 250C. Differential diffraction pattern between 33N and 0N is shown. Fig. 5 (A(SMAE opt)) is plotted from the raw pattern after applying SMAE zero-shot noise reduction, and Fig. 5 (A(raw)) is plotted from the raw pattern. (B) We perform the Fourier transform of the full scattering pattern in reciprocal space to obtain the real-space G(r) plot. In order to study the effect of external loading on the local structure, the 0° sector parallel to the loading force is chosen in this study, and the S(Q) plot is obtained by integrating the red region of the 0° sector in Fig. 5(A).  In the S(Q) and G(r) plots of MG, RT-raw is achieved from MG bulk collected with 120s expose time at room temperature, 250C-raw is achieved from MG ribbon at 250 degrees Celsius, and 250C-SMAE is the 250C-raw data from MG ribbon after being optimized by the zero-shot noise reduction using the SMAE pre-trained model. Since the sample composition is the same while the temperature is different, the difference between the RT-raw data and the 250C-raw data at 0N loading in the G(r) plot only happens on the peak width. Therefore, the RT-raw data can be regard as the reference. (C) Time dependence of G(r) for MG ribbon under loading process. (D) The change of force, strain, and centroid of 3rd shell under loading process.  
![image](https://github.com/zzZhou8/SMAE/blob/master/Fig_in_paper_and_code/PDF.png)

<<<<<<< HEAD
-----
=======
>>>>>>> ad0a4950a9604fd20ae2cc71f2f8a443ef412e4a
Finally, we used the femoral head SAXS data 'hb08' and the SMAE algorithm to pre-train the model, and used the above model on three other femoral head SAXS datasets with the same experimental conditions but different samples to perform the zero-shot noise reduction test and the test with the low SNR data fine-tuning, respectively, and there is not a big difference in the results, which shows that the pre-trained model of the SMAE algorithm is robust to a certain extent.
![image](https://github.com/zzZhou8/SMAE/blob/master/Fig_in_paper_and_code/Finetuning%20using%20similar%20data%20orientation.png)
![image](https://github.com/zzZhou8/SMAE/blob/master/Fig_in_paper_and_code/Finetuning%20using%20similar%20data%20D-period.png)
![image](https://github.com/zzZhou8/SMAE/blob/master/Fig_in_paper_and_code/Finetuning%20using%20similar%20data%20Anisotropy.png)
