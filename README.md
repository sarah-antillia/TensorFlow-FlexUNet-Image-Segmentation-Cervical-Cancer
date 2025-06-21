<h2>TensorFlow-FlexUNet-Image-Segmentation-Cervical-Cancer (2025/06/21)</h2>

This is the first experiment of Image Segmentation for Cervical-Cancer 
 based on our TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) 
and, 512x512 pixels <a href="https://drive.google.com/file/d/1ZaF9fd4MdaJitJFLXi-F6JtYXy9B2HAT/view?usp=sharing">
Augmented-Cervical-Cancer-PNG-ImageMaskDataset.zip</a>, 
which was derived by us from 

<a href="https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed">
<b>Cervical Cancer largest dataset (SipakMed)</b>
</a>
<br><br>
On the derivation of a Cervical-Cancer-ImageMaskDataset, please see also our repository 
<a href="https://github.com/sarah-antillia/ImageMask-Dataset-Cervical-Cancer">ImageMask-Dataset-Cervical-Cancer</a>.
<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the PNG dataset appear similar to the ground truth masks, 
but lack precision in some areas. To improve segmentation accuracy, we could consider using a different 
segmentation model better suited for this task, or explore online data augmentation strategies.<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/images/A-Koilocytotic_214.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/masks/A-Koilocytotic_214.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test_output/A-Koilocytotic_214.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/images/B-Metaplastic_245.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/masks/B-Metaplastic_245.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test_output/B-Metaplastic_245.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/images/A-Parabasal_097.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/masks/A-Parabasal_097.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test_output/A-Parabasal_097.png" width="320" height="auto"></td>
</tr>
</table>

<hr>

<br>

<h3>1. Dataset Citation</h3>

The original dataset used here has been taken from the following kaggle website:<br>
<a href="https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed">
<b>Cervical Cancer largest dataset (SipakMed)</b>
</a>

<h3>1. Dataset Citation</h3>

<b>About Dataset</b><br>

<b>Context</b><br>
Cervical cancer is the fourth most common cancer among women in the world, estimated more than 0.53 million<br> 
women are diagnosed in every year but more than 0.28 million women’s lives are taken by cervical cancer <br>
in every years . Detection of the cervical cancer cell has played a very important role in clinical practice.<br>
<br>
<b>Content</b><br>
The SIPaKMeD Database consists of 4049 images of isolated cells that have been manually cropped from 966 cluster<br>
 cell images of Pap smear slides. These images were acquired through a CCD camera adapted to an optical microscope.<br> 
 The cell images are divided into five categories containing normal, abnormal and benign cells.<br>
<br>
<b>Acknowledgements</b><br>
IEEE International Conference on Image Processing (ICIP) 2018, Athens, Greece, 7-10 October 2018.<br>

<b>Inspiration</b><br>
CERVICAL Cancer is an increasing health problem and an important cause of mortality in women worldwide. <br>
Cervical cancer is a cancer is grow in the tissue of the cervix . It is due to the abnormal growth of cell that <br>
are spread to the other part of the body.<br>
Automatic detection technique are used for cervical abnormality to detect Precancerous cell or cancerous cell<br> 
than no pathologist are required for manually detection process.<br>
<br>

<h3>
<a id="2">
2 Cervical-Cancer ImageMask Dataset
</a>
</h3>
 If you would like to train this Cervical-Cancer Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1ZaF9fd4MdaJitJFLXi-F6JtYXy9B2HAT/view?usp=sharing">
Augmented-Cervical-Cancer-PNG-ImageMaskDataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
                    (The Numbef of files)
./dataset
/Cervical-Cancer
├─test
│  ├─Koilocytotic
│  │  ├─images     ( 25)
│  │  └─masks      ( 25)
│  ├─Metaplastic
│  │  ├─images     ( 28)
│  │  └─masks      ( 28)
│  └─Parabasal
│      ├─images     ( 12)
│      └─masks      ( 12)
├─train
│  ├─Koilocytotic   
│  │  ├─images     (664)
│  │  └─masks      (664)
│  ├─Metaplastic
│  │  ├─images     (756)
│  │  └─masks      (756)
│  └─Parabasal
│      ├─images     (300)
│      └─masks      (300)
└─valid
    ├─Koilocytotic   
    │  ├─images     (188)
    │  └─masks      (188)
    ├─Metaplastic 
    │  ├─images     (216)
    │  └─masks      (216)
    └─Parabasal
        ├─images     ( 84)
        └─masks      ( 84)
</pre>
<br>
On the derivation of this 512x512 pixels PNG dataset, we used Python script 
<a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a><br>
<br>
For simplicity and demonstration purpose, in this experiment, we excluded the following two categories from the orignal dataset.<br>
<pre>
 ├─Dyskeratotic
 ├─Superficial-Intermediate
</pre>
<br>
<br>
<b>Koilocytotic Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/asset/train_images_sample_k.png" width="1024" height="auto">
<br>
<b>Koilocytotic Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/asset/train_masks_sample_k.png" width="1024" height="auto">
<br>
<br>
<b>Metaplastic Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/asset/train_images_sample_m.png" width="1024" height="auto">
<br>
<b>Metaplastic Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/asset/train_masks_sample_m.png" width="1024" height="auto">
<br>
<br>
<b>Parabasal Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/asset/train_images_sample_p.png" width="1024" height="auto">
<br>
<b>Parabasal Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/asset/train_masks_sample_p.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained Cervical-CancerTensorflowUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Cervical-Cancer/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Cervical-Cancerand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowFlexUNet.py">TensorflowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
num_classes    = 4
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.You may train this model by setting this generator parameter to True. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Dataset path</b><br>
Used wildcards in the data paths to include all images and masks for each category.
<pre>
[train]
mages_dir  = "../../../dataset/Cervical-Cancer/train/*/images/"
masks_dir  = "../../../dataset/Cervical-Cancer/train/*/masks/"
[valid]
images_dir = "../../../dataset/Cervical-Cancer/valid/*/images/"
masks_dir  = "../../../dataset/Cervical-Cancer/valid/*/masks/"
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>Mask RGB_map</b><br>
[mask]
<pre>
mask_datatype    = "categorized"
mask_file_format = ".png"
; "Koilocytotic": (0, 0, 255)
; "Metaplastic":  (0, 255, 0)
; "Parabasal"  :  (0, 128, 255)
; Background:black, 
rgb_map = {(0,0,0):0, (0, 0, 255):1, (0,255,0):2, (0, 128, 255):3}
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
As shown below, early in the model training, the predicted masks from our UNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br> 
<br>
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at middlepoint (epoch 18,19,20)</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 38,39,40)</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 40 by EarlyStopping callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/asset/train_console_output_at_epoch_40.png" width="820" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Cervical-Cancer/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Cervical-Cancer/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Cervical-Cancer</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Cervical-Cancer.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/asset/evaluate_console_output_at_epoch_40.png" width="820" height="auto">
<br><br>Image-Segmentation-Cervical-Cancer

<a href="./projects/TensorFlowFlexUNet/Cervical-Cancer/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Cervical-Cancer/test was very low, and dice_coef very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0883
dice_coef_multiclass,0.9715
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Cervical-Cancer</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Cervical-Cancer.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>b
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/images/A-Koilocytotic_214.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/masks/A-Koilocytotic_214.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test_output/A-Koilocytotic_214.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/images/A-Metaplastic_244.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/masks/A-Metaplastic_244.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test_output/A-Metaplastic_244.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/images/A-Parabasal_097.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/masks/A-Parabasal_097.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test_output/A-Parabasal_097.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/images/B-Koilocytotic_215.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/masks/B-Koilocytotic_215.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test_output/B-Koilocytotic_215.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/images/B-Metaplastic_245.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/masks/B-Metaplastic_245.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test_output/B-Metaplastic_245.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/images/B-Parabasal_098.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test/masks/B-Parabasal_098.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Cervical-Cancer/mini_test_output/B-Parabasal_098.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>

<h3>
References
</h3>

<b>1. Tensorflow-Tiled-Image-Segmentation-Augmented-Cervical-Cancer</b><br>
Toshiyuki Arai @antilli.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-Cervical-Cancer">
https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-Cervical-Cancer</a>
<br>
<br>
<b>2. Tensorflow-Image-Segmentation-Cervical-Cancer</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Cervical-Cancer">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Cervical-Cancer
</a>
<br>


