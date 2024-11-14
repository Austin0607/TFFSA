# Top-Level Feature Fusion with Self-Attention for Vehicle Component Detection（TFFSA）
# The code will be made public after the article is accepted

## Abstract
To advance the automation of vehicle damage assessment and address issues of missed and erroneous detections in
vehicle component detection and segmentation caused by insufficient global information extraction, we propose
the universal Top-Level Feature Fusion with Self-Attention (TFFSA) architecture for all backbones with four
feature extraction stages, termed Quad Stage Net (QSNet). This architecture includes the Top-Level Feature
Fusion (TLFF) and Top-Level Self-Attention (TLSA) modules. TLFF enhances global perception by applying
Position Average Pooling and self-attention to the feature map from Stage 3, merging it with the output
from Stage 4. The TLSA module further strengthens feature dependencies by applying multi-head self-attention, 
incorporating a Top-Level Learnable Scaling Factor (TLLSF) to optimize interactions. Together, TLFF and TLSA 
enhance cross-scale information flow through lateral connections and a top-down fusion path, effectively 
capturing global and local feature dependencies. Guided by TLFF, TLSA achieves more effective feature 
relationship modeling. Extensive experiments on a dataset with 59 component categories demonstrate performance 
improvements from each proposed module, with a 5.2% gain in detection accuracy and a 5.8% increase in 
segmentation accuracy over the baseline model. Applying the TFFSA backbone to other QSNet structures and 
segmentation models also led to significant performance gains, showcasing strong generalization capability. 
Our model effectively integrates feature fusion and self-attention, reducing missed and erroneous detections 
in vehicle component recognition.

## Environment configuration:
* Python3.6/3.7/3.8
* Pytorch1.10 or above
* pycocotools(Linux:`pip install pycocotools`;  Windows: 'pip install pycocotools-windows' (no additional installation vs required))
* Ubuntu or Centos(Windows not recommended)
* It is best to use GPU training
* See 'requirements.txt' for detailed environment configuration


## File structure:
` ` `
├── backbone: Feature extraction network
├── network_files: TFFSA network
├── train_utils: Training validation related modules (including coco validation related)
└ ── my_dataset_coco.py: Custom dataset used to read COCO2017 data sets
└ ── train.py: single GPU/CPU training script
├── train_multi_GPU.py: For users with multiple Gpus
└ ── Predict.py: A simple prediction script that uses trained weights to predict
├── validation.py: Validate/test COCO metrics of data with trained weights and generate record_mAP.txt file
└── transforms.py: data preprocessing (randomly flipping images horizontally and bboxes, converting PIL images to Tensor)
```




