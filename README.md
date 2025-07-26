# Exploring the Transfer Learning Capabilities of CLIP on Domain Generalization for Diabetic Retinopathy

**CoOpLVT: Context Optimization with Learnable Visual Tokens - Enhanced Edition**

![Method](images/architecture.png)

## Abstract

Diabetic Retinopathy (DR), a leading cause of vision impairment, requires early detection and treatment. Developing robust AI models for DR classification holds substantial potential, but a key challenge is ensuring their generalization in unfamiliar domains with varying data distributions. To address this, our paper investigates cross-domain generalization, also known as domain generalization (DG), within the context of DR classification. DG, a challenging problem in the medical domain, is complicated by the difficulty of gathering labeled data across different domains, such as patient demographics and disease stages. Some recent studies have shown the effectiveness of using CLIP to handle the DG problem in natural images. In this study, we investigate CLIP’s transfer learning capabilities and its potential for cross-domain generalization in diabetic retinopathy (DR) classification. We carry out comprehensive experiments to assess the efficacy and potential of CLIP in addressing DG for DR classification. Further, we introduce a multi-modal finetuning strategy named Context Optimization with Learnable Visual Tokens (CoOpLVT). The original code is publicly available at https://github.com/Sanoojan/CLIP-DRDG

##  Enhanced Features

This enhanced version includes significant improvements to address class imbalance and improve training robustness:

- ** Medical-Specific Augmentations**: Conservative augmentation strategies optimized for fundus image characteristics  
- ** CLIP Optimization**: Proper CLIP normalization constants for optimal medical domain adaptation
- ** Enhanced Dataset Loader**: Robust DR dataset class with automatic structure detection and error handling
- ** Improved Training Stability**: Class weighting, label smoothing, and advanced regularization techniques

## Install Dependencies

Run the following command to install the required conda environment and dependencies:
```bash
conda env create --file=environment.yml
```

## Download Dataset

Download the Dataset from [APTOS](https://www.kaggle.com/c/aptos2019-blindness-detection), [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection/data), [Messidor](https://www.adcis.net/en/third-party/messidor/), and [Messidor-2](https://www.adcis.net/en/third-party/messidor2/). 

###  Dataset Structure



```
├── DATASET_PATH
│   ├── DR
│   │   ├── aptos
│   │   │   ├── train
│   │   │   │   ├── 0/ (or class_0/)
│   │   │   │   ├── 1/ (or class_1/)
│   │   │   │   └── ...
│   │   │   ├── test
│   │   │   └── valid
│   │   ├── eyepacs
│   │   ├── messidor
│   │   └── messidor_2
```


## Download Pre-trained Weights

The pre-trained weights can be accessed [here](https://drive.google.com/drive/folders/1w9gG3clV3ZlmhIT88n0QFNM29_rOou8Y?usp=sharing).

## CoOpLVT Algorithm Implementation

We implement our proposed **CoOpLVT** algorithm as a class in `domainbed/algorithms.py`:

```python
class Clip_train_prompt_from_image_v2(Algorithm):
    """
    CoOpLVT: Context Optimization with Learnable Visual Tokens
    """
    
    def __init__(self, input_shape, num_classes, num_domains, hparams, weights_for_balance):
       ...
```

## How To Use

###  Enhanced Training Features

The enhanced version includes new hyperparameters for improved performance:

- `medical_aug`: Enable medical-specific augmentations (default: `true`)
- `focal_loss`: Use Focal Loss for class imbalance (default: `true`) 
- `class_balanced`: Apply class-balanced loss weighting (default: `true`)
- `label_smoothing`: Label smoothing factor (default: `0.1`)

### Training Scripts

There are two important scripts to run the training and evaluate the results: `run_train.sh` & `run_evaluation.sh`. By default, we run our experiments for 3 trials and then report the mean and standard deviation of the results. `run_train.sh` will perform a training process and `run_evaluation.sh` will give accuracy and f1-score from the experiment. In addition, we provide a sample output folder (`COOPLVT_TRAINING_LOGS`). In the actual output folder, best models are monitored and saved.

**Enhanced Training Command:**
```bash
python -m domainbed.scripts.train \
    --data_dir=DATASET_PATH \
    --output_dir=ENHANCED_TRAINING_LOGS \
    --algorithm=Clip_train_prompt_from_image_v2 \
    --dataset=DR \
    --test_envs=0 \
    --hparams='{"medical_aug": true, "focal_loss": true, "class_balanced": true, "lr": 0.00002}'
```

`run_train.sh`
``` bash
#!/bin/bash

nvidia-smi

for lr in  0.000005 
do
    for dataset in DR
    do
        for init in clip_full
        do
            for command in delete_incomplete launch
            do
                CUDA_VISIBLE_DEVICES=1,2,3,4,8,15 python -m domainbed.scripts.sweep $command\
                    --data_dir=DATASET_PATH \
                    --output_dir=COOPLVT_TRAINING_LOGS \
                    --command_launcher multi_gpu\
                    --algorithms Clip_train_prompt_from_image_v2 \
                    --single_test_envs \
                    --datasets ${dataset} \
                    --n_hparams 1  \
                    --n_trials 3 \
                    --hparams """{\"weight_init\":\"${init}\",\"backbone\":\"ClipBase\",\"lr\":${lr}}"""\
                    --skip_confirmation
            done > Outs/V23_CLIP_COOP_3_LAYERS_MLP.out
        done
    done
done

# CUDA_VISIBLE_DEVICES --> Denotes the GPUs indexes that we use to run the experiment.
# --data_dir           --> Dataset path.
# --output_dir         --> The path where the experiment outputs are saved in.
# --algorithms         --> The algorithm class that we want to use. See domainbed/algorithms.py to find algorithm variants. CoOpLVT is implemented as Clip_train_prompt_from_image_v2 class.
# --n_trials           --> Denotes how many trials that we want to run the experiment. By default, we set n_trials as 3 to alleviate randomness during training, allowing us to better interprete our experiments.
# Outs/V23_CLIP_COOP_3_LAYERS_MLP.out --> If we want to store terminal outputs.
```

`run_evaluation.sh`
``` bash
python -m domainbed.scripts.collect_results --input_dir COOPLVT_TRAINING_LOGS

# --input_dir --> The path where the experiment outputs are saved in.

# Sample output

# -------- Dataset: DR, model selection method: training-domain validation set
# Algorithm             aptos                 eyepacs               messidor              messidor_2            Avg                  
# Clip_train_prompt_fr  46.2 +/- 4.4          65.9 +/- 2.0          65.5 +/- 0.4          70.6 +/- 0.6          62.1                 

# -------- Averages, model selection method: training-domain validation set
```

###  Enhanced Tools and Utilities

The enhanced version includes several useful tools:


**. Confusion Matrix Plotting**
```bash
python plot_cm_after_training.py --results_path ENHANCED_TRAINING_LOGS
```
- Generates confusion matrices from trained models
- Focuses on out-of-domain evaluation (aptos environment)
- Handles nested dataset structures automatically


```


## Performance Improvements

The enhanced version addresses key challenges in diabetic retinopathy classification:


###  Medical Domain Optimization  
- **CLIP Normalization**: Fixed constants optimized for medical imaging
- **Conservative Augmentations**: Medical-safe transformations preserving diagnostic features
- **Fundus-Specific Processing**: Circular crop and vessel enhancement techniques


## Acknowledgment

The code is built on top of DomainBed: a PyTorch suite containing benchmark datasets and algorithms for domain generalization, as introduced in [In Search of Lost Domain Generalization](https://github.com/facebookresearch/DomainBed). CLIP and ViT codes are based on [CLIP](https://github.com/openai/CLIP) and [Timm](https://github.com/huggingface/pytorch-image-models/tree/main/timm) respectively. 

**Enhanced Version Contributions:**
- Medical augmentation strategies inspired by clinical fundus imaging best practices
- Focal Loss implementation adapted from medical imaging literature
- Class balancing techniques optimized for diabetic retinopathy severity distribution

We thank the authors for releasing their code publicly and the medical imaging community for their guidance on fundus image processing.

## Licence
This source code is released under the MIT license, included [here](./LICENSE)
