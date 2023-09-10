# Learning by Aligning Videos in Time (CVPR 2021)

## Overview
This repository contains the official implementation of our CVPR 2021 paper (https://openaccess.thecvf.com/content/CVPR2021/papers/Haresh_Learning_by_Aligning_Videos_in_Time_CVPR_2021_paper.pdf).

If you use the code, please cite our paper:
```
@inproceedings{haresh2021learning,
  title={Learning by aligning videos in time},
  author={Haresh, Sanjay and Kumar, Sateesh and Coskun, Huseyin and Syed, Shahram N and Konin, Andrey and Zia, Zeeshan and Tran, Quoc-Huy},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5548--5558},
  year={2021}
}
```

For our recent works, please check out our research page (https://retrocausal.ai/research/).


## Installation
Create an environment and install required packages
```
conda env create --name LAV --file=lav_env.yml
conda activate LAV
```

If you face any pytorch related issues during training, uninstall the pytorch first
```
pip3 uninstall torch torchvision torchaudio
```

Go to https://pytorch.org/get-started/locally/ and install the suitable pytorch as per you machine requirements.
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


## Video-to-Frame Conversion
```
python video_to_frames.py videos/
```


## Training/Testing Splits
Split your data into train and test and your directory should look like this
```
$YOUR_PATH_TO_DATASET
    ├─train
        ├──vid1/
        |   ├──000001.jpg
        |   ├──000002.jpg
        |   ├──...
    ├──val
        ├──vid2/
        |   ├──000001.jpg
        |   ├──000002.jpg
        |   ├──...
        ├──...
```


## Training
```
python train.py --description "LAV" --data_path Data
```


## Testing
```
python evaluations.py --model_path path/to/model --dest path/to/log/dest --device 0
```

The expected structure of evaluation is like this:
```
├──<PATH>
    ├──test
        ├──vid2
        |   ├──000001.jpg
        |   ├──000002.jpg
        |   ├──...
```
