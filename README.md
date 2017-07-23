# Image captioning model implemetned using Keras


## Prerequisites

### Python Environment

- Anaconda 3
- Extra Python modules
  - Keras >= 2.0.4
  - TensorFlow >= 1.1.0
  - tqdm
  - pycoco (fixed for python3)

### Dataset

The script `train_td.py` uses [MSCOCO](http://mscoco.org) dataset placed at `./COCO` as follows.

```bash
./COCO
├── annotations
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   ├── image_info_test-dev2015.json
│   ├── image_info_test2014.json
│   ├── image_info_test2015.json
│   ├── instances_train2014.json
│   ├── instances_val2014.json
│   ├── person_keypoints_train2014.json
│   └── person_keypoints_val2014.json
└── images
    ├── test2014
    ├── test2015
    ├── train2014
    └── val2014
```

### Other tools

- git-lfs

## Usage

```bash
# for training
python train_td.py
```

For caption generation demo, see `test_predict.ipynb`.

## Reference

- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)

