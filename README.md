# Image captioning model implemetned using Keras

---

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
└── images
```

## Usage
```bash
# for training
python train_td.py
```

For caption generation demo, see `test_predict.ipynb`.

## Reference

- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)

