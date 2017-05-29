# coding: utf-8

# import modules
import os
from functools import partial
import numpy as np
import tensorflow as tf
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from maeshori.nlp_utils import create_word_dict
from maeshori.caps_utils import CocoGenerator
from maeshori.callbacks import IftttMakerWebHook
from ShowAndTell import ShowAndTell

# configs
embedding_dim = 512
lstm_units = 512
max_sentence_length = 64

print("### parameters")
print("embedding_dim: %s"%embedding_dim)
print("lstm_units: %s"%lstm_units)
print("max_sentence_length: %s"%max_sentence_length)


# configuration
img_channels = 3
img_rows = 224
img_cols = 224
num_classes = 1000


# create resnet model instance
print("Loading image model")
#with tf.device("/gpu:1"):
img_model = ResNet50(weights='imagenet',
                 input_shape=(img_rows, img_cols, img_channels),
                 include_top=False, # without softmax layer (set True for training)
                 classes=num_classes)
img_model.trainable = False

img_feature_dim = img_model.output_shape[-1]

# caption preprocessor
def caps_preprocess(caption):
    import re
    return re.sub(r'\n|\.', '', caption.strip().lower())

# feature extractor
def deep_cnn_feature(img_data): # image (3-dim array) -> (feature_dim,)
    img_feature = img_model.predict(preprocess_input(img_data.astype(np.float32)))
    return img_feature.reshape((-1,))


# load MSCOCO
print("Loading MSCOCO")
# training data
coco_train = CocoGenerator('./COCO/', 'train2014',
                           word_dict_creator=partial(create_word_dict, idx_start_from=1),
                           caps_process=caps_preprocess, raw_img=False,
                           on_memory=True,
                           feature_extractor=deep_cnn_feature,
                           img_size=(img_rows, img_cols))
# validation data
coco_val = CocoGenerator('./COCO/', 'val2014',
                         word_dict=coco_train.word_dict,
                         vocab_size=coco_train.vocab_size,
                         caps_process=caps_preprocess, raw_img=False,
                         on_memory=True,
                         feature_extractor=deep_cnn_feature,
                         img_size=(img_rows, img_cols))


# load model
print("Preparing image captioning model")
#with tf.device("/gpu:0"):
im2txt_model = ShowAndTell(coco_train.vocab_size, img_feature_dim=img_feature_dim, max_sentence_length=max_sentence_length)
im2txt_model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.01), metrics=['acc'])

# callbacks
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=5)
checkpoint = ModelCheckpoint(filepath="./results/model_weight/weights_{epoch:02d}-{val_loss:.2f}_.hdf5",
                             save_best_only=True)
csv_logger = CSVLogger('./results/logs/show_and_tell.csv')
ifttt_url = 'https://maker.ifttt.com/trigger/{event}/with/key/' + os.environ['IFTTT_SECRET']
ifttt_notify = IftttMakerWebHook(ifttt_url)

# fit
print("Start Training")
im2txt_model.fit_generator(coco_train.generator(img_size=(img_rows, img_cols),
                                                feature_extractor=deep_cnn_feature,
                                                maxlen=max_sentence_length, padding='post'),
                           steps_per_epoch=coco_train.num_captions,
                           epochs=100,
                           callbacks=[lr_reducer, early_stopper, csv_logger, checkpoint, ifttt_notify],
                           validation_data=coco_val.generator(img_size=(img_rows, img_cols),
                                                              feature_extractor=deep_cnn_feature,
                                                              maxlen=max_sentence_length, padding='post'),
                           validation_steps=coco_val.num_captions,
                           max_q_size=1000,
                           verbose=0) # supress progress bar
