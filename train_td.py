# coding: utf-8

# import modules
import os
from functools import partial
import numpy as np
import tensorflow as tf
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.models import load_model
#from keras.applications import ResNet50
#from keras.applications.resnet50 import preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from maeshori.nlp_utils import create_word_dict
from maeshori.caps_utils import CocoGenerator
from maeshori.gen_utils import stack_batch
from maeshori.callbacks import IftttMakerWebHook
from maeshori.models import make_parallel
from ShowAndTell import ShowAndTell

# configs
embedding_dim = 512
lstm_units = 512
num_stack_lstm = 1
max_sentence_length = 64
batch_size = 64

print("### parameters")
print("embedding_dim: %d"%embedding_dim)
print("lstm_units: %d"%lstm_units)
print("stack RNN: %d"%num_stack_lstm)
print("max_sentence_length: %d"%max_sentence_length)
print("batch size: %d"%batch_size)


# configuration
num_gpu = 1 # FIXME: error when num_gpu > 1
img_channels = 3
img_rows = 299
img_cols = 299
num_classes = 1000

# create resnet model instance
print("Loading image model")
img_model = InceptionV3(weights='imagenet',
                        input_shape=(img_rows, img_cols, img_channels),
                        include_top=False, # without softmax layer (set True for training)
                        pooling='avg',
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
                           img_size=(img_rows, img_cols),
                           cache='./COCO/cache/inceptionv3_train.pkl')
# validation data
coco_val = CocoGenerator('./COCO/', 'val2014',
                         word_dict=coco_train.word_dict,
                         vocab_size=coco_train.vocab_size,
                         caps_process=caps_preprocess, raw_img=False,
                         on_memory=True,
                         feature_extractor=deep_cnn_feature,
                         img_size=(img_rows, img_cols),
                         cache='./COCO/cache/inceptionv3_val.pkl')


# load model
print("Preparing image captioning model")
if num_gpu == 1:
    im2txt_model = ShowAndTell(coco_train.vocab_size, img_feature_dim=img_feature_dim,
                               units=lstm_units,
                               max_sentence_length=max_sentence_length,
                               time_distributed=True,
                               stack_lstm=num_stack_lstm)
    #im2txt_model = load_model('./results/model_weight/weights_adam64_inceptionv3_td_91-2.21_.hdf5')
else:
    with tf.device('/cpu:0'):
        im2txt_model = ShowAndTell(coco_train.vocab_size, img_feature_dim=img_feature_dim,
                                   units=lstm_units,
                                   max_sentence_length=max_sentence_length,
                                   time_distributed=True,
                                   stack_lstm=num_stack_lstm)
        #im2txt_model = load_model('./results/model_weight/weights_rmsprop64_28-2.93_.hdf5', compile=False)
    im2txt_model = make_parallel(im2txt_model, num_gpu)
im2txt_model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(), metrics=['acc'])

# callbacks
lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=5, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=100)
checkpoint = ModelCheckpoint(
    filepath="./results/model_weight/weights_em%d_lu%d_bs%d_sl%d_{epoch:02d}-{val_loss:.2f}_.hdf5"%(embedding_dim, lstm_units, batch_size, num_stack_lstm),
    save_best_only=True)
csv_logger = CSVLogger('./results/logs/log_em%d_lu%d_bs%d_sl%d_.csv'%(embedding_dim, lstm_units, batch_size, num_stack_lstm))
ifttt_url = 'https://maker.ifttt.com/trigger/keras_callback/with/key/' + os.environ['IFTTT_SECRET']
ifttt_notify = IftttMakerWebHook(ifttt_url, job_name='caption_lstm_td')

# fit
print("Start Training")
coco_train_gen = coco_train.generator(img_size=(img_rows, img_cols),
                                      feature_extractor=deep_cnn_feature,
                                      format_split=False, onehot_y=False,
                                      maxlen=max_sentence_length, padding='post')
coco_val_gen = coco_val.generator(img_size=(img_rows, img_cols),
                                  feature_extractor=deep_cnn_feature,
                                  format_split=False, onehot_y=False,
                                  maxlen=max_sentence_length, padding='post')

coco_train_gen = stack_batch(coco_train_gen, batch_size)
coco_val_gen = stack_batch(coco_val_gen, batch_size)

im2txt_model.fit_generator(coco_train_gen,
                           steps_per_epoch=coco_train.num_captions // batch_size,
                           epochs=100,
                           callbacks=[lr_reducer, early_stopper, csv_logger,
                                      checkpoint, ifttt_notify],
                           #callbacks=[csv_logger, checkpoint, ifttt_notify],
                           validation_data=coco_val_gen,
                           validation_steps=coco_val.num_captions // batch_size,
                           max_q_size=100)
