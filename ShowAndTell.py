# coding: utf-8

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, LSTM, Embedding, Masking, Reshape
from keras.layers.merge import concatenate

def ShowAndTell(vocab_size,
                img_model=None,
                img_feature_dim=None,
                embedding_dim=300,
                lstm_units=200,
                max_sentence_length=30):
    """
    Args:
        vocab_size: vocaburary size
        img_model: image model (such as resnet, vgg, etc.)
        img_feature_dim: image feature dim (specify one of img_model or img_feature_dim)
        embedding_dim: dimension of embedding layer
        lstm_units: number of units in LSTM
        max_sentence_length: maximum length of sequence

    Return:
        image captioning model
    """
    # check args
    if img_model is None and img_feature_dim is None:
        raise ValueError("Specify one of img_model or img_feature_dim")
    if not isinstance(img_feature_dim, int) or img_feature_dim <= 0:
        raise ValueError("img_feature_dim must be positive integer")

    if img_feature_dim is None:
        #  image model
        img_input = Input(shape=img_model.input_shape[1:], name="img_input")
        img_output = img_model(img_input)
        img_output = Dense(embedding_dim, name="i_dense_1")(img_input)
    else:
        # image input layer
        img_input = Input(shape=(img_feature_dim,), name="img_input")
        img_output = Dense(embedding_dim, name="i_dense_1")(img_input)

    img_output = Reshape((-1, embedding_dim), name="i_reshape_2")(img_output)
    img_output = Masking(mask_value=0.0,
                         name="i_mask_3")(img_output) # error at concatenation without this layer

    # language model
    lang_input = Input(shape=(max_sentence_length, ), dtype="int32", name="lang_input")
    lang_output = Embedding(vocab_size + 1, embedding_dim, mask_zero=True,
                            input_length=max_sentence_length,
                            name="l_embed_1")(lang_input) # +1 for padding

    model_output = concatenate([img_output, lang_output], axis=1, name="concat_1")
    model_output = LSTM(lstm_units, recurrent_dropout=0.2,
                        dropout=0.2, name="c_lstm_2")(model_output)
    model_output = Dropout(0.2, name="c_drop_3")(model_output)
    model_output = Dense(vocab_size, activation="softmax", name="c_dense_4")(model_output)
    model = Model(inputs=[img_input, lang_input], outputs=[model_output])

    return model
