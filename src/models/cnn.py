import keras.backend as K
from keras.layers import *
from keras import Model
from keras.regularizers import l2


class Attention(Layer):
    # https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv1D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _residual_block(block_function, filters, repetitions, is_first_layer=False, dropout=0):
    """Builds a residual block with repeating bottleneck blocks.
    """

    def f(input):
        for i in range(repetitions):
            init_strides = 1
            if i == 0 and not is_first_layer:
                init_strides = 2
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0),
                                   dropout=dropout)(input)
        return input

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    dropout = conv_params.setdefault("dropout", 0)
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        if dropout:
            activation = Dropout(dropout)(activation)
        return Conv1D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def basic_block_single_relu(filters, init_strides=1, is_first_block_of_first_layer=False, dropout=0):
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv1D(filters=filters, kernel_size=3,
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_conv(filters=filters, kernel_size=3,
                             strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=3, dropout=dropout)(conv1)
        return _shortcut(input, residual)

    return f


def basic_block(filters, init_strides=1, is_first_block_of_first_layer=False, dropout=0):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """

    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv1D(filters=filters, kernel_size=3,
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=3,
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=3)(conv1)
        return _shortcut(input, residual)

    return f


def _bn_conv(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        norm = BatchNormalization(axis=2)(input)
        conv = Conv1D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(norm)
        return _bn_relu(conv)

    return f


def _bn_relu(input_):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization()(input_)
    return Activation("relu")(norm)


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    equal_channels = input_shape[2] == residual_shape[2]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or not equal_channels:
        shortcut = Conv1D(filters=residual_shape[2],
                          kernel_size=1,
                          strides=stride_width,
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return Add()([shortcut, residual])


def get_model(input_size=(512, 3), k=1, repetitions=[2, 2, 2, 2], dropout=0):
    inp = Input(input_size)

    filters = k * 16
    conv1 = _conv_bn_relu(filters=filters, kernel_size=40,
                          strides=10)(inp)
    pool1 = MaxPooling1D(pool_size=10)(conv1)

    conv2 = _conv_bn_relu(filters=filters, kernel_size=3,
                          strides=1)(pool1)
    pool2 = MaxPooling1D(pool_size=4)(conv2)

    block = pool2
    for i, r in enumerate(repetitions):
        filters = k * 16 * (i + 1)
        block = _residual_block(basic_block_single_relu, filters=filters, repetitions=r, is_first_layer=(i == 0),
                                dropout=dropout)(block)
        if K.int_shape(block)[1] >= 4:
            block = MaxPooling1D(pool_size=4, strides=1)(block)

    # Last activation
    block = _bn_relu(block)

    # Classifier block
    flatten1 = GlobalAveragePooling1D()(block)
    out = Dense(units=1, kernel_initializer="he_normal", activation="sigmoid")(flatten1)

    return Model(inp, out)
