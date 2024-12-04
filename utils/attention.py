# Author @ Alex Wei, based on Majumdar's implementation
# Last revised: 12/03/2024

import tensorflow as tf
from tensorflow.keras.layers import Layer, RNN
from tensorflow.keras import activations, initializers, regularizers, constraints


class ALSTMCell(Layer):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 attention_activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 attention_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 attention_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_attention=False,
                 implementation=1,
                 **kwargs):
        super(ALSTMCell, self).__init__(**kwargs)
        self.bias = None
        self.attention_kernel = None
        self.recurrent_kernel = None
        self.kernel = None
        self.input_dim = None

        # set internal attributes from params
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.attention_activation = activations.get(attention_activation)
        self.use_bias = use_bias

        # initializers config
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.attention_initializer = initializers.get(attention_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        # regularizers config
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.attention_regularizer = attention_regularizer

        # constraints config
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attention_constraint = constraints.get(attention_constraint)

        # dropout rates config
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.return_attention = return_attention
        self.implementation = implementation
        self.state_size = (self.units, self.units)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

        # initialize weights for all kernels
        self.kernel = self.add_weight(
            shape=(self.input_dim, self.units * 4),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            name='recurrent_kernel')
        self.attention_kernel = self.add_weight(
            shape=(self.input_dim, self.units * 4),
            initializer=self.attention_initializer,
            regularizer=self.attention_regularizer,
            constraint=self.attention_constraint,
            name='attention_kernel')
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias')
        else:
            self.bias = None
        self.built = True

    # References: [1][2]
    # noinspection PyMethodOverriding
    def call(self, inputs, states, training=None, mask=None):
        m1, m2 = states  # previous states

        att = tf.matmul(inputs, self.attention_kernel)    # attention weights
        if self.use_bias:
            att += self.bias
        # TODO: apply attention weights to the input (reshape and scale)
        # att = self.attention_activation(att)
        z = tf.matmul(inputs, self.kernel)
        z += tf.matmul(m1, self.recurrent_kernel)
        if self.use_bias:
            z += self.bias

        # split z into i, f, c, o
        z0, z1, z2, z3 = tf.split(z, num_or_size_splits=4, axis=1)

        # calculate gates & cell state
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * m2 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        h = o * self.activation(c)  # hidden state
        return h, [h, c]


class ALSTM(RNN):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 attention_activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 attention_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 attention_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 return_attention=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        cell = ALSTMCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            attention_activation=attention_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            attention_initializer=attention_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            attention_regularizer=attention_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            attention_constraint=attention_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_attention=return_attention)
        super(ALSTM, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
