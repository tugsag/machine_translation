from ast import Num
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.autograph.pyct import transformer


class Transformer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(Transformer, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation='relu'), layers.Dense(embed_dim),])
        self.layernorm1 = layers.LayerNormalization(epsilon=0.001)
        self.layernorm2 = layers.LayerNormalization(epsilon=0.001)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        x = self.attention(inputs, inputs)
        x = self.dropout1(x, training=training)
        out1 = self.layernorm1(inputs + x)
        y = self.ffn(x)
        y = self.dropout2(x, training=training)
        return self.layernorm2(out1 + y)

class Embed(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, glove_matrix=None):
        super(Embed, self).__init__()
        if glove_matrix is not None:
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, weights=[glove_matrix], trainable=False)
        else:
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

