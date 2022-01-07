from ast import Num
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.autograph.pyct import transformer
from dataset import prepare_dataset, train_test_split


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

if __name__ == '__main__':
    df = prepare_dataset()
    x_train, x_test, y_train, y_test, vocab_size, glove_matrix = train_test_split(df, test_size=0.2, seq_len=30, embedding_dim=100)
    maxlen = 30
    embed_dim = 100
    num_heads = 8
    ff_dim = 512

    inputs = layers.Input(shape=(maxlen,))
    embedding = Embed(maxlen, vocab_size, embed_dim, glove_matrix=None)
    x = embedding(inputs)
    transformer_block = Transformer(embed_dim, num_heads, ff_dim, rate=0.3)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.7)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=512, epochs=20, validation_data=(x_test, y_test))

    # plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('Trans_Acc.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('Trans_Loss.png')
    plt.clf()

    # model.save('models/trans.h5')
