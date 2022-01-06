import numpy as np
import re
import string
from unicodedata import normalize
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, TimeDistributed, RepeatVector
from tensorflow.keras import Model, Sequential
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu


def load_data():
    with open('data/deu.txt', 'rt', encoding='utf-8') as f:
        text = f.read()
    lines = text.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs

def clean_data(pairs):
    cleaned = []
    re_filt = re.compile('[^%s]' % re.escape(string.printable))
    table = str.maketrans('', '', string.punctuation)
    for pair in pairs:
        clean_pair = []
        for line in pair:
            line = normalize('NFD',line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            line = line.split()
            line = [word.lower() for word in line]
            # remove punct
            line = [word.translate(table) for word in line]
            # remove nonprintable chars
            line = [re_filt.sub('', w) for w in line]
            # remove tokens with numbers
            line = [word for word in line if word.isalpha()]
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return np.array(cleaned)

def max_length(lines):
    return max(len(line.split()) for line in lines)

def create_tokenizer(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    return tokenizer

def encode_seqs(tokenizer, length, data):
    X = tokenizer.texts_to_sequences(data)
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

def encode_outs(sequences, vocab_size):
    ylist = []
    for s in sequences:
        encoded = to_categorical(s, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

def build_model(src_vocab, tgt_vocab, src_timesteps, tgt_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(Bidirectional(LSTM(n_units)))
    model.add(RepeatVector(tgt_timesteps))
    model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
    model.add(TimeDistributed(Dense(tgt_vocab, activation='softmax')))
    return model

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [np.argmax(vector) for vector in prediction]
    target = []
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)

def eval_model(model, tokenizer, source, raw_dataset):
    actual, pred = [], []
    for i, s in enumerate(source):
        s = s.reshape((1, s.shape[0]))
        translation = predict_sequence(model, tokenizer, s)
        raw_target, raw_src = raw_dataset[i]
        if i < 10:
            print(f'src = {raw_src}, tar = {raw_target}, pred = {translation}')
        actual.append([raw_target.split()])
        pred.append(translation.split())
    print(f'BLEU Score 1: {corpus_bleu(actual, pred, weights=(1.0, 0, 0, 0))}')
    print(f'BLEU Score 2: {corpus_bleu(actual, pred, weights=(0.5, 0.5, 0, 0))}')
    print(f'BLEU Score 3: {corpus_bleu(actual, pred, weights=(0.3, 0.3, 0.3, 0))}')
    print(f'BLEU Score 4: {corpus_bleu(actual, pred, weights=(0.25, 0.25, 0.25, 0.25))}')


if __name__ == '__main__':
    clean = clean_data(load_data())

    eng_tokenizer = create_tokenizer(clean[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_max_length = max_length(clean[:, 0])
    print(f'English vocab size: {eng_vocab_size}')
    print(f'English max length: {eng_max_length}')

    deu_tokenizer = create_tokenizer(clean[:, 1])
    deu_vocab_size = len(deu_tokenizer.word_index) + 1
    deu_max_length = max_length(clean[:, 1])
    print(f'Deutch vocab size: {deu_vocab_size}')
    print(f'Deutch max length: {deu_max_length}')

    # data
    train, test = train_test_split(clean, test_size=0.2)
    print(train.shape, test.shape)
    print(train[0])
    trainx = encode_seqs(deu_tokenizer, deu_max_length, train[:, 1])
    trainy = encode_seqs(eng_tokenizer, eng_max_length, train[:, 0])
    # trainy = encode_outs(trainy, eng_vocab_size)
    testx = encode_seqs(deu_tokenizer, deu_max_length, test[:, 1])
    testy = encode_seqs(eng_tokenizer, eng_max_length, test[:, 0])
    # testy = encode_outs(testy, eng_vocab_size)

    print(trainy.shape)
    model = build_model(deu_vocab_size, eng_vocab_size, deu_max_length, eng_max_length, 256)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    history = model.fit(trainx, trainy, batch_size=256, epochs=30, validation_data=(testx, testy), verbose=1)

    print('TRAIN')
    eval_model(model, eng_tokenizer, trainx, train)
    print('TEST')
    eval_model(model, eng_tokenizer, testx, test)
