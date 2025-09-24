import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Attention
import pickle
import os

# ---------- Config ----------
data_path = 'eng-fra.txt'   # <-- your dataset filename
max_lines = 10000           # reduce to e.g. 2000 for quick testing

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}. Place eng-fra.txt in the script folder or update data_path.")

# ---------- Load + Clean ----------
lines = open(data_path, encoding='utf-8').read().strip().split('\n')

english_sentences = []
french_sentences = []

for line in lines[:max_lines]:
    parts = line.split('\t')
    if len(parts) >= 2:
        eng = parts[0].strip()
        fra = parts[1].strip()
        if eng and fra:
            english_sentences.append(eng.lower())
            french_sentences.append(fra.lower())

print(f"Loaded {len(english_sentences)} sentence pairs")

# Add space-separated start/end tokens
french_sentences = ['<start> ' + sent + ' <end>' for sent in french_sentences]

# ---------- Tokenize ----------
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(english_sentences)
eng_seq = eng_tokenizer.texts_to_sequences(english_sentences)
max_eng_len = max(len(seq) for seq in eng_seq)
eng_seq = pad_sequences(eng_seq, maxlen=max_eng_len, padding='post')

fra_tokenizer = Tokenizer(filters='')  # keep punctuation if desired
fra_tokenizer.fit_on_texts(french_sentences)
fra_seq = fra_tokenizer.texts_to_sequences(french_sentences)
max_fra_len = max(len(seq) for seq in fra_seq)
fra_seq = pad_sequences(fra_seq, maxlen=max_fra_len, padding='post')

eng_vocab = len(eng_tokenizer.word_index) + 1
fra_vocab = len(fra_tokenizer.word_index) + 1

print("English vocab:", eng_vocab)
print("French vocab:", fra_vocab)
print("Max English length:", max_eng_len)
print("Max French length:", max_fra_len)

# ---------- Prepare target sequences ----------
decoder_input_seq = fra_seq[:, :-1]
decoder_target_seq = fra_seq[:, 1:]
decoder_target_seq = np.expand_dims(decoder_target_seq, -1)

# ---------- Model hyperparams ----------
embedding_dim = 256
units = 512

# ---------- Encoder ----------
encoder_inputs = Input(shape=(max_eng_len,), name='encoder_inputs')
enc_embedding_layer = Embedding(input_dim=eng_vocab, output_dim=embedding_dim, name='enc_embedding')
enc_emb = enc_embedding_layer(encoder_inputs)
encoder_lstm = LSTM(units, return_sequences=True, return_state=True, name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

# ---------- Decoder (training) ----------
decoder_inputs = Input(shape=(max_fra_len-1,), name='decoder_inputs')
dec_embedding_layer = Embedding(input_dim=fra_vocab, output_dim=embedding_dim, name='dec_embedding')
dec_emb = dec_embedding_layer(decoder_inputs)
decoder_lstm = LSTM(units, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# Attention and output
attn_layer = Attention(name='attention_layer')
attn_out = attn_layer([decoder_outputs, encoder_outputs])
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
decoder_dense = Dense(fra_vocab, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_concat_input)

# Full training model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------- Train ----------
# For quick debugging reduce epochs or max_lines
model.fit([eng_seq, decoder_input_seq], decoder_target_seq,
          batch_size=64, epochs=10, validation_split=0.2)

# ---------- Save ----------
model.save('eng_fra_translator.h5')
with open('eng_tokenizer.pkl', 'wb') as f:
    pickle.dump(eng_tokenizer, f)
with open('fra_tokenizer.pkl', 'wb') as f:
    pickle.dump(fra_tokenizer, f)

# ---------- Inference models ----------
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

decoder_state_input_h = Input(shape=(units,), name='dec_state_h')
decoder_state_input_c = Input(shape=(units,), name='dec_state_c')
encoder_outputs_input = Input(shape=(max_eng_len, units), name='enc_out_input')
decoder_single_input = Input(shape=(1,), name='decoder_single_input')
dec_emb2 = dec_embedding_layer(decoder_single_input)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])
attn_out2 = attn_layer([decoder_outputs2, encoder_outputs_input])
decoder_concat_input2 = Concatenate(axis=-1)([decoder_outputs2, attn_out2])
decoder_outputs2 = decoder_dense(decoder_concat_input2)

decoder_model = Model([decoder_single_input, encoder_outputs_input, decoder_state_input_h, decoder_state_input_c],
                      [decoder_outputs2, state_h2, state_c2])

# ---------- Translation function ----------
def translate_sentence(sentence, max_len=max_fra_len):
    seq = eng_tokenizer.texts_to_sequences([sentence.lower().strip()])
    seq = pad_sequences(seq, maxlen=max_eng_len, padding='post')
    enc_outs, h, c = encoder_model.predict(seq)
    start_idx = fra_tokenizer.word_index.get('<start>')
    end_idx = fra_tokenizer.word_index.get('<end>')
    if start_idx is None or end_idx is None:
        raise ValueError("'<start>' or '<end>' token missing from tokenizer.")
    target_seq = np.array([[start_idx]])
    output_sentence = []
    for _ in range(max_len):
        dec_outs, h, c = decoder_model.predict([target_seq, enc_outs, h, c])
        sampled_token_index = np.argmax(dec_outs[0, -1, :])
        if sampled_token_index == 0 or sampled_token_index == end_idx:
            break
        sampled_word = fra_tokenizer.index_word.get(sampled_token_index, '')
        if not sampled_word:
            break
        output_sentence.append(sampled_word)
        target_seq = np.array([[sampled_token_index]])
    return ' '.join(output_sentence)

# ---------- Demo ----------
print("I am happy ->", translate_sentence("I am happy"))
print("How are you? ->", translate_sentence("How are you?"))
print("This is a test ->", translate_sentence("This is a test"))
