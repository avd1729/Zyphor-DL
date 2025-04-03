import tensorflow as tf
import numpy as np
import pandas as pd
import os
import re
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Parameters
MAX_SEQUENCE_LENGTH = 5  # Consider 5 previous words
EMBEDDING_DIM = 100
VOCAB_SIZE = 10000
BATCH_SIZE = 128
EPOCHS = 5

text_data = """
This is example text to train a very simple next-word prediction model.
You should replace this with your actual training data.
The more text you have, the better your model will perform.
Use high-quality text that represents the kind of content your users will type.
"""

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)      # Remove multiple spaces
    return text.strip()

processed_text = preprocess_text(text_data)
words = processed_text.split()

# Create input-output pairs for next word prediction
input_sequences = []
for i in range(1, len(words)):
    start_idx = max(0, i - MAX_SEQUENCE_LENGTH)
    input_seq = words[start_idx:i]
    output_word = words[i]
    input_sequences.append((input_seq, output_word))

# Tokenize words
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts([processed_text])
vocab_size = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)

# Create sequences and labels
X = []
y = []

for input_seq, output_word in input_sequences:
    input_seq_tokens = tokenizer.texts_to_sequences([' '.join(input_seq)])[0]
    padded_input = pad_sequences([input_seq_tokens], maxlen=MAX_SEQUENCE_LENGTH, padding='pre')[0]
    output_token = tokenizer.texts_to_sequences([[output_word]])[0]
    
    if output_token:
        X.append(padded_input)
        y.append(output_token[0])

X = np.array(X)
y = np.array(y)

# Convert labels to one-hot encoding
y_one_hot = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Build model
model = Sequential([
    Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train model
model.fit(X, y_one_hot, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Save model in TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False  # Fix tensor list ops issue

tflite_model = converter.convert()

with open('models/next_word_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Save vocabulary
with open('vocabulary/vocabulary.txt', 'w') as f:
    for word, index in sorted(tokenizer.word_index.items(), key=lambda x: x[1]):
        if index < VOCAB_SIZE:
            f.write(f"{word}\n")

print("Model and vocabulary saved successfully.")
