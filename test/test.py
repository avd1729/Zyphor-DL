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
The quick brown fox jumps over the lazy dog. A journey of a thousand miles begins with a single step.
All that glitters is not gold. Actions speak louder than words. Better late than never.
Birds of a feather flock together. Cleanliness is next to godliness. Don't judge a book by its cover.
Easy come, easy go. Every cloud has a silver lining. Fortune favors the bold.
Good things come to those who wait. Honesty is the best policy. If at first you don't succeed, try, try again.
Knowledge is power. Laughter is the best medicine. Money doesn't grow on trees.
Necessity is the mother of invention. Once bitten, twice shy. Practice makes perfect.
Rome wasn't built in a day. Strike while the iron is hot. The early bird catches the worm.
Time flies when you're having fun. Two wrongs don't make a right. Where there's a will, there's a way.
You can't have your cake and eat it too. A penny saved is a penny earned. Curiosity killed the cat.
Don't count your chickens before they hatch. The grass is always greener on the other side.
Look before you leap. The pen is mightier than the sword. When in Rome, do as the Romans do.
Absence makes the heart grow fonder. Beauty is in the eye of the beholder. Beggars can't be choosers.
Better safe than sorry. Blood is thicker than water. Clothes make the man. Don't put all your eggs in one basket.
Every dog has its day. Familiarity breeds contempt. Haste makes waste. Hope for the best, prepare for the worst.
If the shoe fits, wear it. It's no use crying over spilled milk. It takes two to tango. Keep your friends close and your enemies closer.
Let bygones be bygones. Like father, like son. Many hands make light work. Never look a gift horse in the mouth.
No pain, no gain. Out of sight, out of mind. Patience is a virtue. Pride comes before a fall.
The apple doesn't fall far from the tree. The best things in life are free. The customer is always right. The devil is in the details.
The early bird catches the worm. The more the merrier. The squeaky wheel gets the grease. There's no place like home.
There's no smoke without fire. Time heals all wounds. Too many cooks spoil the broth. Two heads are better than one.
When it rains, it pours. You can lead a horse to water, but you can't make it drink. You can't teach an old dog new tricks.
A friend in need is a friend indeed. A picture is worth a thousand words. A stitch in time saves nine. Actions speak louder than words.
All good things must come to an end. All's fair in love and war. An apple a day keeps the doctor away. An ounce of prevention is worth a pound of cure.
Art imitates life. As you sow, so shall you reap. Ask and you shall receive. Beauty is only skin deep.
Behind every successful man is a woman. Better late than never. Blood is thicker than water. Boys will be boys.
Business before pleasure. Carpe diem (Seize the day). Charity begins at home. Curiosity killed the cat.
Diamond cuts diamond. Different strokes for different folks. Do unto others as you would have them do unto you. Don't bite the hand that feeds you.
Don't cry over spilled milk. Don't judge a book by its cover. Don't look a gift horse in the mouth. Don't make a mountain out of a molehill.
Don't put off until tomorrow what you can do today. East or West, home is best. Easy come, easy go. Every cloud has a silver lining.
Every dog has his day. Every man for himself. Everything comes to him who waits. First come, first served.
Fools rush in where angels fear to tread. Forewarned is forearmed. Fortune favors the brave. God helps those who help themselves.
Good things come to those who wait. Great minds think alike. Half a loaf is better than no bread. Haste makes waste.
He laughs best who laughs last. He who hesitates is lost. Health is better than wealth. Honesty is the best policy.
Hope for the best, but prepare for the worst. If at first you don't succeed, try, try again. If the cap fits, wear it. If wishes were horses, beggars would ride.
Ignorance is bliss. In for a penny, in for a pound. It never rains but it pours. It takes two to tango.
It's a small world. It's better to be safe than sorry. It's never too late to learn. It's no use crying over spilled milk.
It's the early bird that catches the worm. Keep your friends close and your enemies closer. Knowledge is power. Laughter is the best medicine.
Learn to walk before you run. Let bygones be bygones. Let sleeping dogs lie. Life begins at forty.
Life is what you make it. Like father, like son. Live and let live. Look before you leap.
Love is blind. Make hay while the sun shines. Man proposes, God disposes. Many hands make light work.
Money doesn't grow on trees. Money talks. Necessity is the mother of invention. Never put off until tomorrow what you can do today.
Never say never. Never too old to learn. No man is an island. No news is good news.
No pain, no gain. Nothing ventured, nothing gained. Once bitten, twice shy. One man's meat is another man's poison.
One swallow doesn't make a summer. Out of sight, out of mind. Patience is a virtue. Penny wise, pound foolish.
People who live in glass houses shouldn't throw stones. Practice makes perfect. Prevention is better than cure. Pride comes before a fall.
Put your best foot forward. Rome wasn't built in a day. Seeing is believing. Slow and steady wins the race.
Still waters run deep. Strike while the iron is hot. The best things in life are free. The bigger they are, the harder they fall.
The customer is always right. The devil finds work for idle hands. The early bird catches the worm. The end justifies the means.
The grass is always greener on the other side of the fence. The leopard doesn't change his spots. The more the merrier. The pen is mightier than the sword.
The proof of the pudding is in the eating. The squeaky wheel gets the grease. There's no place like home. There's no such thing as a free lunch.
There's no time like the present. Time and tide wait for no man. Time is money. Time will tell.
To err is human, to forgive divine. Tomorrow is another day. Too many cooks spoil the broth. Truth is stranger than fiction.
Two heads are better than one. Two wrongs don't make a right. Variety is the spice of life. Waste not, want not.
When in Rome, do as the Romans do. When the cat's away, the mice will play. Where there's a will, there's a way. You can't have your cake and eat it too.
You can't judge a book by its cover. You can't teach an old dog new tricks. You reap what you sow. Youth is wasted on the young.
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
