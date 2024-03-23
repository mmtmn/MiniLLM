import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from training_data import texts

# Define the model hyperparameters
vocab_size = 10000  # Reduced vocabulary size
max_length = 50  # Reduced maximum sequence length
embedding_size = 128  # Increased embedding dimension
lstm_units = 256  # Increased number of LSTM units
dropout_rate = 0.2  # Added dropout rate

# Define the model architecture
model = Sequential([
    Embedding(vocab_size, embedding_size, input_shape=(max_length,)),
    LSTM(lstm_units, return_sequences=True),
    Dropout(dropout_rate),  # Added dropout layer
    LSTM(lstm_units),
    Dropout(dropout_rate),  # Added dropout layer
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Prepare the training data
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
x_train = pad_sequences(sequences, maxlen=max_length)
y_train = np.zeros((len(sequences), vocab_size))
for i, seq in enumerate(sequences):
    for j in range(1, len(seq)):
        y_train[i, seq[j]] = 1

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=20)  # Increased number of epochs

# Generate text using the trained model
def generate_text(seed_text, num_words, temperature=1.0):
    generated_text = seed_text
    input_sequence = tokenizer.texts_to_sequences([seed_text])[0]
    input_sequence = pad_sequences([input_sequence], maxlen=max_length)

    for _ in range(num_words):
        predicted_probs = model.predict(input_sequence, verbose=0)[0]
        predicted_probs = predicted_probs / temperature
        predicted_probs = predicted_probs / np.sum(predicted_probs)
        predicted_token = np.random.choice(range(vocab_size), p=predicted_probs)

        output_word = tokenizer.index_word.get(predicted_token, '')
        generated_text += " " + output_word

        input_sequence = np.append(input_sequence, [[predicted_token]], axis=1)
        input_sequence = input_sequence[:, -max_length:]

    return generated_text

# Generate text
seed_text = "Hey"
num_words = 20
temperature = 0.8  # Adjust the temperature for diversity (higher value = more diverse)
generated_text = generate_text(seed_text, num_words, temperature)
print(generated_text)