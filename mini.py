import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from datasets import load_dataset
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Load the HelpSteer dataset
ds = load_dataset("nvidia/HelpSteer")
train_data = ds['train']
val_data = ds['validation']

# Preprocess the dataset
def preprocess_data(data):
    prompts = [entry['prompt'] for entry in data]
    responses = [entry['response'] for entry in data]
    return prompts, responses

train_prompts, train_responses = preprocess_data(train_data)
val_prompts, val_responses = preprocess_data(val_data)

# Define the model hyperparameters
vocab_size = 10000
max_length = 100
embedding_size = 128
lstm_units = 256
dropout_rate = 0.2

# Prepare the training data
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_responses)
sequences = tokenizer.texts_to_sequences(train_responses)
x_train = pad_sequences(sequences, maxlen=max_length)
y_train = np.zeros((len(sequences), vocab_size))
for i, seq in enumerate(sequences):
    for j in range(1, len(seq)):
        y_train[i, seq[j]] = 1

# Prepare the validation data
val_sequences = tokenizer.texts_to_sequences(val_responses)
x_val = pad_sequences(val_sequences, maxlen=max_length)
y_val = np.zeros((len(val_sequences), vocab_size))
for i, seq in enumerate(val_sequences):
    for j in range(1, len(seq)):
        y_val[i, seq[j]] = 1

# Define the model architecture
model = Sequential([
    Embedding(vocab_size, embedding_size, input_shape=(max_length,)),
    LSTM(lstm_units, return_sequences=True),
    Dropout(dropout_rate),
    LSTM(lstm_units),
    Dropout(dropout_rate),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001))

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_val, y_val), callbacks=[early_stopping])

# Save the trained model
model.save("helpsteer_model.h5")

# Generate text using the trained model
def generate_text(prompt, num_words, temperature=1.0, model=None):
    input_sequence = tokenizer.texts_to_sequences([prompt])[0]
    input_sequence = pad_sequences([input_sequence], maxlen=max_length)
    generated_text = ""
    for _ in range(num_words):
        predicted_probs = model.predict(input_sequence, verbose=0)[0]
        predicted_probs = predicted_probs / temperature
        predicted_probs = predicted_probs / np.sum(predicted_probs)
        predicted_token = np.random.choice(range(vocab_size), p=predicted_probs)
        output_word = tokenizer.index_word.get(predicted_token, '')
        generated_text += " " + output_word
        input_sequence = np.append(input_sequence, [[predicted_token]], axis=1)
        input_sequence = input_sequence[:, -max_length:]
    return generated_text.strip()

# Generate text based on a prompt
prompt = "What are the three most important things to consider when deciding what technology to use to build an assist device to help an elderly person with basic needs?"
num_words = 50
temperature = 0.8
generated_response = generate_text(prompt, num_words, temperature, model)
print("Prompt:", prompt)
print("Generated Response:", generated_response)

# Load the trained model
from keras.models import load_model
loaded_model = load_model("helpsteer_model.h5")

# Generate text based on a prompt
prompt = "What are the three most important things to consider when deciding what technology to use to build an assist device to help an elderly person with basic needs?"
num_words = 50
temperature = 0.8
generated_response = generate_text(prompt, num_words, temperature, loaded_model)
print("Prompt:", prompt)
print("Generated Response:", generated_response)