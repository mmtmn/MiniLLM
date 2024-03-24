from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import pickle

max_length = 100
vocab_size = 10000


# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# Load the trained model
loaded_model = load_model("helpsteer_model.h5")

# Print the model summary
loaded_model.summary()

def generate_text(prompt, num_words, temperature=1.0, model=None):
    input_sequence = loaded_tokenizer.texts_to_sequences([prompt])[0]
    input_sequence = pad_sequences([input_sequence], maxlen=max_length)
    generated_text = ""
    for _ in range(num_words):
        predicted_probs = model.predict(input_sequence, verbose=0)[0]
        predicted_probs = predicted_probs / temperature
        predicted_probs = predicted_probs / np.sum(predicted_probs)
        predicted_token = np.random.choice(range(vocab_size), p=predicted_probs)
        output_word = loaded_tokenizer.index_word.get(predicted_token, '')
        generated_text += " " + output_word
        input_sequence = np.append(input_sequence, [[predicted_token]], axis=1)
        input_sequence = input_sequence[:, -max_length:]
    return generated_text.strip()

# Generate text based on a prompt
prompt = "What is the most important thing to consider when building an assist device for the elderly?"
num_words = 10
temperature = 0.8
generated_response = generate_text(prompt, num_words, temperature, loaded_model)
print("Prompt:", prompt)
print("Generated Response:", generated_response)