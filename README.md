# Mini LLM

Mini LLM is a lightweight language model built using Keras and LSTM (Long Short-Term Memory) networks. This project demonstrates a basic implementation of a text generation model using the HelpSteer dataset. It's designed to provide a foundational understanding of how language models can be trained and used for generating text.

## Features

- Uses LSTM networks for text generation.
- Implements a simple preprocessing pipeline for text data.
- Employs EarlyStopping to prevent overfitting.
- Includes a mechanism for saving and loading both the model and tokenizer.
- Customizable text generation with temperature setting for creativity control.

## Installation

Before running Mini LLM, ensure you have the following prerequisites installed:
- Python 3.x
- Numpy
- Keras
- TensorFlow
- HuggingFace's `datasets` library

Use the following command to install required packages:

```bash
pip install numpy keras tensorflow datasets
```

## Usage

### Training the Model

Run the provided Python script to train the model. The script includes data loading, preprocessing, model training, and saving the trained model and tokenizer.

### Generating Text

After training, you can use the `generate_text` function to generate text based on a given prompt. Example usage:

```python
prompt = "Enter your prompt here"
num_words = 20
temperature = 0.8
print(generate_text(prompt, num_words, temperature, model))
```

### Loading the Model

You can also load the saved model and tokenizer to generate text without needing to retrain:

```python
from keras.models import load_model
model = load_model("path_to_saved_model.h5")
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

print(generate_text(prompt, num_words, temperature, model))
```
