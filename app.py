import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the dataset
file_path = "sma.csv"  # Update path if needed
df = pd.read_csv(file_path, encoding="ISO-8859-1")
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

# Convert labels to binary (spam = 1, ham = 0)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text Preprocessing Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

df['cleaned_message'] = df['message'].apply(clean_text)

# Tokenization & Padding
max_words = 5000  # Limit vocabulary size
max_len = 100  # Max length of each sequence

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned_message'])
X = tokenizer.texts_to_sequences(df['cleaned_message'])
X = pad_sequences(X, maxlen=max_len)
y = np.array(df['label'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM Model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    SpatialDropout1D(0.2),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')  # Output layer (spam or ham)
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the trained model to a file (if it's not already saved)
model.save('spam_model.h5')

# Load the model for prediction
model = load_model('spam_model.h5')

# Prediction function
@app.route("/predict", methods=["POST"])
def predict():
    text = request.json['message']
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(seq)[0][0]
    return jsonify({"prediction": "Spam" if prediction > 0.5 else "Ham"})

if __name__ == "__main__":
    app.run(debug=True)
