# Import necessary libraries
import pandas as pd  # for data manipulation
import numpy as np  # for numerical operations
from sklearn.feature_extraction.text import TfidfVectorizer  # for text vectorization
from sklearn.linear_model import LogisticRegression  # for building the logistic regression model

# Load the dataset from a CSV file
df = pd.read_csv('/twitterclean.csv')

# Display the first few rows of the dataset to check data structure
df.head()

# Split the data into training and testing sets
train_data = df[:8000]  # first 8000 samples for training
test_data = df[8000:]  # remaining samples for testing

# Initialize the TF-IDF vectorizer with n-gram range (1, 3) for unigrams, bigrams, and trigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_train = vectorizer.fit_transform(train_data['twitts'].values)  # Transform training data
X_test = vectorizer.transform(test_data['twitts'].values)  # Transform test data

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, train_data['sentiment'].values)

# Evaluate the model on the test data and print the accuracy
accuracy = model.score(X_test, test_data['sentiment'])
print("Accuracy:", accuracy)

# Define a function to predict sentiment for a given text
def pred_sentiment(text):
    X = vectorizer.transform([text])  # Transform the input text using the TF-IDF vectorizer
    y_pred = model.predict(X)  # Predict sentiment
    sentiment = "positive" if y_pred[0] == 1 else "negative"  # Map prediction to sentiment label
    return sentiment

# Test the sentiment prediction function with example tweets
tweet = "He is a good student"
pred_sentiment(tweet)

tweet = "He is not a good student"
pred_sentiment(tweet)

# Import Gradio for creating a simple web interface
import gradio as gr

# Define Gradio input and output components for the web interface
inputs = gr.inputs.Textbox(lines=4, label="Enter text:")  # Input text box
outputs = gr.outputs.Textbox(label="Sentiment")  # Output text box

# Create the Gradio interface with the prediction function and defined input/output
interface = gr.Interface(pred_sentiment, inputs, outputs)

# Launch the Gradio interface with sharing enabled on a specific port
interface.launch(share=True, server_port=6520)
