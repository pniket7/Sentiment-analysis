#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# In[2]:


df = pd.read_csv('/home/niket/Music/twitterclean.csv')


# In[3]:


df.head()


# In[4]:


train_data = df[:8000]
test_data = df[8000:]


# In[5]:


vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_train = vectorizer.fit_transform(train_data['twitts'].values)
X_test = vectorizer.transform(test_data['twitts'].values)


# In[6]:


model = LogisticRegression()
model.fit(X_train, train_data['sentiment'].values)


# In[7]:


accuracy = model.score(X_test, test_data['sentiment'])
print("Accuracy:", accuracy)


# In[8]:


def pred_sentiment(text):
    X = vectorizer.transform([text])
    y_pred = model.predict(X)
    sentiment = "positive" if y_pred[0] == 1 else "negative"
    return sentiment


# In[9]:


tweet=("He is a good student")
pred_sentiment(tweet)


# In[10]:


tweet=("He is not a good student")
pred_sentiment(tweet)


# In[11]:


import gradio as gr


# In[12]:


inputs = gr.inputs.Textbox(lines=4, label="Enter text:")
outputs = gr.outputs.Textbox(label="Sentiment")
interface = gr.Interface(pred_sentiment, inputs, outputs)


# In[17]:


interface.launch(share=True,server_port=6520)


# In[ ]:




