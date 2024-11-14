**README**

**1. PROJECT NAME:**

- Sentiment Analysis using Logistic Regression.


**2. PROJECT OVERVIW:**

- This code performs sentiment analysis on text data using a logistic regression model.
- It uses the “pandas” library to read and manipulate the data, “scikit-learn” library for vectorization and logistic regression, and “gradio” library for creating a simple web-based user interface.


**3. DEPENDENCIES:**

The following dependencies need to be installed to run the code:

- pandas
- numpy
- scikit-learn
- gradio

You can install them using pip or conda package manager.


**4. USAGE:**

- Load Data: The code reads the input data from a CSV file located at "twitterclean.csv' using the "pd.read\_csv()" function from pandas. The data should have a column named "twitts" containing the text data to be analyzed, and a column named "sentiment" containing the sentiment labels (positive or negative).
- Train the Model: The data is split into training and testing sets using pandas dataframe slicing. Then, the "TfidfVectorizer" class from scikit-learn is used to convert the text data into a numerical representation. A logistic regression model is trained on the training data using the "LogisticRegression" class from scikit-learn.
- Predict Sentiment: The "pred\_sentiment()" function takes a text input, vectorizes it using the trained vectorizer, and predicts the sentiment using the trained logistic regression model. The predicted sentiment is returned as 'positive' or 'negative' based on the predicted class label.
- Gradio Interface: The "gradio" library is used to create a simple web-based user interface for the sentiment analysis model. It takes input from the user through a textbox, passes it to the "pred\_sentiment()" function, and displays the predicted sentiment in another textbox.
- Run the Application: The "interface.launch()" function is used to launch the web-based interface with the share parameter set to "True" to generate a shareable link for the application.


**5. EXAMPLE USAGE:**

- Enter text: "He is a good student"

Sentiment: "positive"

- Enter text: "He is not a good boy"

Sentiment: "negative"

**6. LICENSE:**

- This project is licensed under the MIT LICENSE.

7\. **CONTACT INFORMATION:**

- For any questions or feedback, please contact:

Name- Niket Virendra Patil

Email- pniket7@gmail.com

