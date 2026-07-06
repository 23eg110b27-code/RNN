#RNN practical(many to one)
# SMS spam detection using simple RNN(many to one)
# Dataset: spam.csv
# Imoport Libraries
import os
import re
import pickle
from tabnanny import verbose
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#configuration

MODEL = "spam_model.keras"
TOKENIZER = "tokenizer.pkl"

MAX_WORDS = 5000
MAX_LEN = 50

#clean text

def clean_text(text):
    text = str(text).lower()
    text =re.sub(r"[^a-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

#train model
def train_model():
    print("Training Dataset...")
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]
    print(df.head())
    print(df[["label"]].value_counts())

    #convert labels into numbers
    df["label"] = df["label"].map({
        "ham": 0, 
        "spam": 1
    })
    #clean SMS
    df["message"] = df["message"].apply(clean_text)

    #Tokenization
    tokenizer = Tokenizer(num_words=MAX_WORDS,
                           oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(df["message"])
    sequences = tokenizer.texts_to_sequences(df["message"])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")
    y = df["label"]
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    #save tokenizer
    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

        #Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Build RNN model
        model = Sequential()

        #Embedding layer
        model.add(
           Embedding(
                input_dim=MAX_WORDS,
                output_dim=128,
                input_length=MAX_LEN)
        )

        #Simple RNN layer
        model.add(
            SimpleRNN(
                128
            ))
        
        #midense layer
        model.add(Dense(32, activation="relu"))


        #output layer
        model.add(Dense(1, activation="sigmoid"))
        model.summary()
        
        #train model
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2
        )


        #save the model
        model.save(MODEL)

        #evaluate 
        loss, accuracy = model.evaluate(X_test, y_test)
        print("\nAccuracy:", accuracy)
        #predictions

        predictions = (
            model.predict(X_test)>0.5
        ).astype(int)

        print(
            classification_report(y_test, predictions)
        )

        print(
            confusion_matrix(y_test, predictions)
        )

        #prediction
        def predict_sms(message):
            model = load_model(MODEL)
            with open(TOKENIZER, "rb") as f:
                tokenizer = pickle.load(f)
                message = clean_text(message)
                sequence = tokenizer.texts_to_sequences([message])

                sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")

                prediction = model.predict(
                    X_test,
                    verbose=0
                    )


                if probability > 0.5:
                    return "Spam", probability
                else:
                    return "Ham", 1 - probability
                
            if not os.path.exists(MODEL):
                train_model()    

            #streamlit UI
            st.title("SMS Spam Detection")
            st.write("Many to one RNN Example")

            message = st.text_area(
                "Enter SMS message:"
            )

            if st.button("Predict"):
                prediction, probability = predict_sms(message)
                st.success(prediction)

                st.write(
                    "confidence:", round(probability * 100, 2), "%"
                )