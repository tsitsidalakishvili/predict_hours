import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import re
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
nltk.download('wordnet')
nltk.download('stopwords')

st.title('Task and Bug Management System')

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = None

# Sidebar for model hyperparameters and actions
with st.sidebar:
    st.header("Model Configuration")
    embedding_dim = st.slider('Embedding Dimension', 50, 300, 100, key='embed_dim')
    lstm_units = st.slider('LSTM Units', 32, 128, 64, key='lstm_units')
    dense_units = st.slider('Dense Layer Units', 16, 64, 32, key='dense_units')
    train_size = st.slider('Training Set Size (%)', 60, 90, 80, key='train_size')
    uploaded_file = st.file_uploader("Upload CSV with task data", type='csv')

    missing_value_handling = st.selectbox(
        'Select how to handle missing values:',
        ('Select method', 'Drop Rows with Missing Values', 'Fill with Mean', 'Fill with Median')
    )

# Function to preprocess text
@st.cache_data
def preprocess_text(text_series):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned_text = text_series.fillna('').str.lower().str.split().apply(
        lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x if word not in stop_words]))
    cleaned_text = cleaned_text.str.replace(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', regex=True)
    cleaned_text = cleaned_text.str.replace(r'http\S+|www\S+|<.*?>', '', regex=True)
    return cleaned_text

# Function to create the model
def create_model(vocab_size, input_length, num_numeric_data):
    text_input = Input(shape=(input_length,), name='text_input')
    text_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_input)
    text_output = Bidirectional(LSTM(lstm_units))(text_embedding)
    numeric_input = Input(shape=(num_numeric_data,), name='numeric_input')
    numeric_output = Dense(dense_units, activation='relu')(numeric_input)
    combined = Concatenate()([text_output, numeric_output])
    final_output = Dense(1, activation='linear')(Dense(dense_units, activation='relu')(combined))
    model = Model(inputs=[text_input, numeric_input], outputs=final_output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to handle missing values
def handle_missing_values(data, method):
    if method == 'Drop Rows with Missing Values':
        return data.dropna()
    elif method == 'Fill with Mean':
        return data.fillna(data.mean(numeric_only=True))
    elif method == 'Fill with Median':
        return data.fillna(data.median(numeric_only=True))

# Load data and train the model
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())

    if missing_value_handling != 'Select method':
        data = handle_missing_values(data, missing_value_handling)
        st.write("Data after handling missing values:", data.head())

        train_button = st.button('Train Model')
        if train_button:
            data['processed_text'] = preprocess_text(data['Summary'])
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(data['processed_text'])
            text_data = pad_sequences(tokenizer.texts_to_sequences(data['processed_text']), padding='post')
            vocab_size = len(tokenizer.word_index) + 1
            max_length = text_data.shape[1]
            numeric_data = np.random.rand(len(data), 10)  # Adjust this to match your numeric feature count

            # Model creation and training
            model = create_model(vocab_size, max_length, numeric_data.shape[1])
            indices = np.arange(text_data.shape[0])
            np.random.shuffle(indices)
            text_data = text_data[indices]
            numeric_data = numeric_data[indices]
            target_data = data['Hours'].iloc[indices].values

            train_samples = int((train_size / 100) * len(data))
            X_train_text = text_data[:train_samples]
            X_train_numeric = numeric_data[:train_samples]
            y_train = target_data[:train_samples]
            X_test_text = text_data[train_samples:]
            X_test_numeric = numeric_data[train_samples:]
            y_test = target_data[train_samples:]

            history = model.fit(
                [X_train_text, X_train_numeric], y_train,
                validation_data=([X_test_text, X_test_numeric], y_test),
                epochs=10, batch_size=32
            )

            st.success("Model trained successfully!")

            # Store the trained model and history in session state
            st.session_state.model = model
            st.session_state.history = history

# Display results if the model and history exist
if st.session_state.model is not None and st.session_state.history is not None:
    y_pred = st.session_state.model.predict([X_test_text, X_test_numeric])
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
    st.header('Actual vs. Predicted Values')
    st.write(results_df)

    # Visualize actual vs. predicted values in a line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Predicted'], mode='lines', name='Predicted'))
    fig.update_layout(title='Actual vs. Predicted Values', xaxis_title='Index', yaxis_title='Value')
    st.plotly_chart(fig)

    # Plot training history
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Loss Over Epochs", "Mean Absolute Error Over Epochs"))
    fig.add_trace(go.Scatter(y=st.session_state.history.history['loss'], name='Loss'), row=1, col=1)
    fig.add_trace(go.Scatter(y=st.session_state.history.history['val_mae'], name='Validation MAE'), row=2, col=1)
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig)
