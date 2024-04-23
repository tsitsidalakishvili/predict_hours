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
from nltk.stem import WordNetLemmatizer
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
from sklearn.inspection import permutation_importance


# Ensure NLTK resources are downloaded
nltk.download('wordnet')
nltk.download('stopwords')

st.title('Task Time Predictor')

# Introduction/About section
st.markdown("""
Task Time Predictor is a tool designed to help users predict the time required to complete various tasks or projects. By leveraging natural language processing (NLP) techniques and machine learning algorithms, Task Time Predictor analyzes textual descriptions of tasks along with associated numerical features to estimate the duration needed for completion.

### Key Features:

- **Upload Task Data**: Users can upload a CSV file containing task descriptions and associated numerical data.
- **Model Configuration**: Task Time Predictor allows users to configure the model's hyperparameters, including embedding dimension, LSTM units, dense layer units, and training set size.
- **Handling Missing Values**: Users can choose from various methods to handle missing values in the uploaded data, such as dropping rows with missing values or filling them with mean or median values.
- **Training the Model**: Once the data is uploaded and preprocessing options are chosen, users can train the predictive model with the click of a button.
- **Visualization of Results**: Task Time Predictor provides visualizations of the actual vs. predicted task durations and the training history, allowing users to assess the model's performance.
""")

# Sidebar for file upload and model configuration
st.header("Upload CSV with task data")
uploaded_file = st.file_uploader("Upload CSV", type='csv')

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = None

# Function to handle missing values
def handle_missing_values(data, method):
    if method == 'Drop Rows with Missing Values':
        return data.dropna()
    elif method == 'Fill with Mean':
        return data.fillna(data.mean(numeric_only=True))
    elif method == 'Fill with Median':
        return data.fillna(data.median(numeric_only=True))





# Function to preprocess text
def preprocess_text(text_series):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def clean_text(text):
        # Convert text to lowercase
        text = text.lower()
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')  # Remove emojis by replacing them with an empty string
        # Remove URLs and emails
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)
        text = re.sub(r'http\S+|www\S+|https?://\S+', '', text)
        # Remove HTML tags and special characters
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Removes special characters, retains letters and spaces
        # Tokenize and remove stop words
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)

    return text_series.fillna('').apply(clean_text)




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





# Load data and train the model
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())

    # Ask about missing values
    st.header("Deal with Missing Values")
    missing_values_question = st.radio("Do you have missing values in the uploaded file?", ("Yes", "No"))
    if missing_values_question == "Yes":
        missing_value_handling = st.sidebar.selectbox(
            'Select how to handle missing values:',
            ('Select method', 'Drop Rows with Missing Values', 'Fill with Mean', 'Fill with Median')
        )
        if missing_value_handling != 'Select method':
            data = handle_missing_values(data, missing_value_handling)
            st.write("Data after handling missing values:", data.head())

    # Filtering for numerical columns
    numerical_columns = data.select_dtypes(include=np.number).columns.tolist()

    # Sidebar for other model configurations
    with st.sidebar:
        embedding_dim = st.slider('Embedding Dimension', 50, 300, 100)
        lstm_units = st.slider('LSTM Units', 32, 128, 64)
        dense_units = st.slider('Dense Layer Units', 16, 64, 32)
        train_size = st.slider('Training Set Size (%)', 60, 90, 80)


    st.header("Select Prediction and Target Columns")

    # Main page selection for target and prediction columns
    prediction_columns = st.multiselect("Select Columns for Predictions", data.columns, default=['Summary'] if 'Summary' in data.columns else [])
    target_column = st.selectbox("Select Target Column (Numerical)", numerical_columns)

    train_button = st.button('Train Model')

    # Train Model Button and Model Training Logic
    if train_button:
        st.write("Training started...")
        progress_bar = st.progress(0)

        # Assuming 'Summary' is selected as the text column for processing
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
        target_data = data[target_column].iloc[indices].values

        train_samples = int((train_size / 100) * len(data))
        X_train_text = text_data[:train_samples]
        X_train_numeric = numeric_data[:train_samples]
        y_train = target_data[:train_samples]
        X_test_text = text_data[train_samples:]
        X_test_numeric = numeric_data[train_samples:]
        y_test = target_data[train_samples:]

        # Initialize lists to store epoch histories
        epoch_loss = []
        epoch_val_loss = []
        epoch_mae = []
        epoch_val_mae = []

        num_epochs = 10



        for epoch in range(num_epochs):
            history = model.fit(
                [X_train_text, X_train_numeric], y_train,
                validation_data=([X_test_text, X_test_numeric], y_test),
                epochs=1, batch_size=32, verbose=0
            )

            # Collect the history from each epoch
            epoch_loss.append(history.history['loss'][0])
            epoch_val_loss.append(history.history['val_loss'][0])
            epoch_mae.append(history.history['mae'][0])
            epoch_val_mae.append(history.history['val_mae'][0])

            # Update progress bar
            progress_bar.progress((epoch + 1) / num_epochs)
            st.text(f"Epoch {epoch + 1}/{num_epochs}: loss: {history.history['loss'][0]}, mae: {history.history['mae'][0]}, val_loss: {history.history['val_loss'][0]}, val_mae: {history.history['val_mae'][0]}")

        st.success("Model trained successfully!")



        # Predictions
        y_pred = model.predict([X_test_text, X_test_numeric])
        st.session_state.predictions = y_pred

    # Display results if the model, history, and predictions are present
    if 'model' in st.session_state and 'history' in st.session_state and 'predictions' in st.session_state:
        test_data_with_predictions = data.iloc[indices[train_samples:]].copy()
        test_data_with_predictions['Predicted'] = st.session_state.predictions.flatten()
        test_data_with_predictions['Actual'] = y_test

        results_df = test_data_with_predictions[prediction_columns + ['Actual', 'Predicted']]
        
        st.header('Actual vs. Predicted Values')
        st.dataframe(results_df)
    
        # Plotting
        col1, col2 = st.columns(2)

        with col1:
            # Loss plot
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=epoch_loss, mode='lines', name='Training Loss'))
            fig_loss.add_trace(go.Scatter(y=epoch_val_loss, mode='lines', name='Validation Loss'))
            fig_loss.update_layout(title='Learning Curve', xaxis_title='Epochs', yaxis_title='Loss')
            st.plotly_chart(fig_loss)

            # Residuals plot
            residuals = y_test - st.session_state.predictions.flatten()
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(x=st.session_state.predictions.flatten(), y=residuals, mode='markers'))
            fig_residuals.update_layout(title='Residuals Plot', xaxis_title='Predicted Values', yaxis_title='Residuals')
            st.plotly_chart(fig_residuals)

        with col2:
            # Prediction accuracy plot
            fig_results = go.Figure()
            fig_results.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual'], mode='lines', name='Actual'))
            fig_results.add_trace(go.Scatter(x=results_df.index, y=results_df['Predicted'], mode='lines', name='Predicted'))
            fig_results.update_layout(title='Actual vs. Predicted Values', xaxis_title='Index', yaxis_title='Value')
            st.plotly_chart(fig_results)

            # MAE plot
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Scatter(y=epoch_val_mae, mode='lines', name='Validation MAE'))
            fig_mae.update_layout(title='Mean Absolute Error Over Epochs', xaxis_title='Epochs', yaxis_title='MAE')
            st.plotly_chart(fig_mae)





# ----------------------------------------Display results if the model and history exist ------------------------------------------------------# 





# Display results if the model and history exist
if st.session_state.model is not None and st.session_state.history is not None:
    y_pred = st.session_state.model.predict([X_test_text, X_test_numeric])
    test_data_with_predictions = data.iloc[indices[train_samples:]].copy()
    test_data_with_predictions['Predicted'] = y_pred.flatten()
    test_data_with_predictions['Actual'] = y_test
    
    # Including selected columns in the results DataFrame
    results_df = test_data_with_predictions[prediction_columns + ['Actual', 'Predicted']]
    
    
    
    st.header('Actual vs. Predicted Values')
    st.dataframe(results_df)
  
  
  
  
    # Define the layout for the 2x2 grid
    col1, col2 = st.columns(2)
    


    # Column 1: Table, Loss Chart, and Residuals Plot
    with col1:


        # Plot training history for Loss
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=epoch_loss, mode='lines', name='Training Loss'))
        fig_loss.add_trace(go.Scatter(y=epoch_val_loss, mode='lines', name='Validation Loss'))
        fig_loss.update_layout(title='Learning Curve', xaxis_title='Epochs', yaxis_title='Loss')
        st.plotly_chart(fig_loss)

        # Plot residuals
        residuals = y_test - y_pred.flatten()
        fig_residuals = go.Figure()
        fig_residuals.add_trace(go.Scatter(x=y_pred.flatten(), y=residuals, mode='markers', name='Residuals'))
        fig_residuals.update_layout(title='Residuals Plot', xaxis_title='Predicted Values', yaxis_title='Residuals')
        st.plotly_chart(fig_residuals)

    # Column 2: Prediction Chart and MAE Chart
    with col2:
        # Visualize actual vs. predicted values in a line chart
        fig_results = go.Figure()
        fig_results.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual'], mode='lines', name='Actual'))
        fig_results.add_trace(go.Scatter(x=results_df.index, y=results_df['Predicted'], mode='lines', name='Predicted'))
        fig_results.update_layout(title='Actual vs. Predicted Values', xaxis_title='Index', yaxis_title='Value')
        st.plotly_chart(fig_results)

        # Plot training history for Mean Absolute Error
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Scatter(y=epoch_val_mae, name='Validation MAE', mode='lines'))
        fig_mae.update_layout(title='Mean Absolute Error Over Epochs', xaxis_title='Epochs', yaxis_title='MAE')
        st.plotly_chart(fig_mae)