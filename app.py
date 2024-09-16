import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Binarizer, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt

# Function to fetch financial data
@st.cache_data
def fetch_yfinance_data(ticker, period='1y'):
    try:
        stock_data = yf.download(ticker, period=period)
        stock_data.reset_index(inplace=True)
        return stock_data
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}. Error: {e}")
        return pd.DataFrame()

# Streamlit app
st.title('ML Advanced Feature Engineering Playground')

st.markdown("""
    **Feature Engineering Techniques:**

    - **Scaling**: Normalize numeric features to have a mean of 0 and a standard deviation of 1. Useful for algorithms that rely on distance metrics, such as K-Nearest Neighbors (KNN) and Support Vector Machines (SVM).
    - **Binning**: Convert continuous variables into categorical bins to reduce the impact of outliers. Useful for decision trees and models that are sensitive to outliers.
    - **Text Features**: Extract numerical features from text data using methods like Count Vectorizer or TF-IDF Vectorizer. Useful for Natural Language Processing (NLP) tasks and models like Logistic Regression and Naive Bayes.
    - **Encoding**: Convert categorical data into numerical format using techniques like One-Hot Encoding. Useful for algorithms that require numerical input, such as Linear Regression and Neural Networks.
    - **Imputation**: Handle missing values by filling them with mean or median values. Useful for all types of machine learning models to ensure data completeness.
    - **Power Transforms**: Apply transformations such as logarithmic to stabilize variance and make the data more normally distributed. Useful for linear models and algorithms that assume normality in data.
    """)

# File upload
uploaded_file = st.file_uploader("Upload your own dataset (CSV format)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", data.head())
else:
    # Input for stock ticker and period
    ticker = st.text_input('or Fetch sample data from Stock API', 'AAPL')
    period = st.selectbox('Select Period', ['1d', '1mo', '1y', '5y'])

    # Fetch data
    data = fetch_yfinance_data(ticker, period)

if data.empty:
    st.write("No data available.")
else:
    # Header with menu
    menu = st.selectbox('Select Feature Engineering Technique',
                        ['None', 'Scaling', 'Binning', 'Text Features', 'Encoding', 'Imputation', 'Power Transforms'])

    if menu == 'Scaling':
        columns_to_scale = st.multiselect('Select Columns to Scale', data.columns.tolist())
        if columns_to_scale:
            scaler = StandardScaler()
            data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
            st.write("Scaled Data", data.head())
            st.write("Scaled Data Distribution")
            fig, ax = plt.subplots(figsize=(6, 3))
            data[columns_to_scale].hist(ax=ax)
            plt.tight_layout()
            st.pyplot(fig)

    elif menu == 'Binning':
        bin_column = st.selectbox('Select Column to Bin', data.columns.tolist())
        bins = st.number_input('Number of Bins', min_value=2, value=5)
        if bin_column:
            data[bin_column + '_binned'] = pd.cut(data[bin_column], bins=bins)
            st.write("Binned Data", data.head())
            st.write("Binned Data Distribution")
            fig, ax = plt.subplots(figsize=(6, 3))
            data[bin_column + '_binned'].value_counts().plot(kind='bar', ax=ax)
            plt.tight_layout()
            st.pyplot(fig)

    elif menu == 'Text Features':
        text_column = st.selectbox('Select Text Column', data.columns.tolist())
        if text_column:
            vectorizer = st.selectbox('Select Vectorizer', ['Count Vectorizer', 'TF-IDF Vectorizer'])
            if vectorizer == 'Count Vectorizer':
                cv = CountVectorizer()
                text_features = cv.fit_transform(data[text_column].astype(str)).toarray()
                feature_names = cv.get_feature_names_out()
            elif vectorizer == 'TF-IDF Vectorizer':
                tfidf = TfidfVectorizer()
                text_features = tfidf.fit_transform(data[text_column].astype(str)).toarray()
                feature_names = tfidf.get_feature_names_out()
            feature_df = pd.DataFrame(text_features, columns=feature_names)
            st.write("Text Features", feature_df.head())

    elif menu == 'Encoding':
        encoding_column = st.selectbox('Select Column to Encode', data.columns.tolist())
        if encoding_column:
            encoder = OneHotEncoder(sparse=False)
            encoded_features = encoder.fit_transform(data[[encoding_column]])
            feature_names = encoder.get_feature_names_out([encoding_column])
            encoded_df = pd.DataFrame(encoded_features, columns=feature_names)
            st.write("Encoded Data", encoded_df.head())

    elif menu == 'Imputation':
        impute_column = st.selectbox('Select Column for Imputation', data.columns.tolist())
        if impute_column:
            imputer = SimpleImputer(strategy='mean')
            data[impute_column] = imputer.fit_transform(data[[impute_column]])
            st.write("Data after Imputation", data.head())

    elif menu == 'Power Transforms':
        power_column = st.selectbox('Select Column for Power Transform', data.columns.tolist())
        if power_column:
            transformer = FunctionTransformer(np.log1p, validate=True)
            data[power_column] = transformer.fit_transform(data[[power_column]])
            st.write("Data after Power Transform", data.head())
            st.write("Transformed Data Distribution")
            fig, ax = plt.subplots(figsize=(6, 3))
            data[power_column].hist(ax=ax)
            plt.tight_layout()
            st.pyplot(fig)

# Footer with nicer font
st.markdown("""
    <footer style="bottom:0;width:100%;text-align:center;background-color:#ffffff;padding:10px;">
    <p>Developed by Vignesh Varadharajan</p>
    </footer>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    </style>
    """, unsafe_allow_html=True)

