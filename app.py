import pickle
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Ensure the necessary NLTK resources are available
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Check and download necessary NLTK resources if they are not found
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def transform_txt(text):
    text = text.lower()
    tokens = word_tokenize(text)
    
    # Filter tokens: keep alphanumeric words that are not stopwords
    y = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    
    return " ".join(y)

# Load the saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit interface
st.set_page_config(page_title="Spam Classifier", layout="centered")
st.title("SMS Spam Classifier")
st.markdown("Enter a message below and click **Predict** to check if it's spam or not.")

# Session state for input text
if 'input_sms' not in st.session_state:
    st.session_state.input_sms = ""

# Text input from the user
st.session_state.input_sms = st.text_area("✏️ Message", value=st.session_state.input_sms, height=150)

# Predict button
if st.button("🔍 Predict"):
    if st.session_state.input_sms.strip() != "":
        transformed_sms = transform_txt(st.session_state.input_sms)
        vector_input = tfidf.transform([transformed_sms]).toarray()
        result = model.predict(vector_input)[0]
        if result == 1:
            st.error("🚫 This message is **SPAM**!")
        else:
            st.success("✅ This message is **NOT SPAM**.")
    else:
        st.warning("Please enter a message before predicting.")

# Clear button placed below Predict
if st.button("🧹 Clear"):
    st.session_state.input_sms = ""
    st.experimental_rerun()  # <- This reloads the app
