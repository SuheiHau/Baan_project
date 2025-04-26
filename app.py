import pickle
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_txt(text):
    text = text.lower()
    tokens = word_tokenize(text)
    
    y = []
    for word in tokens:
        if word.isalnum() and word not in stopwords.words('english'):
            y.append(ps.stem(word))
    
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
    st.rerun()  # <- This is the correct function now
