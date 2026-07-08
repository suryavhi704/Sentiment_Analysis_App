import pickle
import torch
import torch.nn as nn
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import streamlit as st

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Load vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    tf = pickle.load(f)

# Define RNN model class
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Load model
model = RNN(input_size=5000)
model.load_state_dict(torch.load("sentiment_model.pth"))
model.eval()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

# Prediction function
def predict_sentiment(text):
    text = preprocess_text(text)
    vector = tf.transform([text]).toarray()
    tensor = torch.from_numpy(vector).float()

    with torch.no_grad():
        tensor = tensor.unsqueeze(1)
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    if prob > 0.5:
        return True, prob
    else:
        return False, prob

# Streamlit UI

# ---------------- Streamlit UI ---------------- #
# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #0f1116;
        color: #f5f5f5;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextArea textarea {
        background-color: #1c1e24;
        color: #f5f5f5;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #0078ff;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar pipeline steps
st.sidebar.title("⚙️ Execution Pipeline")
st.sidebar.markdown("""
1. 📥 **Data Collection**  
2. 🧹 **Data Cleaning**  
3. ❌ **Remove Stopwords**  
4. 🔄 **Stemming & Lemmatization**  
5. 🧮 **TF-IDF Vectorization**  
6. 🤖 **Model Training & Testing**  
7. 💻 **Local Deployment**
""")

# Main UI
st.markdown("""
    <style>
    .stApp h1 {
        font-weight: 900 !important;
        color: #ffffff !important;   /* pure white for strong contrast */
    }
    .stApp h3 {
        font-weight: 800 !important;
        color: #e0e0e0 !important;   /* slightly softer white */
    }
    </style>
""", unsafe_allow_html=True)

# Your headings
st.title("🎬 IMDB Sentiment Analyzer")
st.subheader("🚀 AI-powered Sentiment Analysis APP")
st.write("Enter a review and get sentiment prediction using **RNN + TF-IDF**")

user_input = st.text_area("💬 Enter your sentence here...", height=150)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        result, prob = predict_sentiment(user_input)

        # Confidence gauge
        st.progress(int(prob * 100))

        if result:
            st.success(f"✅ Positive 😊 (Confidence: {prob:.2f})")
        else:
            st.error(f"❌ Negative 😞 (Confidence: {1-prob:.2f})")

        st.info("Confidence gauge above shows model certainty.")
    else:
        st.warning("⚠️ Please enter some text before analyzing.")



