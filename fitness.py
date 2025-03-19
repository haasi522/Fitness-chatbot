import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Streamlit Page Configuration
st.set_page_config(page_title="College Chatbot 🎓", page_icon="🎓", layout="centered")

# Store messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load CSV File (Fix: Ensure correct file path or raw GitHub link)
csv_url = "fitness(csvfile).csv"  # Update this to the correct file path

try:
    df = pd.read_csv(csv_url, encoding="ISO-8859-1")  # Fix encoding issue
    df = df.fillna("")  # Fill missing values
except Exception as e:
    st.error(f"⚠️ Failed to load the CSV file. Error: {e}")
    st.stop()

# Ensure required columns exist
if 'Question' not in df.columns or 'Answer' not in df.columns:
    st.error("⚠️ The CSV file must contain 'Question' and 'Answer' columns.")
    st.stop()

# Convert text to lowercase for consistency
df['Question'] = df['Question'].str.lower()
df['Answer'] = df['Answer'].str.lower()

# Vectorize Questions
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df['Question'])

# Secure API Key (Use environment variable instead of hardcoding)
API_KEY = st.secrets["GEMINI_API_KEY"]  # Store in Streamlit secrets

# Configure Google Gemini AI
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to find the best matching question
def find_closest_question(user_query, vectorizer, question_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]

    if best_match_score > 0.3:  # Adjust similarity threshold if needed
        return df.iloc[best_match_index]['Answer']
    return None

# Streamlit UI
st.title("A Fitness Chatbot Agent 🏋️‍♂️")
st.write("💪 Your Virtual Coach for a Healthier You!")

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get closest answer from dataset
    closest_answer = find_closest_question(prompt, vectorizer, question_vectors, df)

    if closest_answer:
        st.session_state.messages.append({"role": "assistant", "content": closest_answer})
        with st.chat_message("assistant"):
            st.markdown(closest_answer)
    else:
        # If no close match, use Gemini AI
        try:
            response = model.generate_content(prompt)
            ai_response = response.candidates[0].content if response.candidates else "I couldn't find an answer."

            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.markdown(ai_response)
        except Exception as e:
            st.error(f"⚠️ Sorry, I couldn't generate a response. Error: {e}")
