import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import pipeline
from torch.nn.functional import cosine_similarity


# Load precomputed book embeddings
df = pd.read_pickle("data/books_with_embeddings.pkl")

# Load models
@st.cache_resource
def load_models():
    sbert_model = SentenceTransformer("all-mpnet-base-v2")
    theme_extractor = KeyBERT()
    emotion_model = pipeline("text-classification", 
                           model="j-hartmann/emotion-english-distilroberta-base", 
                           top_k=2)
    return sbert_model, theme_extractor, emotion_model

# Function to extract themes
def extract_themes(text):
    if isinstance(text, str):  # Ensure text is valid
        themes = theme_extractor.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=5)
        return [theme[0] for theme in themes]  # Extract only the keywords (ignore scores)
    return []

# Function to extract emotions
def extract_emotions(text):
    emo = []
    emotions = emotion_model(text)
    emo.append(emotions[0][0]["label"])
    emo.append(emotions[0][1]["label"])
    return emo

# Function to convert user input to embedding
def get_user_embedding(user_text):
    themes = extract_themes(user_text)
    emotions = extract_emotions(user_text)
    features = " ".join(themes) + " " + " ".join(emotions)
    return sbert_model.encode(features, convert_to_tensor=True)

# Function to recommend books
def recommend_books(user_embedding, df, top_n=5):
    # Reshape user_embedding to [1, embedding_dim]
    user_embedding = user_embedding.unsqueeze(0)
    
    df["similarity"] = df["embedding"].apply(
        lambda x: cosine_similarity(user_embedding, x.unsqueeze(0)).item()
    )
    return df.sort_values("similarity", ascending=False).head(top_n)

if __name__ == '__main__':
    # Streamlit App
    st.set_page_config(page_title="Book Recommender")
    st.title("Read What You Dream")
    sbert_model, theme_extractor, emotion_model = load_models()
    dream_input = st.text_area("Describe your dream:")
    if st.button("Find Books"):
        if dream_input.strip():
            with st.status("Extracting Features...", expanded=False) as status:
                user_embedding = get_user_embedding(dream_input)
                status.update(label="Feature extraction completed!", state="complete", expanded=False)

            # Get recommendations
            recommendations = recommend_books(user_embedding, df)

            st.subheader("Recommended Books:")
            for _, row in recommendations.iterrows():
                st.write(f"📖  **[{row['title']}]({row['link']})**  \n👤 *{row['author']}*  \n⭐ {row['average_rating']}")

        else:
            st.warning("Please enter a dream description.")