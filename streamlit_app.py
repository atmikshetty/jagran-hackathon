import subprocess

# Install missing system dependencies
subprocess.run(["apt-get", "update"])
subprocess.run(["apt-get", "install", "-y", "libfontconfig1", "libglib2.0-0"])  # Needed for wordcloud

# Install missing Python dependencies
subprocess.run(["pip", "install", "spacy", "wordcloud", "textblob", "pandas", "plotly", "streamlit"])

# Download spaCy model
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])


import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from textblob import TextBlob
import spacy
import re
import subprocess

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

@st.cache_data
def load_data():
    df_main = pd.read_csv("datasets/influencer_data_final.csv")
    df_comments = pd.read_csv("datasets/10_influencers_comments_data.csv")

    if "id" not in df_main.columns or "post_id" not in df_comments.columns:
        st.error("🚨 Error: Missing necessary columns in datasets!")
        return df_main  

    df_comments.rename(columns={"text": "comment_text"}, inplace=True)
    df_merged = df_main.merge(df_comments, left_on="id", right_on="post_id", how="left")

    return df_merged

df = load_data()

@st.cache_data
def get_influencer_names():
    return list(df["influencer_name"].unique())

@st.cache_data
def compute_sentiment_and_promotion(df):
    """
    Cache sentiment analysis & promotional post detection
    to avoid recomputing on every dropdown change.
    """
    def analyze_sentiment(text):
        return TextBlob(str(text)).sentiment.polarity  

    def contains_brand_name(text):
        doc = nlp(str(text))  
        return any(ent.label_ in ["ORG", "PRODUCT", "GPE"] for ent in doc.ents)

    def detect_promotional_post(caption):
        if isinstance(caption, str):  
            words = caption.lower().split()
            sponsored_tags = {"#ad", "#sponsored", "#promotion", "#brandpartner", "#collab", "#gifted", "#prpackage", "#promocode", "#partnership"}
            promotional_phrases = {
                "Use code", "Limited offer", "Partnered with", "Check out", "Special discount",
                "Exclusive deal", "Click the link", "Promo ends soon", "Collab with"
            }

            return any(tag in words for tag in sponsored_tags) or \
                   any(phrase.lower() in caption.lower() for phrase in promotional_phrases) or \
                   re.search(r'https?://\S+', caption) or contains_brand_name(caption)

        return 0  

    df["caption_sentiment"] = df["caption"].apply(analyze_sentiment)
    df["is_sponsored"] = df["caption"].apply(detect_promotional_post)

    return df

df = compute_sentiment_and_promotion(df)

# Streamlit UI
st.title("📢 InfluenceCheck - Misinformation Detection & Fact-Checking Dashboard")
tab1, tab2 = st.tabs(["🏠 Home", "📊 Influencer Analysis"])

with tab1:
    st.header("🚀 Why InfluenceCheck?")
    st.write("""
    - **Track Sponsored Content:** Identify promotional posts.
    - **Verify Product Claims:** Detect misinformation.
    - **Sentiment Analysis:** Understand audience reactions.
    - **Impact on Audience:** See how influencers shape consumer behavior.
    """)

with tab2:
    st.header("📊 Influencer Analysis")

    influencer_name = st.selectbox("Select an Influencer", get_influencer_names())
    
    # **Filter only the required influencer (efficiently)**
    df_filtered = df[df["influencer_name"] == influencer_name].copy()

    if df_filtered.empty:
        st.warning("No data available for the selected influencer.")
    else:
        total_posts = len(df_filtered)
        most_liked_post = df_filtered.loc[df_filtered["like_count"].idxmax()]
        
        st.write(f"### 📌 {influencer_name} - Summary")
        st.metric("Total Posts", total_posts)
        st.metric("Most Liked Post", f"{most_liked_post['like_count']} likes")

        # **Sentiment Analysis**
        st.subheader("📄 Caption Sentiment")
        fig_sentiment = px.histogram(df_filtered, x="caption_sentiment", nbins=20, title="Caption Sentiment Distribution", color_discrete_sequence=["blue"])
        st.plotly_chart(fig_sentiment, use_container_width=True)

        # **Word Cloud**
        st.subheader("🌍 Theme Analysis")
        text = " ".join(df_filtered["caption"].dropna())

        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
            st.image(wordcloud.to_array(), use_container_width=True)
        else:
            st.write("No text available for word cloud.")

        # **Engagement Insights**
        st.subheader("📈 Likes vs Comments")
        fig_engagement = px.scatter(df_filtered, x="like_count", y="comments_count", title="Likes vs Comments", color_discrete_sequence=["purple"])
        st.plotly_chart(fig_engagement, use_container_width=True)

        # **Sponsored Posts**
        st.subheader("📢 Sponsored Posts Analysis")
        promo_counts = df_filtered["is_sponsored"].value_counts().rename(index={0: "Non-Sponsored", 1: "Sponsored"})
        fig_promo = px.pie(promo_counts, names=promo_counts.index, values=promo_counts.values, title="Sponsored vs Non-Sponsored Posts")
        st.plotly_chart(fig_promo, use_container_width=True)

st.write("🔥 Powered by Streamlit 🚀")
