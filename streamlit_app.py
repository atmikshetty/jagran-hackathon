import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from textblob import TextBlob
import spacy
import re
import subprocess

# added this 
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is missing, install it
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


@st.cache_data
def load_data():
    df_main = pd.read_csv("datasets/influencer_data_final.csv")
    df_comments = pd.read_csv("datasets/10_influencers_comments_data.csv")

    if "id" not in df_main.columns or "post_id" not in df_comments.columns:
        st.error("🚨 Error: 'id' or 'post_id' column missing in one of the datasets!")
        return df_main  

    df_comments = df_comments.rename(columns={"text": "comment_text"})
    df_merged = df_main.merge(df_comments, left_on="id", right_on="post_id", how="left")

    return df_merged

df = load_data()

# Extract influencer names
@st.cache_data
def get_influencer_names():
    return list(df["influencer_name"].unique())

# Sentiment analysis function
def analyze_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity  

df["caption_sentiment"] = df["caption"].apply(analyze_sentiment)

# To detect sponsorship posts
SPONSORED_HASHTAGS = {"#ad", "#sponsored", "#promotion", "#brandpartner", "#collab", "#gifted", "#prpackage", "#promocode", "#partnership"}

# To detect promotional phrases
PROMOTIONAL_PHRASES = {
    "Use code", "Limited offer", "Partnered with", "Check out", "Special discount",
    "Exclusive deal", "Click the link", "Promo ends soon", "Collab with"
}

# Function to detect brand mentions using Named Entity Recognition (NER)
def contains_brand_name(text):
    doc = nlp(str(text))  
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "GPE"]:  
            return 1  
    return 0  

# Function to detect promotional posts
def detect_promotional_post(caption):
    if isinstance(caption, str):  
        words = caption.lower().split()

        # Check for sponsored posts
        if any(tag in words for tag in SPONSORED_HASHTAGS):
            return 1  

        # Check for promotional phrases
        if any(phrase.lower() in caption.lower() for phrase in PROMOTIONAL_PHRASES):
            return 1  

        # Check for links 
        if re.search(r'https?://\S+', caption):
            return 1  

        # Check if brand names are mentioned
        if contains_brand_name(caption):
            return 1  

    return 0  

# Apply sponsorship detection
df["is_sponsored"] = df["caption"].apply(detect_promotional_post)


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

    # Dropdown to select an influencer
    influencer_name = st.selectbox("Select an Influencer", get_influencer_names())
    df_filtered = df[df["influencer_name"] == influencer_name]

    # Summary statistics
    total_posts = len(df_filtered)
    most_liked_post = df_filtered.loc[df_filtered["like_count"].idxmax()]
    post_link = most_liked_post["post_url"]
    post_image = most_liked_post["thumbnail_url"]

    st.write(f"### 📌 {influencer_name} - Summary")
    st.metric("Total Posts", total_posts)
    st.metric("Most Liked Post", f"{most_liked_post['like_count']} likes")

    # Sentiment Analysis Visualization
    st.subheader("📄 Caption Sentiment")
    fig_sentiment = px.histogram(df_filtered, x="caption_sentiment", nbins=20, title="Caption Sentiment Distribution", color_discrete_sequence=["blue"])
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # Theme Analysis (Word Cloud)
    st.subheader("🌍 Theme Analysis")
    text = " ".join(df_filtered["caption"].dropna())

    if text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wordcloud.to_array(), use_container_width=True)
    else:
        st.write("No text available for word cloud.")

    # Engagement Insights (Likes vs Comments)
    st.subheader("📈 Likes vs Comments")
    fig_engagement = px.scatter(df_filtered, x="like_count", y="comments_count", title="Likes vs Comments", color_discrete_sequence=["purple"])
    st.plotly_chart(fig_engagement, use_container_width=True)

    # Promotional Posts Analysis
    st.subheader("📢 Sponsored Posts Analysis")
    promo_counts = df_filtered["is_sponsored"].value_counts().rename(index={0: "Non-Sponsored", 1: "Sponsored"})
    fig_promo = px.pie(promo_counts, names=promo_counts.index, values=promo_counts.values, title="Sponsored vs Non-Sponsored Posts")
    st.plotly_chart(fig_promo, use_container_width=True)

st.write("🔥 Powered by Streamlit 🚀")
