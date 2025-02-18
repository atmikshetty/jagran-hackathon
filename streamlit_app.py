import subprocess

# Install missing system dependencies
#subprocess.run(["apt-get", "update"])
#subprocess.run(["apt-get", "install", "-y", "libfontconfig1", "libglib2.0-0"])  # Needed for wordcloud

# Install missing Python dependencies
subprocess.run(["pip", "install", "spacy", "wordcloud", "textblob", "pandas", "plotly", "streamlit"])

# Download spaCy model
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
from textblob import TextBlob
import spacy
import re
import subprocess
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

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
    def analyze_sentiment(text):
        polarity = TextBlob(str(text)).sentiment.polarity  
        if polarity > 0:
            return "Positive"
        elif polarity < 0:
            return "Negative"
        else:
            return "Neutral"

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

def detect_emotions(text):
    emotion_dict = {
        "Happy": 0,
        "Sad": 0,
        "Angry": 0,
        "Surprise": 0,
        "Fear": 0,
        "Disgust": 0
    }
    
    # Analyze the text for emotions using VADER sentiment intensity
    sentiment_score = analyzer.polarity_scores(str(text))
    
    # Classification based on sentiment scores
    if sentiment_score["compound"] >= 0.05:
        emotion_dict["Happy"] += 1
    elif sentiment_score["compound"] <= -0.05:
        emotion_dict["Sad"] += 1
    if sentiment_score["neg"] >= 0.5:
        emotion_dict["Angry"] += 1
    if sentiment_score["neu"] >= 0.5:
        emotion_dict["Fear"] += 1
    if sentiment_score["pos"] >= 0.5:
        emotion_dict["Surprise"] += 1
    
    return emotion_dict

df = compute_sentiment_and_promotion(df)

# Streamlit UI
st.title("📢 InfluenceCheck - Misinformation Detection & Fact-Checking Dashboard")

# Single Tab for everything
st.header("📊 Influencer Analysis")
influencer_name = st.selectbox("Select an Influencer", get_influencer_names())

df_filtered = df[df["influencer_name"] == influencer_name].copy()

if df_filtered.empty:
    st.warning("No data available for the selected influencer.")
else:
    total_posts = len(df_filtered)
    most_liked_post = df_filtered.loc[df_filtered["like_count"].idxmax()]
    
    st.write(f"### 📌 {influencer_name} - Summary")
    st.metric("Total Posts", total_posts)
    st.metric("Most Liked Post", f"{most_liked_post['like_count']} likes")

    # Emotion Analysis - Spider Plot
    st.subheader("📊 Emotion Analysis (Spider Plot)")
    emotion_counts = {"Happy": 0, "Sad": 0, "Angry": 0, "Surprise": 0, "Fear": 0, "Disgust": 0}

    # Detect emotions for each caption
    for caption in df_filtered["caption"].dropna():
        emotions = detect_emotions(caption)
        for emotion, count in emotions.items():
            emotion_counts[emotion] += count

    categories = list(emotion_counts.keys())
    values = list(emotion_counts.values())

    fig_spider = go.Figure()
    fig_spider.add_trace(go.Scatterpolar(
        r=values + [values[0]],  
        theta=categories + [categories[0]],  
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.6)',  
        line=dict(color='darkblue'),
        name="Emotion Distribution"
    ))

    fig_spider.update_layout(
        polar=dict(
            bgcolor='black',  
            radialaxis=dict(
                visible=True,
                range=[0, max(values) + 1],
                gridcolor='rgba(50, 50, 255, 0.4)',
                gridwidth=2,
            ),
            angularaxis=dict(
                tickfont=dict(color='white'),
            ),
        ),
        showlegend=False,
        paper_bgcolor="black",
        font=dict(color="white")
    )

    st.plotly_chart(fig_spider, use_container_width=True)

    # Sentiment Analysis - Spider Plot
    st.subheader("📊 Sentiment Analysis (Spider Plot)")
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

    # Detect sentiment for each caption
    for caption in df_filtered["caption"].dropna():
        sentiment = TextBlob(str(caption)).sentiment.polarity
        if sentiment > 0:
            sentiment_counts["Positive"] += 1
        elif sentiment < 0:
            sentiment_counts["Negative"] += 1
        else:
            sentiment_counts["Neutral"] += 1

    sentiment_categories = list(sentiment_counts.keys())
    sentiment_values = list(sentiment_counts.values())

    fig_sentiment_spider = go.Figure()
    fig_sentiment_spider.add_trace(go.Scatterpolar(
        r=sentiment_values + [sentiment_values[0]],  
        theta=sentiment_categories + [sentiment_categories[0]],  
        fill='toself',
        fillcolor='rgba(255, 99, 132, 0.6)',  
        line=dict(color='darkred'),
        name="Sentiment Distribution"
    ))

    fig_sentiment_spider.update_layout(
        polar=dict(
            bgcolor='black',  
            radialaxis=dict(
                visible=True,
                range=[0, max(sentiment_values) + 1],
                gridcolor='rgba(50, 50, 255, 0.4)',
                gridwidth=2,
            ),
            angularaxis=dict(
                tickfont=dict(color='white'),
            ),
        ),
        showlegend=False,
        paper_bgcolor="black",
        font=dict(color="white")
    )

    st.plotly_chart(fig_sentiment_spider, use_container_width=True)

    # Word Cloud
    st.subheader("🌍 Theme Analysis")
    text = " ".join(df_filtered["caption"].dropna())
    if text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wordcloud.to_array(), use_container_width=True)
    else:
        st.write("No text available for word cloud.")

    # Engagement Insights - Scatter Plot
    st.subheader("📈 Likes vs Comments")
    fig_engagement = px.scatter(df_filtered, x="like_count", y="comments_count", title="Likes vs Comments", color_discrete_sequence=["purple"])
    st.plotly_chart(fig_engagement, use_container_width=True)

    # Sponsored Posts - Pie Chart
    st.subheader("📢 Sponsored Posts Analysis")
    promo_counts = df_filtered["is_sponsored"].value_counts().rename(index={0: "Non-Sponsored", 1: "Sponsored"})
    fig_promo = px.pie(promo_counts, names=promo_counts.index, values=promo_counts.values, title="Sponsored vs Non-Sponsored Posts")
    st.plotly_chart(fig_promo, use_container_width=True)

# Trending Posts Bar Plot
st.subheader("🔥 Most Trending Posts (Top Likes)")
most_liked_posts = df.loc[df.groupby("influencer_name")["like_count"].idxmax()]
fig_trending = px.bar(most_liked_posts, x="influencer_name", y="like_count", color="influencer_name", title="Most Trending Posts by Influencer", color_discrete_sequence=["skyblue"])
st.plotly_chart(fig_trending, use_container_width=True)
