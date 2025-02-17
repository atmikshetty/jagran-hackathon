import subprocess
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
from textblob import TextBlob
import spacy
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gdown
import requests
from io import BytesIO
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyDfPoNzsJJ1kvNh88ape_36KEfgcoRPSkU"))

# Install and load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

@st.cache_data
def load_data():
    # # old
    # df_main = pd.read_csv("datasets/influencer_data_final.csv")
    # df_comments = pd.read_csv("datasets/10_influencers_comments_data.csv")

    # if "id" not in df_main.columns or "post_id" not in df_comments.columns:
    #     st.error("🚨 Error: Missing necessary columns in datasets!")
    #     return df_main  

    # df_comments.rename(columns={"text": "comment_text"}, inplace=True)
    # df_merged = df_main.merge(df_comments, left_on="id", right_on="post_id", how="left")

    # return df_merged

    file_id = "1s2mwzkFjQai5Lc27r45ecjHVNOtTwCDl"
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = "final_merged_data.csv" 

    try:
        gdown.download(url, output_path, quiet=False)
        df = pd.read_csv(output_path)
        return df
    except Exception as e:
        st.error(f"🚨 Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

@st.cache_data
def get_influencer_names():
    return list(df["influencer_name"].unique())

@st.cache_data
def compute_sentiment_and_promotion(df):
    def analyze_sentiment(text):
        polarity = TextBlob(str(text)).sentiment.polarity  
        return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

    def contains_brand_name(text):
        doc = nlp(str(text))  
        return any(ent.label_ in ["ORG", "PRODUCT", "GPE"] for ent in doc.ents)

    def detect_promotional_post(caption):
        if isinstance(caption, str):  
            words = caption.lower().split()
            sponsored_tags = {"#ad", "#sponsored", "#promotion", "#brandpartner", "#collab", "#gifted", "#prpackage", "#promocode", "#partnership"}
            promotional_phrases = {"Use code", "Limited offer", "Partnered with", "Check out", "Special discount",
                                   "Exclusive deal", "Click the link", "Promo ends soon", "Collab with"}

            return any(tag in words for tag in sponsored_tags) or \
                   any(phrase.lower() in caption.lower() for phrase in promotional_phrases) or \
                   re.search(r'https?://\S+', caption) or contains_brand_name(caption)
        return 0  

    df["caption_sentiment"] = df["caption"].apply(analyze_sentiment)
    df["is_sponsored"] = df["caption"].apply(detect_promotional_post)
    return df

def detect_emotions(text):
    emotion_dict = {"Happy": 0, "Sad": 0, "Angry": 0, "Surprise": 0, "Fear": 0, "Disgust": 0}
    sentiment_score = analyzer.polarity_scores(str(text))
    
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

def generate_summary(text_data):
    """Summarizes influencer content using Gemini"""
    if not isinstance(text_data, str) or not text_data.strip():
        return "No summary available."

    prompt = f"Summarize the following influencer's social media content in 2-3 sentences:\n{text_data[:5000]}"

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Summary generation failed."

df = compute_sentiment_and_promotion(df)

# Streamlit UI
st.title("📢 InfluenceCheck - Misinformation Detection & Fact-Checking Dashboard")
st.header("📊 Influencer Analysis")

influencer_name = st.selectbox("Select an Influencer", get_influencer_names())
df_filtered = df[df["influencer_name"] == influencer_name].copy()

# User summary
st.subheader(f"✨ {influencer_name}'s Bio")

# top 10 posts
df_captions = (
    df_filtered[['text']]
    .dropna()           
    .drop_duplicates()  
    .head(10)          
)

captions_list = df_captions['text'].tolist()
captions_text = "\n".join(captions_list)

# generate summary
summary = generate_summary(captions_text)
st.write(summary)

st.subheader(f"📸 {influencer_name}'s Recent Posts")

# 3 images only
df_thumbnails = (
    df_filtered[['thumbnail_url']]
    .dropna()                
    .drop_duplicates()       
    .head(3)                
)

if df_thumbnails.empty:
    st.warning("No images available for this influencer.")
else:
    # 3 column layout
    cols = st.columns(3)
    for index, (_, row) in enumerate(df_thumbnails.iterrows()):
        thumbnail_url = row["thumbnail_url"]
        
        # request to the url
        try:
            response = requests.get(thumbnail_url, timeout=10)
            if response.status_code == 200:
                # something from chatgpt
                image_bytes = BytesIO(response.content)
                with cols[index % 3]:
                    st.image(
                        image_bytes,
                        caption=f"Post {index+1}",
                        use_container_width=True
                    )
            else:
                # exceptions
                with cols[index % 3]:
                    st.warning(f"Failed to fetch image (status code: {response.status_code})")
        except Exception as e:
            with cols[index % 3]:
                st.error(f"Error fetching image: {e}")

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
            radialaxis=dict(visible=True, range=[0, max(values) + 1]),
        ),
        showlegend=False
    )
    st.plotly_chart(fig_spider, use_container_width=True)

    # Sentiment Analysis Pie Chart
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df_filtered["caption_sentiment"].value_counts()
    fig_sentiment_pie = px.pie(
        names=sentiment_counts.index, 
        values=sentiment_counts.values, 
        title="Sentiment Distribution"
    )
    st.plotly_chart(fig_sentiment_pie, use_container_width=True)

    # Word Cloud
    st.subheader("🌍 Theme Analysis")
    text = " ".join(df_filtered["caption"].dropna())
    if text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wordcloud.to_array(), use_container_width=True)
    else:
        st.write("No text available for word cloud.")
    
    # Fact Checking
    # Bar Plot for "No claims found" responses
    st.subheader("📊 Fact-Checked Claims Distribution")
    claims_counts = df_filtered['fact_checked_claim_comments'].value_counts()
    fig_claims = px.bar(
        x=claims_counts.index, 
        y=claims_counts.values, 
        title="Fact-Checked Claims Distribution",
        labels={'x': 'Claim Status', 'y': 'Count'}
    )
    st.plotly_chart(fig_claims, use_container_width=True)
