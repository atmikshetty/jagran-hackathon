import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
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
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import openai
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, fcluster

gemini_api_key = st.secrets["gemini"]["api_key"]
openai_api_key = st.secrets["openai"]["openai_api_key"]

# Configure the genai client
genai.configure(api_key=gemini_api_key)

# configure the openai client
openai.api_key = openai_api_key

def get_theme():
    # Use the new non-experimental API
    query_params = st.query_params
    theme = query_params.get("theme", None)
    
    # If theme is not in query params, check local storage via JavaScript
    if theme is None:
        # Insert JavaScript to check local storage
        st.markdown(
            """
            <script>
                var theme = localStorage.getItem('theme');
                if (theme) {
                    window.parent.postMessage({
                        type: 'streamlit:setQueryParam',
                        queryParams: { theme: theme }
                    }, '*');
                }
            </script>
            """,
            unsafe_allow_html=True
        )
    
    # Default to light if we can't detect the theme
    return theme if theme in ['light', 'dark'] else 'light'

def apply_metric_styles():
    st.markdown(
        """
        <style>
        /* Base styles for metric containers */
        div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 1rem;
            border-radius: 0.5rem;
            width: 100%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Style for metric labels */
        div[data-testid="metric-container"] label {
            color: var(--text-color);
            font-size: 1rem !important;
            font-weight: 600 !important;
        }

        /* Style for metric values */
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
            color: var(--text-color);
            font-weight: 700 !important;
        }

        /* Responsive adjustments */
        @media screen and (max-width: 768px) {
            div[data-testid="metric-container"] label {
                font-size: 0.875rem !important;
            }
            div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
                font-size: 1.5rem !important;
            }
        }

        /* Dark mode specific styles */
        [data-theme="dark"] div[data-testid="metric-container"] {
            background-color: rgba(255, 255, 255, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to display metrics with enhanced styling
def display_metrics(total_posts, most_liked_post, sponsored_percentage):
    apply_metric_styles()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Posts", f"{total_posts:,}")
    
    with col2:
        st.metric("Most Liked Post", f"{most_liked_post:,}")
    
    with col3:
        st.metric("Claims Found in Posts", "0%")
    
    with col4:
        st.metric("Sponsored Posts", f"{sponsored_percentage:.2f}%")

# Define the base color scheme
COLOR_SCHEME = {
    'primary': '#1f77b4',  # Blue
    'secondary': '#ff7f0e',  # Orange
    'accent': '#2ca02c',  # Green
    'neutral': '#7f7f7f',  # Gray
    'background': '#ffffff',  # White
    'text': '#000000',      # Black
    'text_light': '#ffffff'  # White text for dark backgrounds
}

# Define theme-specific color overrides
THEME_COLORS = {
    'light': {
        'background': '#ffffff',  # White
        'text': '#000000',      # Black
        'grid': '#e0e0e0',      # Light gray for grid
        'text_contrast': '#000000'  # Black text for contrast
    },
    'dark': {
        'background': '#0e1117',  # Dark background
        'text': '#ffffff',      # White
        'grid': '#2b2b2b',      # Dark gray for grid
        'text_contrast': '#ffffff'  # White text for contrast
    }
}

def get_current_colors():
    theme = get_theme()
    colors = COLOR_SCHEME.copy()
    colors.update(THEME_COLORS[theme])
    return colors

# Define layout parameters using the base color scheme
PLOT_HEIGHT = 500
PLOT_WIDTH = 800
PLOT_BGCOLOR = COLOR_SCHEME['background']
PLOT_GRIDCOLOR = '#e0e0e0'  # Slightly darker grid for better visibility

# Define common layout settings
COMMON_LAYOUT = {
    'height': PLOT_HEIGHT,
    'width': PLOT_WIDTH,
    'paper_bgcolor': 'rgba(0,0,0,0)',  # Changed to transparent
    'plot_bgcolor': 'rgba(0,0,0,0)',   # Changed to transparent
    'font': {'size': 12, 'color': COLOR_SCHEME['text']},
    'margin': dict(l=50, r=50, t=50, b=50)
}

# Function to get theme-aware plot layout
def get_plot_layout(override_colors=None):
    theme_colors = get_current_colors()
    if override_colors:
        theme_colors.update(override_colors)
    
    return {
        'height': PLOT_HEIGHT,
        'width': PLOT_WIDTH,
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {
            'size': 12,
            'color': theme_colors['text']
        },
        'margin': dict(l=50, r=50, t=50, b=50),
        'xaxis': dict(
            gridcolor=theme_colors['grid'],
            tickfont=dict(color=theme_colors['text']),
            title_font=dict(color=theme_colors['text'])
        ),
        'yaxis': dict(
            gridcolor=theme_colors['grid'],
            tickfont=dict(color=theme_colors['text']),
            title_font=dict(color=theme_colors['text'])
        )
    }

plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
paper_bgcolor='rgba(0,0,0,0)'  # Transparent figure background

# Install and load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Set page title and layout
st.set_page_config(page_title="InfluenceCheck - Misinformation Detection", layout="wide")

# Title and Introduction
st.write(f" ## Welcome to InfluenceCheck")

st.write(f"#####  A platform designed to analyze and verify influencer content in the beauty, fitness, and lifestyle industries. ")
st.write(f"##### We investigate whether the products they promote are genuine, how many of them are flagged as sponsorships, and what their audience thinks of them.")

st.subheader(f"Why InfluenceCheck?")
st.write(f"##### Track Sponsored Content: Identify the percentage of posts that are promotional.")
st.write(f"##### Verify Product Claims: Check if influencers promote genuine products or misinformation.")
st.write(f"##### Sentiment Analysis: Understand audience reactions to influencer recommendations.")
st.write(f"##### Impact on Audience: Explore how influencers shape consumer behavior and opinions.")
st.write(f"#### InfluenceCheck aims to provide **transparency** in influencer marketing and its effect on people's **mindset and choices**.")

@st.cache_data
def load_data():

    file_id = "1s2mwzkFjQai5Lc27r45ecjHVNOtTwCDl"
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = "final_merged_data.csv" 

    try:
        gdown.download(url, output_path, quiet=False)
        df = pd.read_csv(output_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

class TopicMap:
    def __init__(self, df: pd.DataFrame, text_column: str):
        self.df = df
        self.text_column = text_column
        
        self.api_key = st.secrets["openai"]["openai_api_key"]
        openai.api_key = self.api_key

    def preprocess_text(self, text):
        if pd.isna(text):
            return ''
        text = re.sub(r'[^\w\s,]', '', str(text))
        return text

    def perform_clustering(self, texts, method='ward', distance_threshold=2):
        processed_texts = [self.preprocess_text(text) for text in texts]
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            Z = linkage(tfidf_matrix.toarray(), method=method)
            return fcluster(Z, t=distance_threshold, criterion='distance')
        except Exception as e:
            print(f"Clustering error: {e}")
            return []

    def generate_cluster_label(self, cluster_texts):
        prompt = f"""
        You are an expert in clustering and text summarization. Below are text samples from a cluster:
        {cluster_texts}
        
        Based on these samples, suggest a concise and descriptive label for this cluster (2-3 words maximum).
        """
        try:
            client = OpenAI(api_key=st.secrets["openai"]["openai_api_key"]) 

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in clustering and text analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating label: {e}")
            return "Unlabeled Topic"

    def get_topics_for_influencer(self, influencer_name):
        influencer_data = self.df[self.df['influencer_name'] == influencer_name].copy()
        if influencer_data.empty:
            return []

        texts = influencer_data[self.text_column].dropna().tolist()
        if not texts:
            return []

        clusters = self.perform_clustering(texts)
        if not len(clusters):
            return []

        clustered_texts = pd.DataFrame({
            'text': texts,
            'cluster': clusters
        })

        topics = []
        for cluster_id in clustered_texts['cluster'].unique():
            cluster_texts = clustered_texts[clustered_texts['cluster'] == cluster_id]['text'].tolist()
            sample_texts = "\n".join(cluster_texts[:3])
            label = self.generate_cluster_label(sample_texts)
            topics.append(label)

        return list(set(topics))  # Remove duplicates

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

    df.loc[:, "caption_sentiment"] = df["caption"].apply(analyze_sentiment)
    df.loc[:, "is_sponsored"] = df["caption"].apply(detect_promotional_post)
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

def load_influencer_images(influencer_name):

    current_dir = os.path.dirname(os.path.abspath(__file__))

    image_dir = os.path.join(current_dir, "downloaded_images", influencer_name)
    images = []
    
    if os.path.exists(image_dir):
        # Get all jpg files in the influencer's directory
        image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.endswith('.jpg')
        ])[:3]  # Get first 3 images
        
        for image_file in image_files:
            try:
                image_path = os.path.join(image_dir, image_file)
                image = Image.open(image_path)
                images.append(image)
            except Exception as e:
                st.error(f"Error loading image {image_file}: {e}")
    
    return images

# Update the spider plot
def create_spider_plot(emotion_counts):
    theme_colors = get_current_colors()
    
    categories = list(emotion_counts.keys())
    values = list(emotion_counts.values())
    
    fig_spider = go.Figure()
    fig_spider.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor=f'rgba{tuple(int(theme_colors["primary"][1:][i:i+2], 16) for i in (0, 2, 4)) + (0.6,)}',
        line=dict(color=theme_colors["primary"]),
        name="Emotion Distribution"
    ))
    
    layout = get_plot_layout()
    layout.update({
        "polar": dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) + 1],
                gridcolor=theme_colors['grid'],
                linecolor=theme_colors['text'],
                tickfont={'color': theme_colors['text'], 'size': 16},  # Ensuring white labels in dark mode
                title=dict(text="Count", font=dict(size=14, color=theme_colors['text']))
            ),
            angularaxis=dict(
                linecolor=theme_colors['text'],
                gridcolor=theme_colors['grid'],
                tickfont={'color': theme_colors['text'], 'size': 18},  # Ensuring white labels in dark mode
                type='category'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        "showlegend": False
    })
    
    fig_spider.update_layout(**layout)
    return fig_spider


def create_sentiment_pie(sentiment_counts):
    theme_colors = get_current_colors()
    
    fig_sentiment_pie = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts.values,
        title="Sentiment Distribution",
        color_discrete_sequence=[
            theme_colors["primary"],
            theme_colors["secondary"],
            theme_colors["accent"]
        ]
    )
    
    # Explicitly setting text colors for dark mode
    fig_sentiment_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont=dict(color=theme_colors['text_contrast'], size=18),
        insidetextfont=dict(color=theme_colors['text_contrast'], size=20)
    )
    
    # Get and update layout
    layout = get_plot_layout()
    layout.update({
        "showlegend": True,
        "legend": dict(
            font=dict(color=theme_colors['text']),  # Ensuring legend text is white in dark mode
            bgcolor='rgba(0,0,0,0)'
        ),
        "title": dict(
            font=dict(color=theme_colors['text'], size=20)  # Ensuring title is white in dark mode
        )
    })
    
    fig_sentiment_pie.update_layout(**layout)
    
    # Explicitly set the background and remove any template overrides
    fig_sentiment_pie.layout.template = None
    fig_sentiment_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig_sentiment_pie


def create_correlation_heatmap(df_corr):
    theme_colors = get_current_colors()
    
    corr_values = df_corr.to_numpy()
    x_labels = list(df_corr.columns)
    y_labels = list(df_corr.index)
    
    fig_corr = ff.create_annotated_heatmap(
        z=corr_values,
        x=x_labels,
        y=y_labels,
        annotation_text=np.round(corr_values, 2),
        colorscale=[[0, theme_colors["primary"]], [1, theme_colors["secondary"]]],
        showscale=True
    )
    
    layout = get_plot_layout()
    layout.update({
        "width": 500,
        "height": 500,
        "xaxis": dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=theme_colors['text'], size=16),  # Ensuring white labels in dark mode
            title_font=dict(color=theme_colors['text']),
            side="bottom"
        ),
        "yaxis": dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=theme_colors['text'], size=16),  # Ensuring white labels in dark mode
            title_font=dict(color=theme_colors['text'])
        ),
        "coloraxis_colorbar": dict(
            tickfont=dict(color=theme_colors['text'], size=16),
            title_font=dict(color=theme_colors['text'])
        )
    })
    
    fig_corr.update_layout(**layout)

    # Force annotation text to be white in dark mode
    for annotation in fig_corr.layout.annotations:
        annotation.font.color = theme_colors['text_contrast']
        annotation.font.size = 16

    # Explicitly set the background and remove any template overrides
    fig_corr.layout.template = None
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig_corr


# Initialize session state for the dashboard visibility
if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()
df = load_data()
df = compute_sentiment_and_promotion(df)

st.markdown(
    """
    <style>
    div.stButton { 
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add the Explore Dashboard button
if st.button("Explore Dashboard", type="primary"):
    st.session_state.show_dashboard = True

if st.session_state.show_dashboard:

    st.subheader(f"Select an Influencer:")
    influencer_name = st.selectbox("Label", get_influencer_names(), label_visibility="hidden")
    df_filtered = df[df["influencer_name"] == influencer_name].copy()

    # User summary
    st.subheader(f"{influencer_name}'s Bio")

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
    st.write(f"##### {summary}")

    # Recent Images, Only 3
    st.subheader(f"{influencer_name}'s Recent Posts")

    if influencer_name:
        # Display images in a horizontal layout
        images = load_influencer_images(influencer_name)
        if images:
            cols = st.columns(3)
            for idx, (col, image) in enumerate(zip(cols, images)):
                with col:
                    st.image(image, caption=f"Post {idx + 1}", use_container_width=True)
        else:
            st.warning(f" #### No images found for {influencer_name}")

    if df_filtered.empty:
        st.warning("No data available for the selected influencer.")
    else:
       
        total_posts = len(df_filtered)
        most_liked_post = df_filtered["like_count"].max()
        sponsored_count = df_filtered["is_sponsored"].sum()
        sponsored_percentage = (sponsored_count / total_posts) * 100 if total_posts > 0 else 0
    
        st.subheader(f"{influencer_name} - Profile Summary")
        display_metrics(total_posts, most_liked_post, sponsored_percentage)

        # st.subheader(f" {influencer_name} - Profile Summary")

        # # sizes
        # st.markdown(
        #     """
        #     <style>
        #     div[data-testid="metric-container"] {
        #         font-size: 40px !important; /* Increases text size */
        #     }
        #     div[data-testid="stMetricLabel"] {
        #         font-size: 20px !important; /* Increases label size */
        #         font-weight: bold;
        #     }
        #     div[data-testid="stMetricValue"] {
        #         font-size: 28px !important; /* Increases value size */
        #         color: #FFFFFF; /* Optional: Change value color */
        #     }
        #     </style>
        #     """,
        #     unsafe_allow_html=True
        # )

        # # 4x4 Grid
        # col1, col2, col3, col4 = st.columns(4)

        # with col1:
        #     st.metric("Total Posts", total_posts)

        # with col2:
        #     st.metric("Most Liked Post", f"{most_liked_post['like_count']}")
        
        # with col3:
        #     # Fact Checking Details
        #     st.metric("Claims Found in Posts", "0%")

        # with col4:
        #     st.metric("Sponsored Posts", f"{sponsored_percentage:.2f}%")

        # Emotion Analysis - Spider Plot
        st.subheader("What emotions drive audience interaction with influencer content?")
        emotion_counts = {"Happy": 0, "Sad": 0, "Angry": 0, "Surprise": 0, "Fear": 0, "Disgust": 0}

        # for caption in df_filtered["caption"].dropna():
        #     emotions = detect_emotions(caption)
        #     for emotion, count in emotions.items():
        #         emotion_counts[emotion] += count

        # categories = list(emotion_counts.keys())
        # values = list(emotion_counts.values())

        # fig_spider = go.Figure()
        # fig_spider.add_trace(go.Scatterpolar(
        #     r=values + [values[0]],
        #     theta=categories + [categories[0]],
        #     fill='toself',
        #     fillcolor=f'rgba{tuple(int(COLOR_SCHEME["primary"][1:][i:i+2], 16) for i in (0, 2, 4)) + (0.6,)}',
        #     line=dict(color=COLOR_SCHEME["primary"]),
        #     name="Emotion Distribution"
        # ))

        # # Create a copy of COMMON_LAYOUT to avoid duplicate keyword arguments
        # custom_layout = COMMON_LAYOUT.copy()
        # custom_layout.update({
        #     "plot_bgcolor": 'rgba(0,0,0,0)',  # Transparent plot background
        #     "paper_bgcolor": 'rgba(0,0,0,0)',  # Transparent figure background
        #     "polar": dict(
        #         radialaxis=dict(
        #             visible=True,
        #             range=[0, max(values) + 1],
        #             gridcolor=PLOT_GRIDCOLOR,
        #             linecolor=COLOR_SCHEME['text'],
        #             tickfont={'color': "white", 'size': 16}  # Ensure tick labels are white
        #         ),
        #         angularaxis=dict(
        #             linecolor=COLOR_SCHEME['text'],
        #             gridcolor=PLOT_GRIDCOLOR,
        #             tickfont={'color': "white", 'size':18}  # Ensure category labels are white
        #         ),
        #         bgcolor=PLOT_BGCOLOR
        #     ),
        #     "font": dict(color="white"),  # Set all text labels to white
        #     "showlegend": False
        # })

        # fig_spider.update_layout(**custom_layout)
        # st.plotly_chart(fig_spider, use_container_width=True)

        # st.subheader("What emotions drive audience interaction with influencer content?")
        # emotion_counts = {"Happy": 0, "Sad": 0, "Angry": 0, "Surprise": 0, "Fear": 0, "Disgust": 0}
        
        for caption in df_filtered["caption"].dropna():
            emotions = detect_emotions(caption)
            for emotion, count in emotions.items():
                emotion_counts[emotion] += count
        
        fig_spider = create_spider_plot(emotion_counts)
        st.plotly_chart(fig_spider, use_container_width=True)


    # Sentiment Analysis Pie Chart
        st.subheader("Audience Sentiment Analysis: A Breakdown of Reactions to Influencer Content")
        sentiment_counts = df_filtered["caption_sentiment"].value_counts()

        # fig_sentiment_pie = px.pie(
        #     names=sentiment_counts.index,
        #     values=sentiment_counts.values,
        #     title="Sentiment Distribution",
        #     color_discrete_sequence=[
        #         COLOR_SCHEME["primary"], 
        #         COLOR_SCHEME["secondary"], 
        #         COLOR_SCHEME["accent"]
        #     ]
        # )

        # fig_sentiment_pie.update_traces(
        #     textposition='inside',
        #     textinfo='percent+label',
        #     textfont=dict(color=COLOR_SCHEME['text_light'], size=18),
        #     insidetextfont=dict(color=COLOR_SCHEME['text_light'], size=20)
        # )

        # # Create a copy of COMMON_LAYOUT and update it to avoid multiple keyword argument issues
        # custom_layout = COMMON_LAYOUT.copy()
        # custom_layout.update({
        #     "plot_bgcolor": 'rgba(0,0,0,0)',  # Transparent plot background
        #     "paper_bgcolor": 'rgba(0,0,0,0)',  # Transparent figure background
        #     "legend": dict(
        #         bgcolor=COLOR_SCHEME['background'],
        #         bordercolor=COLOR_SCHEME['text'],
        #         borderwidth=1,
        #         font=dict(color=COLOR_SCHEME['text'])
        #     )
        # })

        # fig_sentiment_pie.update_layout(**custom_layout)
        # st.plotly_chart(fig_sentiment_pie, use_container_width=True)

         # For the sentiment pie chart
        # st.subheader("Audience Sentiment Analysis: A Breakdown of Reactions to Influencer Content")
        # sentiment_counts = df_filtered["caption_sentiment"].value_counts()
        fig_sentiment_pie = create_sentiment_pie(sentiment_counts)
        st.plotly_chart(fig_sentiment_pie, use_container_width=True)

        # Correlation Heatmap
        st.subheader("Understanding Influencer Impact via Audience Reactions and Engagement Correlations")
        numeric_cols = ["like_count", "comments_count", "comments_score", "fact_check_rating_comments"]
        df_corr = df_filtered[numeric_cols].corr()

        # Drop all-NaN rows/columns
        df_corr = df_corr.dropna(how="all", axis=0).dropna(how="all", axis=1)

        # # Extract correlation values and labels
        # corr_values = df_corr.to_numpy()
        # x_labels = list(df_corr.columns)
        # y_labels = list(df_corr.index)

        # fig_corr = ff.create_annotated_heatmap(
        #     z=corr_values,
        #     x=x_labels,
        #     y=y_labels,
        #     annotation_text=np.round(corr_values, 2),
        #     colorscale=[[0, COLOR_SCHEME["primary"]], [1, COLOR_SCHEME["secondary"]]],
        #     showscale=True,
        #     font_colors=['white', 'white']
        # )

        # # Force background transparency for heatmap
        # fig_corr.update_layout(
        #     width =500,
        #     height = 500,
        #     paper_bgcolor='rgba(0,0,0,0)',  # Remove white background
        #     plot_bgcolor='rgba(0,0,0,0)',  # Ensure transparency
        #     xaxis=dict(
        #         showgrid=False,
        #         zeroline=False,
        #         tickfont=dict(color="white", size=16),
        #         title_font=dict(color="white"),
        #         side= "bottom"
        #     ),
        #     yaxis=dict(
        #         showgrid=False,
        #         zeroline=False,
        #         tickfont=dict(color="white", size=16),
        #         title_font=dict(color="white")
        #     ),
        #     coloraxis_colorbar=dict(
        #         tickfont=dict(color="white", size=16),
        #         title_font=dict(color="white")
        #     )
        # )

        # # Update each annotation to ensure they blend well
        # for annotation in fig_corr.layout.annotations:
        #     annotation.font.color = "white"
        #     annotation.font.size = 16

        # st.plotly_chart(fig_corr, use_container_width=True)

        # For the correlation heatmap
        # st.subheader("Understanding Influencer Impact via Audience Reactions and Engagement Correlations")
        # numeric_cols = ["like_count", "comments_count", "comments_score", "fact_check_rating_comments"]
        # df_corr = df_filtered[numeric_cols].corr()
        # df_corr = df_corr.dropna(how="all", axis=0).dropna(how="all", axis=1)
        fig_corr = create_correlation_heatmap(df_corr)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Top 10 Topics 
        st.subheader("What's the conversation around influencer content?")
        
        topic_analyzer = TopicMap(df, text_column="caption")

        if influencer_name:
            topics = topic_analyzer.get_topics_for_influencer(influencer_name)
            
            if topics:
                st.write("### Most discussed topics by this influencer:")
                for i, topic in enumerate(topics[:10], 1):
                    st.write(f"{i}. **{topic}**")
            else:
                st.write("No topics could be analyzed for this influencer.")
    