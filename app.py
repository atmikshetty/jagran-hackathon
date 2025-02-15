import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# load data
@st.cache_data
def load_data():
    return pd.read_csv("datasets\influencer_data_final.csv")  # final file with influencers name as well 

df = load_data()

# Extract influencer names dynamically
@st.cache_data
def get_influencer_names():
    return list(df["influencer_name"].unique())  # Ensure influencer_name column is present

# Streamlit UI
st.title("📊 Influencer Insights Dashboard")

# Influencer Selection Dropdown
influencer_name = st.selectbox("Select an Influencer", get_influencer_names())

# Filter Data for Selected Influencer
df_filtered = df[df["influencer_name"] == influencer_name]

st.write(f"### Insights for {influencer_name}")

# 📝 Sentiment Analysis Visualization
st.subheader("📝 Caption & Comment Sentiment")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# sns.histplot(df_filtered["caption_score"], bins=10, kde=True, ax=ax[0], color="blue")
# ax[0].set_title("Caption Sentiment Distribution")
# ax[0].set_xlabel("Sentiment Score")

sns.histplot(df_filtered["comments_score"], bins=10, kde=True, ax=ax[1], color="red")
ax[1].set_title("Comment Sentiment Distribution")
ax[1].set_xlabel("Sentiment Score")

st.pyplot(fig)

# # 🔍 Fact-Check Results
# st.subheader("🔍 Fact-Check Results")

# fact_counts = df_filtered["fact_check_rating"].value_counts()
# fig, ax = plt.subplots(figsize=(6, 4))
# fact_counts.plot(kind="bar", color=["green", "red", "orange"], ax=ax)
# ax.set_title("Fact-Check Summary")
# ax.set_ylabel("Number of Captions")
# ax.set_xticklabels(fact_counts.index, rotation=45)

# st.pyplot(fig)

# 🌍 Theme Analysis (Word Cloud)
st.subheader("🌍 Theme Analysis")

text = " ".join(df_filtered["text"].dropna())
if text.strip():
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.write("No text available for word cloud.")

# 📈 Engagement Insights (Likes vs Comments)
st.subheader("📈 Likes vs Comments")

fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x=df_filtered["like_count"], y=df_filtered["comments_count"], ax=ax, color="purple")
ax.set_xlabel("Likes")
ax.set_ylabel("Comments")
ax.set_title("Likes vs Comments")
st.pyplot(fig)

st.write("🔥 Powered by Streamlit 🚀")

