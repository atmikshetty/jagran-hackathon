import openai
import os
import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, fcluster

class TopicMap:
    def __init__(self, df: pd.DataFrame, text_column: str):
        self.df = df
        self.text_column = text_column
        
        self.api_key = st.secrets["openai_api_key"]
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
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in clustering and text analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.7
            )
            return response['choices'][0]['message']['content'].strip()
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

st.subheader("🎯 Top 10 Topics")

topic_analyzer = TopicMap(df, text_column="caption")

if influencer_name:
    topics = topic_analyzer.get_topics_for_influencer(influencer_name)
    
    if topics:
        st.write("Most discussed topics by this influencer:")
        for i, topic in enumerate(topics[:10], 1):
            st.write(f"{i}. **{topic}**")
    else:
        st.write("No topics could be analyzed for this influencer.")