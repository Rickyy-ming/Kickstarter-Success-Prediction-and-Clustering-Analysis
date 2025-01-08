#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:37:45 2024

@author: yuanminglu
"""

####################################################  Task 2  #####################################################

# Import necessary libraries for data processing, clustering, and evaluation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Load the Kickstarter dataset
df_2 = pd.read_excel("Kickstarter.xlsx")

# Filter for only 'successful' or 'failed' states (ignores other outcomes)
df_2 = df_2[df_2['state'].isin(['successful', 'failed'])]

# Display all columns in the console for better inspection
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Handle missing values by dropping rows with any null values
df_2 = df_2.dropna()

# Drop variables unavailable at the launch date to focus on predictive features
df_2 = df_2.drop(columns=['id'])

# Group countries into 'US', 'GB', and 'Non-US' for better analysis
df_2['country'].value_counts()
df_2['country'] = df_2['country'].apply(lambda x: x if x in ['US', 'GB','CA','MX','DE','AU','FR'] else 'Non-US')

# Group less common categories into 'Other' to reduce dimensionality
df_2['category'].value_counts()
top_categories = df_2['category'].value_counts().nlargest(50).index.tolist()
# Convert categories not in the top list to 'Other'
df_2['category'] = df_2['category'].apply(lambda x: x if x in top_categories else 'Other')

# One-hot encode weekday-related features to convert them into numeric form
categorical_features = ['deadline_weekday', 'created_at_weekday', 'launched_at_weekday', 'state_changed_at_weekday','category']
df_2 = pd.get_dummies(df_2, columns=categorical_features, drop_first=True)

# Group main categories into broader categories for simplification
category_mapping = {
    'Film & Video': 'Media & Entertainment',
    'Fashion': 'Lifestyle',
    'Music': 'Media & Entertainment',
    'Journalism': 'Media & Entertainment',
    'Technology': 'Science & Technology',
    'Design': 'Art & Design',
    'Publishing': 'Art & Design',
    'Theater': 'Media & Entertainment',
    'Games': 'Media & Entertainment',
    'Food': 'Lifestyle',
    'Art': 'Art & Design',
    'Crafts': 'Art & Design',
    'Comics': 'Art & Design',
    'Photography': 'Art & Design',
    'Dance': 'Media & Entertainment'
}

# Map the broader categories and drop the original column
df_2['main_category_grouped'] = df_2['main_category'].map(category_mapping)
df_2 = df_2.drop(columns=['main_category'])

# Convert currency into binary encoding for USD and other major currencies
df_2['currency'] = df_2['currency'].apply(lambda x: x if x in ['USD','GBP','EUR','CAD'] else 'Else')

# Convert binary variables to integers for processing
binary_variables = ['disable_communication', 'staff_pick', 'staff_pick.1', 'show_feature_image', 'video']
df_2[binary_variables] = df_2[binary_variables].astype(int)
df_2['state'] = df_2['state'].apply(lambda x: 1 if x == 'successful' else 0)  # Encode target as binary
# Apply lambda function to each element of the selected columns
df_2 = pd.get_dummies(df_2, columns=['main_category_grouped','country','currency'], drop_first=True)
 
# Preprocess text columns for sentiment analysis using NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function to remove stopwords
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(str(text))  # Tokenize text
    filtered_words = [word for word in words if word.lower() not in stop_words]  # Remove stopwords
    return ' '.join(filtered_words)  # Return cleaned text

# Function to calculate sentiment score using VADER
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)  # Get sentiment scores
    return sentiment_scores['compound']  # Return compound sentiment score

# Apply sentiment analysis on the 'name' column
df_2['Title Sentiment'] = df_2['name'].apply(lambda x: analyze_sentiment(preprocess_text(x)))

### Date-Based Features ###
# Calculate campaign duration in days
df_2['campaign_duration'] = (pd.to_datetime(df_2['deadline']) - pd.to_datetime(df_2['launched_at'])).dt.days

# Categorize the launch season based on month
df_2['launch_season'] = pd.cut(df_2['launched_at_month'], bins=[0, 3, 6, 9, 12], labels=['1', '2', '3', '4'])

# Drop unnecessary date-related columns
df_2 = df_2.drop(columns=['deadline', 'created_at', 'launched_at', 'name', 'state_changed_at'])

### Goal and Pledge-Related Features ###
# Convert goal and pledged amounts to USD using a static conversion rate
df_2['goal_usd'] = df_2['goal'] * df_2['static_usd_rate']
df_2['pledged_usd'] = df_2['pledged'] * df_2['static_usd_rate']

df_2['state'].value_counts()
# Standardize the features for clustering
scaler = StandardScaler()
X_std = scaler.fit_transform(df_2) 
scaled_X = pd.DataFrame(X_std, columns=df_2.columns)

# Dimensionality reduction with PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization and to alleviate sparsity
X_pca = pca.fit_transform(scaled_X)

# Calculate the correlation matrix to identify highly correlated features
correlation_matrix = scaled_X.corr()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Create a mask to extract highly correlated features (threshold: 0.6)
correlated_features = [
    column
    for column in upper_triangle.columns
    if any(upper_triangle[column].abs() > 0.6)
]

# Filter out features with high correlations
correlated_features = [
    feature for feature in scaled_X.columns if feature not in correlated_features
]

# Subset the DataFrame with the remaining features
reduced_X = scaled_X[correlated_features]
print(reduced_X.columns)
# Define a refined set of features for clustering
selected_features = ['pledged_usd', 'main_category_encoded', 'country', 'goal_usd', 'backers_count', 'staff_pick']
X_clustering = scaled_X[selected_features]

# Perform K-Means clustering with varying numbers of clusters
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

k_range = range(2, 20)
inertia_scores = []
silhouette_scores = []
f_statistic_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_clustering)
    inertia_scores.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(X_clustering, labels)
    silhouette_scores.append(silhouette_avg)
    f_statistic = calinski_harabasz_score(X_clustering, labels)
    f_statistic_scores.append(f_statistic)

# Display evaluation metrics
print("Silhouette Scores:", silhouette_scores)
print("f_statistic Scores:", f_statistic_scores)
print("Inertia Scores:", inertia_scores)
  
# Perform final K-Means clustering with the chosen number of clusters (e.g., 2)
kmeans = KMeans(n_clusters=2, random_state=42)
model = kmeans.fit(X_clustering)
labels = model.labels_

# Evaluate the final clustering
silhouette_avg = silhouette_score(X_clustering, labels)
print("Silhouette Score for final clustering:", silhouette_avg)

from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Hierarchical Clustering
hierarchical_clustering = AgglomerativeClustering(n_clusters=2)  # Adjust the number of clusters as needed
hierarchical_labels = hierarchical_clustering.fit_predict(X_clustering)

# Evaluate Hierarchical Clustering
silhouette_hierarchical = silhouette_score(X_clustering, hierarchical_labels)
f_stat_hierarchical = calinski_harabasz_score(X_clustering, hierarchical_labels)

print("Hierarchical Clustering Results:")
print(f"Silhouette Score: {silhouette_hierarchical}")
print(f"F-statistic: {f_stat_hierarchical}")

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as needed
dbscan_labels = dbscan.fit_predict(X_clustering)

# Filter out noise points (-1 label) for evaluation
dbscan_core_labels = dbscan_labels[dbscan_labels != -1]
if len(set(dbscan_core_labels)) > 1:  # Ensure there are at least 2 clusters
    silhouette_dbscan = silhouette_score(X_clustering[dbscan_labels != -1], dbscan_core_labels)
    f_stat_dbscan = calinski_harabasz_score(X_clustering[dbscan_labels != -1], dbscan_core_labels)
else:
    silhouette_dbscan = None
    f_stat_dbscan = None

print("\nDBSCAN Results:")
if silhouette_dbscan is not None:
    print(f"Silhouette Score: {silhouette_dbscan}")
    print(f"F-statistic: {f_stat_dbscan}")
else:
    print("Not enough clusters were formed by DBSCAN for evaluation.")

# Analyze cluster characteristics
df_cluster_analysis = pd.DataFrame(X_clustering, columns=selected_features)
df_cluster_analysis['Cluster Label'] = labels
cluster_characteristics = df_cluster_analysis.groupby('Cluster Label').mean()
print("Cluster Characteristics:\n", cluster_characteristics)

