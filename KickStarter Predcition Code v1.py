#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 18:18:24 2024

@author: yuanminglu
"""

#####################################################  Task 1   #####################################################

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV


# Load the Kickstarter dataset
df = pd.read_excel("Kickstarter.xlsx")

# Filter for only 'successful' or 'failed' states (ignore other states)
df = df[df['state'].isin(['successful', 'failed'])]

# Display all columns in the console
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Check for missing values and drop rows with missing data
print(df.isnull().sum())  # Display missing value counts per column
df.info()  # Display dataset info after cleaning

df['main_category'] = df['main_category'].fillna('Other')
df['main_category'].value_counts()

# Drop unnecessary variables that are not available at the project launch
df = df.drop(columns=[
    'id', 'pledged', 'spotlight', 'usd_pledged', 'backers_count', 
    'state_changed_at', 'state_changed_at_weekday', 'state_changed_at_month', 
    'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr', 'spotlight'
])

# Group countries into 'US', 'GB', and 'Non-US'
df['country'].value_counts()
df['country'] = df['country'].apply(lambda x: x if x in ['US', 'GB','CA','MX','DE','AU','FR'] else 'Non-US')


df['category'].value_counts()
top_categories = df['category'].value_counts().nlargest(50).index.tolist()

# Convert categories not in the top list to 'Other'
df['category'] = df['category'].apply(lambda x: x if x in top_categories else 'Other')


# One-hot encode categorical features (remove the first category to avoid multicollinearity)
categorical_features = ['deadline_weekday', 'created_at_weekday', 'launched_at_weekday','category']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Map main_category to broader groups and drop the original column
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

df['main_category_grouped'] = df['main_category'].map(category_mapping)  # Map categories
df = df.drop(columns=['main_category'])  # Drop the original main_category column

# Encode currency as binary (1 for USD, 0 otherwise)
df['currency'].value_counts()
df['currency'] = df['currency'].apply(lambda x: x if x in ['USD','GBP','EUR','CAD'] else 'Else')

binary_variables = ['disable_communication', 'staff_pick', 'staff_pick.1', 'show_feature_image', 'video']
df[binary_variables] = df[binary_variables].astype(int)

# Apply lambda function to each element of the selected columns
df = pd.get_dummies(df, columns=['main_category_grouped','country','currency'], drop_first=True)

# Perform sentiment analysis on the 'name' column   
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download required NLTK resources for sentiment analysis
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function to remove stopwords from text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(str(text))  # Tokenize text
    filtered_words = [word for word in words if word.lower() not in stop_words]  # Remove stopwords
    return ' '.join(filtered_words)  # Return the cleaned text

# Analyze sentiment using VADER
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)  # Get sentiment scores
    return sentiment_scores['compound']  # Return the compound score

# Apply sentiment analysis to the 'name' column
df['Title Sentiment'] = df['name'].apply(lambda x: analyze_sentiment(preprocess_text(x)))

### Date-Based Features ###
# Calculate campaign duration in days
df['campaign_duration'] = (pd.to_datetime(df['deadline']) - pd.to_datetime(df['launched_at'])).dt.days

# Create launch season categories based on month
df['launch_season'] = pd.cut(df['launched_at_month'], bins=[0, 3, 6, 9, 12], labels=['1', '2', '3', '4'])

# Drop unnecessary date columns
df = df.drop(columns=['deadline', 'created_at', 'launched_at', 'name'])

### Goal and Pledge-Related Features ###
# Convert goal to USD using static conversion rate
df['goal_usd'] = df['goal'] * df['static_usd_rate']

# Prepare features (X) and target (y)
X = df.drop(columns=['state'])  # Exclude the target variable
y = df['state'].apply(lambda x: 1 if x == 'successful' else 0)  # Encode target variable


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)


# Define the parameter grid for Random Forest


# Use GridSearchCV to tune Random Forest
# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Extract feature importances and sort them
feature_importance_df = pd.DataFrame(list(zip(X_train.columns,rf_classifier.feature_importances_)),
                                     columns=['predictor', 'feature importance'])
feature_importance_df = feature_importance_df.sort_values(by='feature importance', ascending=False)
print("Feature Importances:\n", feature_importance_df)

# Filter features with importance above a threshold
top_features = feature_importance_df[feature_importance_df['feature importance'] >= 0]['predictor'].tolist()
print("Selected top features by Random Forest:", top_features)

# Calculate the correlation matrix for the selected features
correlation_matrix = X_train[top_features].corr()
print("Correlation Matrix:\n", correlation_matrix)


# Create a set to store features to drop
features_to_drop = set()

# Identify and drop features with high correlations
threshold = 0.6  # Correlation threshold
for feature in correlation_matrix.columns:
    # Find features highly correlated with the current feature
    high_corr_features = correlation_matrix.index[(abs(correlation_matrix[feature]) > threshold) & 
                                                  (correlation_matrix.index != feature)].tolist()
    for correlated_feature in high_corr_features:
        # Keep the feature with higher importance
        if feature_importance_df.loc[feature_importance_df['predictor'] == feature, 'feature importance'].values[0] > \
           feature_importance_df.loc[feature_importance_df['predictor'] == correlated_feature, 'feature importance'].values[0]:
            features_to_drop.add(correlated_feature)
        else:
            features_to_drop.add(feature)

# Filter out the features to drop
final_features = [feature for feature in top_features if feature not in features_to_drop]

print("Features to drop due to high correlation:", features_to_drop)
print("Final selected features:", final_features)

# Subset the training and test datasets with the final features
X_train_final = X_train[final_features]
X_test_final = X_test[final_features]

# Standardize the features
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)  # Fit and transform training data
scaled_X_test = scaler.transform(X_test)  # Transform test data
scaled_X_train = pd.DataFrame(scaled_X_train, columns=X_train.columns)
scaled_X_test = pd.DataFrame(scaled_X_test, columns=X_test.columns)



# Manually exclude highly correlated features (correlation > 0.6)
final_features = ['goal_usd', 'staff_pick', 'campaign_duration', 'launched_at_hr', 
                  'created_at_day', 'blurb_len', 'launched_at_day', 
                  'deadline_hr', 'name_len', 'launched_at_yr', 'video', 'created_at_month', 
                  'static_usd_rate', 'Title Sentiment', 'deadline_month']

# Subset the data with selected features
X_train_rf = scaled_X_train[final_features]
X_test_rf = scaled_X_test[final_features]

# Loop through models and evaluate performance with cross-validation
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),  # Adjust k as needed
    'Classification Tree': DecisionTreeClassifier(max_depth=3),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(8), max_iter=1000, random_state=42)
}

from sklearn.model_selection import cross_val_score
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train_rf, y_train, cv=5, scoring='accuracy')
    print(f"Model: {model_name}")
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Accuracy: {np.mean(cv_scores):.4f}")
    print(f"Standard Deviation of Accuracy: {np.std(cv_scores):.4f}")
    print("\n")
    
    
param_grid = {
    'n_estimators': [50, 100, 150, 200],  # Number of trees
    'max_features': [3, 4, 5, 6],  # Max number of features per split
    'min_samples_leaf': [1, 2, 3, 4]  # Min samples required in leaf nodes
}

# Perform hyperparameter tuning for Random Forest
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train_rf, y_train)
print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best cross-validation accuracy for Random Forest:", grid_search_rf.best_score_)


# Perform hyperparameter tuning for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 8, 10]
}
gb = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_gb.fit(X_train_rf, y_train)
print("Best parameters for Gradient Boosting:", grid_search_gb.best_params_)
print("Best cross-validation accuracy for Gradient Boosting:", grid_search_gb.best_score_)

# Perform hyperparameter tuning for Neural Network
param_grid_nn = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)]
}
nn = MLPClassifier(max_iter=200, random_state=42)
grid_search_nn = GridSearchCV(estimator=nn, param_grid=param_grid_nn, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_nn.fit(X_train_rf, y_train)
print("Best parameters for Neural Network:", grid_search_nn.best_params_)
print("Best cross-validation accuracy for Neural Network:", grid_search_nn.best_score_)


# Make predictions on the test set of the best model
y_pred = grid_search_gb.predict(X_test_rf)

# Evaluate the model's performance using accuracy
accuracy_gbt = accuracy_score(y_test, y_pred)
print(accuracy_gbt)

#####################################################  GRADING SCRIPT  #####################################################

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the Kickstarter grading dataset
df_grading = pd.read_excel("Kickstarter.xlsx")

df_grading = df_grading.dropna()

# Filter for only 'successful' or 'failed' states (ignore other states)
df_grading = df_grading[df_grading['state'].isin(['successful', 'failed'])]

# Display all columns in the console for verification
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Check for missing values and remove rows with missing data
print(df_grading.isnull().sum())  # Display missing value counts
df_grading = df_grading.dropna()  # Drop rows with missing values
df_grading.info()  # Display dataset information after cleaning

# Drop unnecessary columns and keep only variables available at project launch
df_grading = df_grading.drop(columns=[
    'id', 'pledged', 'spotlight', 'usd_pledged', 'backers_count', 
    'state_changed_at', 'state_changed_at_weekday', 'state_changed_at_month', 
     'state_changed_at_day', 'state_changed_at_yr', 'state_changed_at_hr', 'spotlight'
])

# Group countries into 'US', 'GB', and 'Non-US'
df_grading['country'] = df_grading['country'].apply(lambda x: x if x in ['US', 'GB','CA','MX','DE','AU','FR'] else 'Non-US')

# One-hot encode categorical features
categorical_features = ['deadline_weekday', 'created_at_weekday', 'launched_at_weekday']
df_grading = pd.get_dummies(df_grading, columns=categorical_features, drop_first=True)

# Map main categories to broader groups
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

# Map categories and drop the original column
df_grading['main_category_grouped'] = df_grading['main_category'].map(category_mapping)
df_grading = df_grading.drop(columns=['main_category'])

df_grading['currency'] = df_grading['currency'].apply(lambda x: x if x in ['USD','GBP','EUR','CAD'] else 'Else')

binary_variables = ['disable_communication', 'staff_pick', 'staff_pick.1', 'show_feature_image', 'video']
df_grading[binary_variables] = df_grading[binary_variables].astype(int)

# Apply lambda function to each element of the selected columns
df_grading = pd.get_dummies(df_grading, columns=['main_category_grouped','country','currency'], drop_first=True)


# Perform sentiment analysis on project titles
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download required NLTK resources for sentiment analysis
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text by removing stopwords
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(str(text))  # Tokenize text
    filtered_words = [word for word in words if word.lower() not in stop_words]  # Remove stopwords
    return ' '.join(filtered_words)  # Return cleaned text

# Function to analyze sentiment using VADER
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)  # Get sentiment scores
    return sentiment_scores['compound']  # Return compound score

# Apply sentiment analysis on project titles
df_grading['Title Sentiment'] = df_grading['name'].apply(lambda x: analyze_sentiment(preprocess_text(x)))

### Date-Based Features ###
# Calculate campaign duration in days
df_grading['campaign_duration'] = (pd.to_datetime(df_grading['deadline']) - pd.to_datetime(df_grading['launched_at'])).dt.days

# Categorize launch month into seasons
df_grading['launch_season'] = pd.cut(df_grading['launched_at_month'], bins=[0, 3, 6, 9, 12], labels=['1', '2', '3', '4'])

# Drop unnecessary columns
df_grading = df_grading.drop(columns=['deadline', 'created_at', 'launched_at', 'name', 'category'])

### Goal and Pledge-Related Features ###
# Convert goal to USD using static conversion rate
df_grading['goal_usd'] = df_grading['goal'] * df_grading['static_usd_rate']

# Prepare features and target
X_grading = df_grading.drop(columns=['state'])  # Features
y_grading = df_grading['state'].apply(lambda x: 1 if x == 'successful' else 0)  # Target variable

# Standardize the features
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X_grading)  # Standardize training data
scaled_X = pd.DataFrame(scaled_X, columns=X_grading.columns)

# Select features identified by Lasso (or other feature selection techniques)
scaled_X = scaled_X[['goal_usd', 'staff_pick', 'campaign_duration', 'launched_at_hr', 
                  'created_at_day', 'blurb_len_clean', 'launched_at_day', 
                  'created_at_hr', 'name_len', 'launched_at_yr', 'created_at_month', 
                  'static_usd_rate', 'Title Sentiment','video']]


# Make predictions on the test set
y_grading_pred = grid_search_gb.predict(scaled_X)

# Evaluate the model's performance using accuracy
accuracy_gbt = accuracy_score(y_grading, y_grading_pred)
print("Accuracy on the grading dataset:", accuracy_gbt)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:48:13 2024

@author: yuanminglu
"""

