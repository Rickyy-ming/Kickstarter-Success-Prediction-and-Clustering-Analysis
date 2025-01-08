# Kickstarter-Success-Prediction-and-Clustering-Analysis

This repository contains a comprehensive analysis of Kickstarter project data, focusing on predicting project success and uncovering patterns through clustering. The project employs machine learning techniques to provide actionable insights for both Kickstarter administrators and project creators.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Usage](#usage)
6. [Dependencies](#dependencies)
7. [Acknowledgments](#acknowledgments)

## Introduction

The project addresses two key tasks:
1. **Classification**: Predict whether a Kickstarter project will succeed at the moment of its launch using only features available at that time.
2. **Clustering**: Group Kickstarter projects into distinct clusters to identify patterns and improve project performance.

The dataset was sourced from Kickstarter and includes details such as project goals, categories, launch times, and country information.

## Features

Key features engineered and used in this project include:
- **Goal Amount in USD**: A predictor of project success.
- **Campaign Duration**: Derived from launch and deadline dates.
- **Temporal Features**: Day, month, year, and hour attributes for deadlines and launches.
- **Staff Pick and Video Indicators**: Highlighting platform endorsements and presentation quality.

## Methodology

### Task 1: Predictive Modeling
- **Algorithms Used**: Gradient Boosting, Random Forest, Logistic Regression, K-Nearest Neighbors, Neural Networks.
- **Best Model**: Gradient Boosting with a test accuracy of 81.48%.
- **Key Insights**:
  - Realistic goals increase success likelihood.
  - Staff picks significantly boost project visibility and engagement.

### Task 2: Clustering Analysis
- **Clustering Method**: K-Means with 3 clusters.
- **Cluster Insights**:
  - **Cluster 1**: High visibility and support (strong backer engagement, realistic goals).
  - **Cluster 2**: Moderate support (potential for growth with better visibility).
  - **Cluster 3**: Low support (overly ambitious goals, niche categories).

## Results

### Predictive Modeling
- **Accuracy**: 81.48% (Gradient Boosting).
- **Top Features**: `goal_usd`, `staff_pick`, `campaign_duration`, `launch_at_hr`.

### Clustering
- **Optimal Clusters**: 3 clusters with a Silhouette Score of 0.52.
- **Insights**: Identification of traits associated with successful projects and strategies for improvement.
