import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template

st.set_page_config(page_title="Sentiment Analysis Dashboard",page_icon="📊",layout="wide")
layout="centered"

st.markdown("<h1 style='text-align: center;'>Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;'>Dataset Preview</h2>", unsafe_allow_html=True)

df=pd.read_csv(r"D:\current documents\Desktop\Sentimental Analysis\Sentiment_analysis.csv")
st.dataframe(df)


st.header("Key Question 1")
st.write("What is the overall sentiment of user reviews?")
st.write("→ Classify each review as Positive, Neutral, or Negative and compute their proportions.")
sentiment_counts = df["sentiment"].value_counts()

st.subheader("Sentiment Distribution")

# Show table
st.write(sentiment_counts)

# Bar chart
fig, ax = plt.subplots()
sentiment_counts.plot(kind="bar", ax=ax)

ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of Reviews")

st.pyplot(fig)

# Percentage calculation
st.subheader("Sentiment Proportions")

sentiment_percent = df["sentiment"].value_counts(normalize=True) * 100
st.write(sentiment_percent)

st.header("Key Question 2")
st.write("How does sentiment vary by rating?")
st.write("→ Do 1-star reviews always contain negative sentiment? Is there any mismatch between ratings and actual text?")

# Sentiment vs Rating table
sentiment_rating = pd.crosstab(df["rating"], df["sentiment"])

st.subheader("Sentiment Distribution by Rating")

st.dataframe(sentiment_rating)

# Visualization
fig, ax = plt.subplots()
sentiment_rating.plot(kind="bar", stacked=True, ax=ax)

ax.set_xlabel("Rating")
ax.set_ylabel("Number of Reviews")
ax.set_title("Sentiment vs Rating")

st.pyplot(fig)


st.header("Key Question 3")

st.write("How has sentiment changed over time?")
st.write("→ Analyze sentiment trends by month or week to spot peaks in satisfaction or dissatisfaction.")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Extract month
df["month"] = df["date"].dt.to_period("M")

# Sentiment count by month
trend = pd.crosstab(df["month"], df["sentiment"])

st.subheader("Monthly Sentiment Trend")

st.dataframe(trend)

# Plot trend
fig, ax = plt.subplots()

trend.plot(ax=ax)

ax.set_xlabel("Month")
ax.set_ylabel("Number of Reviews")
ax.set_title("Sentiment Trend Over Time")

st.pyplot(fig)

st.header("Key Question 4")

st.write("Do verified users tend to leave more positive or negative reviews?")
st.write("→ Compare sentiment distribution between verified_purchase = Yes vs No.")

# Create sentiment comparison table
verified_sentiment = pd.crosstab(df["verified_purchase"], df["sentiment"])

st.subheader("Sentiment Distribution by Verification Status")

st.dataframe(verified_sentiment)

# Visualization
fig, ax = plt.subplots()

verified_sentiment.plot(kind="bar", stacked=True, ax=ax)

ax.set_xlabel("Verified Purchase")
ax.set_ylabel("Number of Reviews")
ax.set_title("Verified vs Non-Verified Sentiment")

st.pyplot(fig)

st.header("Key Question 5")

st.write("Are longer reviews more likely to be negative or positive?")
st.write("→ Compare average sentiment scores with review length.")
# Calculate review length (word count)
df["review_length"] = df["review"].astype(str).apply(lambda x: len(x.split()))

# Average review length by sentiment
length_analysis = df.groupby("sentiment")["review_length"].mean()

st.subheader("Average Review Length by Sentiment")

st.write(length_analysis)

# Visualization
fig, ax = plt.subplots()

length_analysis.plot(kind="bar", ax=ax)

ax.set_xlabel("Sentiment")
ax.set_ylabel("Average Review Length (words)")
ax.set_title("Review Length vs Sentiment")

st.pyplot(fig)

st.header("Key Question 6")

st.write("Which locations show the most positive or negative sentiment?")
st.write("→ Helps uncover region-based user experience issues or appreciation.")

# Sentiment count by location
location_sentiment = pd.crosstab(df["location"], df["sentiment"])

st.subheader("Sentiment Distribution by Location")

st.dataframe(location_sentiment)

# Visualization
fig, ax = plt.subplots()

location_sentiment.plot(kind="bar", stacked=True, ax=ax)

ax.set_xlabel("Location")
ax.set_ylabel("Number of Reviews")
ax.set_title("Sentiment by Location")

st.pyplot(fig)

st.header("Key Question 7")

st.write("Is there a difference in sentiment across platforms (Web vs Mobile)?")
st.write("→ Identify where the user experience might need improvement.")


# Sentiment count by platform
platform_sentiment = pd.crosstab(df["platform"], df["sentiment"])

st.subheader("Sentiment Distribution by Platform")

st.dataframe(platform_sentiment)

# Visualization
fig, ax = plt.subplots()

platform_sentiment.plot(kind="bar", stacked=True, ax=ax)

ax.set_xlabel("Platform")
ax.set_ylabel("Number of Reviews")
ax.set_title("Sentiment by Platform")

st.pyplot(fig)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.header("Key Question 8")

st.write("Which ChatGPT versions are associated with higher or lower sentiment?")
st.write("→ Determine if a version release impacted user satisfaction.")

# Sentiment distribution by version
version_sentiment = pd.crosstab(df["version"], df["sentiment"])

st.subheader("Sentiment Distribution by Version")

st.dataframe(version_sentiment)

# Visualization
fig, ax = plt.subplots()

version_sentiment.plot(kind="bar", stacked=True, ax=ax)

ax.set_xlabel("ChatGPT Version")
ax.set_ylabel("Number of Reviews")
ax.set_title("Sentiment by ChatGPT Version")

st.pyplot(fig)

from sklearn.feature_extraction.text import CountVectorizer

st.header("Key Question 9")

st.write("What are the most common negative feedback themes?")
st.write("→ Identify recurring issues mentioned in negative reviews.")


# Filter negative reviews
negative_reviews = df[df["sentiment"] == "Negative"]["review"]

# Convert text to keyword frequency
vectorizer = CountVectorizer(stop_words='english', max_features=15)
X = vectorizer.fit_transform(negative_reviews)

words = vectorizer.get_feature_names_out()
counts = X.sum(axis=0).A1

keyword_df = pd.DataFrame({"Keyword": words, "Count": counts})
keyword_df = keyword_df.sort_values(by="Count", ascending=False)

st.subheader("Top Negative Keywords")

st.dataframe(keyword_df)

# Plot
fig, ax = plt.subplots()

ax.bar(keyword_df["Keyword"], keyword_df["Count"])

ax.set_xlabel("Keywords")
ax.set_ylabel("Frequency")
ax.set_title("Most Common Negative Feedback Themes")

plt.xticks(rotation=45)

st.pyplot(fig)

st.header("Key Question 10")

st.write("Which keywords or phrases are most associated with each sentiment class?")
st.write("→ Identify common words in Positive, Neutral, and Negative reviews.")


# Function to extract keywords
def get_keywords(text):
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    X = vectorizer.fit_transform(text)

    words = vectorizer.get_feature_names_out()
    counts = X.sum(axis=0).A1

    keyword_df = pd.DataFrame({"Keyword": words, "Count": counts})
    keyword_df = keyword_df.sort_values(by="Count", ascending=False)

    return keyword_df

# Positive keywords
st.subheader("Positive Review Keywords")
positive_words = get_keywords(df[df["sentiment"]=="Positive"]["review"])
st.dataframe(positive_words)

# Neutral keywords
st.subheader("Neutral Review Keywords")
neutral_words = get_keywords(df[df["sentiment"]=="Neutral"]["review"])
st.dataframe(neutral_words)

# Negative keywords
st.subheader("Negative Review Keywords")
negative_words = get_keywords(df[df["sentiment"]=="Negative"]["review"])
st.dataframe(negative_words)