import glob
import streamlit as st
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import os

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Make sure the diary folder exists
if not os.path.exists("diary"):
    os.mkdir("diary")

# Load the existing diary entries
filepaths = sorted(glob.glob("diary/*.txt"))

# Initialize lists to store positivity and negativity values
negativity = []
positivity = []

# Read existing diary entries and analyze sentiment
for filepath in filepaths:
    with open(filepath) as file:
        content = file.read()
    scores = analyzer.polarity_scores(content)
    positivity.append(scores["pos"])
    negativity.append(scores["neg"])

# Getting the dates from the file names
dates = [name.strip("diary/").strip(".txt") for name in filepaths]

# Streamlit title and entry section
st.title("Diary Tone")

# Create a text input box for the user to write their diary entry
diary_entry = st.text_area("Write your diary entry here:")

# Create a button to submit the diary entry
if st.button("Analyze Sentiment"):
    if diary_entry:
        # Analyze the sentiment of the diary entry
        sentiment = analyzer.polarity_scores(diary_entry)

        # Display the sentiment analysis results
        st.write("Sentiment Analysis Results:")
        st.write(f"Positive: {sentiment['pos']*100:.2f}%")
        st.write(f"Neutral: {sentiment['neu']*100:.2f}%")
        st.write(f"Negative: {sentiment['neg']*100:.2f}%")

        # Save the entry to a file
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"diary/entry_{timestamp}.txt"
        with open(filename, "w") as f:
            f.write(diary_entry)

        # Append the new entry's sentiment to the lists
        positivity.append(sentiment['pos'])
        negativity.append(sentiment['neg'])
        dates.append(timestamp)

        st.write(f"Your diary entry has been saved as {filename}.")
    else:
        st.write("Please write something in your diary entry.")

# Display graphs of positivity and negativity over time
st.subheader("Positivity")
pos_figure = px.line(x=dates, y=positivity,
                     labels={"x": "Date", "y": "Positivity"})
st.plotly_chart(pos_figure)

st.subheader("Negativity")
neg_figure = px.line(x=dates, y=negativity,
                     labels={"x": "Date", "y": "Negativity"})
st.plotly_chart(neg_figure)
