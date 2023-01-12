import glob
import streamlit as st
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

filepaths = sorted(glob.glob("diary/*.txt"))


negativity = []
positivity = []

for filepath in filepaths:
    with open(filepath) as file:
        content = file.read()
    scores = analyzer.polarity_scores(content)
    positivity.append(scores["pos"])
    negativity.append(scores["neg"])

# Getting the dates from the file names
dates = [name.strip("diary\\").strip(".txt") for name in filepaths]


st.title("Diary Tone")
st.subheader("Positivity")
pos_figure = px.line(x=dates, y=positivity,
                     labels={"x": "Date", "y": "Positivity"})
st.plotly_chart(pos_figure)

st.subheader("Negativity")
pos_figure = px.line(x=dates, y=negativity,
                     labels={"x": "Date", "y": "Negativity"})
st.plotly_chart(pos_figure)
