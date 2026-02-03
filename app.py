import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import joblib
import os
import joblib
import streamlit as st

if not os.path.exists("model.pkl"):
    st.error("❌ Model not found. Run train_model.py first.")
    st.stop()

model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="wide"
)

st.markdown(
    "<h1 style='color:#7C4DFF'>🎓 BALAJI Elite School</h1>",
    unsafe_allow_html=True
)

st.write(" Student Performance Dashboard")

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER,
    name TEXT,
    grade INTEGER,
    gender TEXT,
    maths INTEGER,
    english INTEGER,
    science INTEGER,
    attendance INTEGER
)
""")

df = pd.read_csv("students.csv")
df.to_sql("students", conn, if_exists="replace", index=False)

conn.commit()
conn.close()
df = pd.read_csv("students.csv")

c1, c2, c3 = st.columns(3)
c1.metric("👨‍🎓 Students", len(df))
c2.metric("📅 Attendance", f"{df.attendance.mean():.1f}%")
c3.metric("📊 Exam Avg", f"{df[['maths','english','science']].mean().mean():.1f}%")

st.divider()

fig1 = px.pie(
    df,
    names="grade",
    hole=0.6,
    title="Student Distribution by Grade"
)

fig2 = px.bar(
    df,
    x="name",
    y=["maths", "english", "science"],
    barmode="group",
    title="Subject-wise Performance"
)

col1, col2 = st.columns(2)
col1.plotly_chart(fig1, use_container_width=True)
col2.plotly_chart(fig2, use_container_width=True)

st.subheader(" Performance Prediction")

model = joblib.load("model.pkl")
student = st.selectbox("Select Student", df["name"])

row = df[df["name"] == student]
prediction = model.predict(
    row[['maths','english','science','attendance']]
)[0]

st.success(f"Predicted Final Score: {prediction:.2f}%")
