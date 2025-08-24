import streamlit as st
import pandas as pd

# Title
st.title("📈 Student Data Analysis - Line Graphs")

data = {
"Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
"Age": [16, 17, 16, 17, 16],
"Math": [78, 88, 90, 65, 82],
"Science": [85, 79, 95, 72, 88],
"English": [92, 85, 87, 70, 91]
}

df = pd.DataFrame(data)
st.subheader("Student Data")
st.write(df)

st.subheader("Math Scores Over Students")
st.line_chart(df.set_index("Name")["Math"])