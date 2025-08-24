# Streamlit Student Dashboard (simple + no matplotlib)
# Save as: student_dashboard_simple.py
# Run:
#   pip install streamlit pandas numpy
#   streamlit run student_dashboard_simple.py

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# ------------------------------
# 1) APP SETUP
# ------------------------------
st.set_page_config(page_title="Student Dashboard (Simple)", layout="wide")
st.title("📊 Student Dashboard (Simple, No Matplotlib)")
st.caption("Interactive insights for ~100 students — filters + built-in charts only")

# ------------------------------
# 2) SYNTHETIC DATA (EMBEDDED)
# ------------------------------
np.random.seed(42)
num_students = 100

names = [f"Student{str(i).zfill(3)}" for i in range(1, num_students + 1)]
classes = np.random.choice(["A", "B", "C", "D"], size=num_students, p=[0.28, 0.28, 0.22, 0.22])
genders = np.random.choice(["Female", "Male"], size=num_students)
ages = np.random.randint(15, 19, size=num_students)
attendance = np.clip(np.random.normal(92, 5, size=num_students), 70, 100).round(1)

def subject_scores(mu, sigma, cls):
    bump = {"A": 3, "B": 1, "C": -1, "D": -3}[cls]
    return int(np.clip(np.random.normal(mu + bump, sigma), 30, 100))

math = [subject_scores(75, 10, c) for c in classes]
science = [subject_scores(78, 9, c) for c in classes]
english = [subject_scores(80, 8, c) for c in classes]
history = [subject_scores(73, 11, c) for c in classes]
it = [subject_scores(82, 7, c) for c in classes]

df = pd.DataFrame({
    "Name": names,
    "Gender": genders,
    "Class": classes,
    "Age": ages,
    "Attendance%": attendance,
    "Math": math,
    "Science": science,
    "English": english,
    "History": history,
    "IT": it
})
df["Total"] = df[["Math", "Science", "English", "History", "IT"]].sum(axis=1)
df["Average"] = (df["Total"] / 5).round(2)
df["Passed_All"] = (df[["Math", "Science", "English", "History", "IT"]] >= 50).all(axis=1)

# ------------------------------
# 3) SIDEBAR FILTERS (SIMPLE)
# ------------------------------
st.sidebar.header("🔎 Filters")
class_sel = st.sidebar.multiselect("Class", sorted(df["Class"].unique()), default=sorted(df["Class"].unique()))
gender_sel = st.sidebar.multiselect("Gender", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))
min_att = st.sidebar.slider("Minimum Attendance (%)", 70, 100, 80)
subject_sel = st.sidebar.selectbox("Focus Subject", ["Math", "Science", "English", "History", "IT"])
top_n = st.sidebar.slider("Top N by Average", 5, 20, 10)

mask = (
    df["Class"].isin(class_sel) &
    df["Gender"].isin(gender_sel) &
    (df["Attendance%"] >= min_att)
)
fdf = df.loc[mask].copy()

st.sidebar.success(f"Active filters → Rows: {len(fdf)} / {len(df)}")

# ------------------------------
# 4) KPIs
# ------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Students", len(fdf))
with c2:
    st.metric("Avg Attendance", f"{fdf['Attendance%'].mean():.1f}%" if len(fdf) else "—")
with c3:
    st.metric("Overall Average", f"{fdf['Average'].mean():.1f}" if len(fdf) else "—")
with c4:
    st.metric("Pass Rate (All Subjects)", f"{(fdf['Passed_All'].mean()*100):.1f}%" if len(fdf) else "—")
with c5:
    st.metric(f"{subject_sel} Mean", f"{fdf[subject_sel].mean():.1f}" if len(fdf) else "—")

st.divider()

# ------------------------------
# 5) DATA TABLE + DOWNLOAD
# ------------------------------
st.subheader("📄 Filtered Dataset")
st.dataframe(fdf.sort_values("Average", ascending=False), use_container_width=True)

csv_buf = StringIO()
fdf.to_csv(csv_buf, index=False)
st.download_button("⬇️ Download filtered CSV", csv_buf.getvalue(), "students_filtered.csv", "text/csv")

st.divider()

# ------------------------------
# 6) CHARTS (BUILT-IN ONLY)
# ------------------------------
left, right = st.columns(2)

# A) Subject means (Bar)
with left:
    st.subheader("📦 Average Score per Subject (Bar)")
    if len(fdf):
        subj_means = fdf[["Math", "Science", "English", "History", "IT"]].mean().sort_values(ascending=False)
        st.bar_chart(subj_means)
    else:
        st.info("No data after filters.")

# B) Average by Student index (Line)
with right:
    st.subheader("📈 Average Score by Student (Line)")
    if len(fdf):
        temp = fdf.reset_index(drop=True)[["Average"]]
        temp.index = np.arange(1, len(temp) + 1)  # Student index on X
        st.line_chart(temp)
    else:
        st.info("No data after filters.")

st.divider()

# C) Top N by Average (Bar)
st.subheader(f"🏆 Top {top_n} Students by Average")
if len(fdf):
    top_df = fdf.nlargest(top_n, "Average")[["Name", "Average"]].set_index("Name")
    st.bar_chart(top_df)
else:
    st.info("No data after filters.")

st.divider()

# D) Scatter: Math vs Science (Built-in scatter)
st.subheader("🔬 Scatter: Math vs Science")
if len(fdf):
    st.scatter_chart(fdf, x="Math", y="Science", color="Class")
else:
    st.info("No data after filters.")

st.divider()

# E) Simple Histogram for Focus Subject (bin + bar_chart)
st.subheader(f"📊 {subject_sel} Score Distribution (Binned)")
if len(fdf):
    # Create simple bins and count
    bins = np.arange(0, 101, 10)  # 0-100 step 10
    counts, edges = np.histogram(fdf[subject_sel], bins=bins)
    # Build a dataframe with bin labels for a bar chart
    bin_labels = [f"{edges[i]}–{edges[i+1]-1}" for i in range(len(edges)-1)]
    hist_df = pd.DataFrame({"Bins": bin_labels, "Count": counts}).set_index("Bins")
    st.bar_chart(hist_df)
else:
    st.info("No data after filters.")

st.caption("Tip: Use the sidebar to filter. All charts use Streamlit’s built-in chart functions (no matplotlib).")
