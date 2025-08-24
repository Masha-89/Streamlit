# Streamlit Student Analytics Dashboard (100 students, self-contained)
# Paste this into a file (e.g., student_dashboard.py) and run:
#   pip install streamlit pandas numpy matplotlib
#   streamlit run student_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# ------------------------------
# 1) APP SETUP
# ------------------------------
st.set_page_config(page_title="Student Analytics Dashboard", layout="wide")
st.title("📊 Student Analytics Dashboard")
st.caption("Interactive insights for ~100 students with embedded demo data")

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

# Subject scores with slight class-based bias to make it interesting
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
# 3) SIDEBAR FILTERS
# ------------------------------
st.sidebar.header("🔎 Filters")
class_sel = st.sidebar.multiselect("Class", options=sorted(df["Class"].unique()), default=sorted(df["Class"].unique()))
gender_sel = st.sidebar.multiselect("Gender", options=sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))
min_att = st.sidebar.slider("Minimum Attendance (%)", 70, 100, 80)
subject_sel = st.sidebar.selectbox("Focus Subject", ["Math", "Science", "English", "History", "IT"])
top_n = st.sidebar.slider("Top N by Average", 5, 20, 10)

filt = (
    df["Class"].isin(class_sel) &
    df["Gender"].isin(gender_sel) &
    (df["Attendance%"] >= min_att)
)
fdf = df.loc[filt].copy()

st.sidebar.success(f"Active filters -> Rows: {len(fdf)} / {len(df)}")

# ------------------------------
# 4) KPIs
# ------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Students", len(fdf))
with c2:
    st.metric("Avg Attendance", f"{fdf['Attendance%'].mean():.1f}%")
with c3:
    st.metric("Overall Average", f"{fdf['Average'].mean():.1f}")
with c4:
    pass_rate = (fdf["Passed_All"].mean() * 100) if len(fdf) else 0
    st.metric("Pass Rate (All Subjects)", f"{pass_rate:.1f}%")
with c5:
    st.metric(f"{subject_sel} Mean", f"{fdf[subject_sel].mean():.1f}")

st.divider()

# ------------------------------
# 5) DATA TABLE & DOWNLOAD
# ------------------------------
st.subheader("📄 Filtered Dataset")
st.dataframe(fdf.sort_values("Average", ascending=False), use_container_width=True)

csv_buf = StringIO()
fdf.to_csv(csv_buf, index=False)
st.download_button("⬇️ Download filtered CSV", csv_buf.getvalue(), file_name="students_filtered.csv", mime="text/csv")

st.divider()

# ------------------------------
# 6) VISUALS - BAR / LINE / HIST / BOX / SCATTER / HEATMAP
# ------------------------------
left, right = st.columns([1, 1])

# A) Average per Subject (Bar)
with left:
    st.subheader("📦 Average Score per Subject (Bar)")
    subj_means = fdf[["Math", "Science", "English", "History", "IT"]].mean().sort_values(ascending=False)
    st.bar_chart(subj_means)

# B) Trend of Averages by Student Index (Line)
with right:
    st.subheader("📈 Average Score by Student (Line)")
    if len(fdf) > 0:
        tmp = fdf.reset_index(drop=True).copy()
        tmp["StudentIndex"] = np.arange(1, len(tmp) + 1)
        tmp = tmp[["StudentIndex", "Average"]].set_index("StudentIndex")
        st.line_chart(tmp)
    else:
        st.info("No data after filters.")

st.divider()

# C) Distribution of Focus Subject (Histogram)
st.subheader(f"📊 {subject_sel} Score Distribution")
fig_hist, ax_hist = plt.subplots()
ax_hist.hist(fdf[subject_sel], bins=12, edgecolor="black")
ax_hist.set_xlabel(subject_sel)
ax_hist.set_ylabel("Count")
ax_hist.set_title(f"{subject_sel} Distribution")
st.pyplot(fig_hist)

# D) Boxplot for All Subjects
st.subheader("🧰 Boxplot of All Subjects")
fig_box, ax_box = plt.subplots()
ax_box.boxplot([fdf["Math"], fdf["Science"], fdf["English"], fdf["History"], fdf["IT"]],
               labels=["Math", "Science", "English", "History", "IT"])
ax_box.set_ylabel("Score")
ax_box.set_title("Score Spread by Subject")
st.pyplot(fig_box)

st.divider()

# E) Top N Students by Average (Bar)
st.subheader(f"🏆 Top {top_n} Students by Average")
top_df = fdf.nlargest(top_n, "Average")[["Name", "Average"]].set_index("Name")
st.bar_chart(top_df)

# F) Scatter: Math vs Science (bubble by Attendance)
st.subheader("🔬 Scatter: Math vs Science (Bubble = Attendance)")
fig_scatter, ax_scatter = plt.subplots()
sizes = (fdf["Attendance%"] - fdf["Attendance%"].min() + 1) * 5
ax_scatter.scatter(fdf["Math"], fdf["Science"], s=sizes, alpha=0.6)
ax_scatter.set_xlabel("Math")
ax_scatter.set_ylabel("Science")
ax_scatter.set_title("Math vs Science — Bubble size = Attendance%")
st.pyplot(fig_scatter)

st.divider()

# G) Correlation Heatmap (Subjects Only)
st.subheader("🧠 Correlation Heatmap (Subjects)")
subj_cols = ["Math", "Science", "English", "History", "IT"]
corr = fdf[subj_cols].corr()

fig_heat, ax_heat = plt.subplots()
im = ax_heat.imshow(corr.values, aspect="auto")
ax_heat.set_xticks(range(len(subj_cols)))
ax_heat.set_yticks(range(len(subj_cols)))
ax_heat.set_xticklabels(subj_cols, rotation=45, ha="right")
ax_heat.set_yticklabels(subj_cols)
for i in range(len(subj_cols)):
    for j in range(len(subj_cols)):
        ax_heat.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
ax_heat.set_title("Correlation Heatmap")
fig_heat.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
st.pyplot(fig_heat)

st.divider()

# ------------------------------
# 7) PER-CLASS COMPARISON (Bar)
# ------------------------------
st.subheader("🏫 Average Score by Class (All Subjects Combined)")
class_avg = fdf.groupby("Class")["Average"].mean().sort_values(ascending=False)
st.bar_chart(class_avg)

# ------------------------------
# 8) NOTES
# ------------------------------
st.caption("Tip: Use the sidebar to slice by Class, Gender, and Attendance. Download the filtered CSV to share.")
