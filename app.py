# app.py

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import fitz  # PyMuPDF

# Step 2: Sample Dataset for Prediction
data = {
    'cgpa': [7.8, 8.5, 6.2, 9.1, 7.0],
    'technical_score': [70, 85, 50, 90, 60],
    'communication': [3, 4, 2, 5, 3],
    'teamwork': [4, 5, 3, 4, 2],
    'leadership': [3, 4, 1, 5, 2],
    'extracurricular': [2, 4, 1, 5, 3],
    'placement': ['yes', 'yes', 'no', 'yes', 'no']
}
df = pd.DataFrame(data)

# Preprocessing
le = LabelEncoder()
df['placement'] = le.fit_transform(df['placement'])
X = df.drop('placement', axis=1)
y = df['placement']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Resume Analysis Functions
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.lower()

def recommend_jobs(text):
    job_database = {
        "Data Scientist": {"keywords": ["machine learning", "pandas", "data"], "company": "Google", "salary": "₹12-20 LPA"},
        "Web Developer": {"keywords": ["html", "css", "javascript"], "company": "Infosys", "salary": "₹5-8 LPA"},
        "AI Engineer": {"keywords": ["tensorflow", "deep learning", "neural"], "company": "Microsoft", "salary": "₹10-18 LPA"},
        "Android Developer": {"keywords": ["android", "kotlin", "java"], "company": "Samsung", "salary": "₹6-10 LPA"},
        "DevOps Engineer": {"keywords": ["docker", "ci/cd", "kubernetes"], "company": "Amazon", "salary": "₹12-22 LPA"}
    }

    matches = []
    for role, data in job_database.items():
        if any(kw in text for kw in data["keywords"]):
            matches.append({
                "Role": role,
                "Company": data["company"],
                "Salary": data["salary"]
            })
    return matches

# Step 3: Streamlit App
def main():
    st.set_page_config(page_title="AI Placement Predictor", layout="wide")
    st.markdown("""
        <style>
        .title {
            text-align: center;
            font-size: 40px;
            color: #4A90E2;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: gray;
            margin-bottom: 30px;
        }
        </style>
        <div class="title">🤖 AI-Based Placement Prediction & Job Suggestion System</div>
        <div class="subtitle">Predict your placement chances and explore job opportunities based on your resume</div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎯 Predict Placement", "📄 Resume Job Suggestions"])

    with tab1:
        st.header("📝 Enter Academic & Soft Skill Details")

        cgpa = st.slider("CGPA (0-10)", 0.0, 10.0, 7.0)
        tech_score = st.slider("Technical Score (0-100)", 0, 100, 60)
        communication = st.slider("Communication Skills (1-5)", 1, 5, 3)
        teamwork = st.slider("Teamwork Skills (1-5)", 1, 5, 3)
        leadership = st.slider("Leadership Skills (1-5)", 1, 5, 3)
        extracurricular = st.slider("Extracurricular Involvement (1-5)", 1, 5, 3)

        if st.button("🔍 Predict Placement"):
            input_data = np.array([[cgpa, tech_score, communication, teamwork, leadership, extracurricular]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

            if prediction == 1:
                st.success("🎉 You are likely to be placed!")
            else:
                st.error("⚠️ You are unlikely to be placed.")

            st.subheader("📌 Personalized Recommendations:")
            if communication < 3:
                st.write("- Enhance communication skills through practice or clubs.")
            if tech_score < 60:
                st.write("- Strengthen your technical profile with certifications.")
            if leadership < 3:
                st.write("- Join leadership roles in student organizations.")
            if extracurricular < 3:
                st.write("- Participate in hackathons, events or cultural activities.")

    with tab2:
        st.header("📤 Upload Resume for Job Suggestions")
        uploaded_file = st.file_uploader("Upload Resume (PDF only)", type="pdf")

        if uploaded_file is not None:
            resume_text = extract_text_from_pdf(uploaded_file)
            st.text_area("📄 Resume Extract", resume_text[:1000], height=200)

            st.subheader("💼 Recommended Jobs:")
            jobs = recommend_jobs(resume_text)
            if jobs:
                for job in jobs:
                    st.markdown(f"""
                        <div style='border:1px solid #ccc; padding:10px; margin:10px; border-radius:10px;'>
                            <strong>Role:</strong> {job['Role']}<br>
                            <strong>Company:</strong> {job['Company']}<br>
                            <strong>Estimated Salary:</strong> {job['Salary']}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No clear matches found. Try improving your resume with more technical terms.")

if __name__ == "__main__":
    main()
