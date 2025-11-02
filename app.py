import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


# Page config & load CSS

st.set_page_config(page_title="Personalized Healthcare System", layout="wide")

# Load CSS file
if os.path.exists("styles.css"):
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
   
    st.markdown("""
    <style>
      .stApp { background-color:#121212; color:#E6E6E6; }
    </style>
    """, unsafe_allow_html=True)


# Database helpers

DB_PATH = "healthcare_system.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def authenticate_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    res = c.fetchone()
    conn.close()
    return res

def log_recommendation(username, symptoms, medicines):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO recommendations (username, symptoms, medicines) VALUES (?, ?, ?)", (username, symptoms, ", ".join(medicines)))
    conn.commit()
    conn.close()


# Symptom - Medicine dataset

# Each entry has 'symptoms' text and 'medicines' 
symptom_templates = [
    {"symptoms":"fever cough body ache sore throat","medicines":"Paracetamol, Dolo-650, Crocin"},
    {"symptoms":"high fever persistent cough","medicines":"Paracetamol, Azithromycin"},
    {"symptoms":"headache nausea light sensitivity","medicines":"Aspirin, Ibuprofen"},
    {"symptoms":"running nose sneezing congestion","medicines":"Cetirizine, Sinarest"},
    {"symptoms":"stomach pain acidity heartburn","medicines":"Pantoprazole, Digene"},
    {"symptoms":"persistent cough phlegm wheeze","medicines":"Ambroxol, Salbutamol"},
    {"symptoms":"sore throat cough fever","medicines":"Paracetamol, Benadryl"},
    {"symptoms":"urinary pain burning sensation","medicines":"Nitrofurantoin, Ciprofloxacin"},
    {"symptoms":"chest pain shortness of breath","medicines":"Aspirin, Atorvastatin"},
    {"symptoms":"low energy excessive thirst frequent urination","medicines":"Metformin, Insulin"}
]
templates_df = pd.DataFrame(symptom_templates)


vectorizer = TfidfVectorizer(stop_words='english')
template_vectors = vectorizer.fit_transform(templates_df['symptoms'])


# ML-based medicine recommendation 

def recommend_medicines_ml(user_symptom_text, top_k_templates=3, top_meds=3):
    if not user_symptom_text.strip():
        return []

    # compute similarity to templates
    user_vec = vectorizer.transform([user_symptom_text])
    sims = cosine_similarity(user_vec, template_vectors).flatten()
    top_idx = sims.argsort()[-top_k_templates:][::-1]  # top templates

    # collect medicines from top templates, split and count frequency
    meds = []
    for i in top_idx:
        meds += [m.strip() for m in templates_df.loc[i, 'medicines'].split(",")]
    # frequency and sort
    meds_series = pd.Series(meds).value_counts()
    recommended = meds_series.index.tolist()[:top_meds]
    return recommended


# Disease risk predictor 

def predict_disease_risk(age, bp, glucose, hr):
    score = 0
    score += (bp - 90) / 100.0
    score += (glucose - 80) / 200.0
    score += (hr - 60) / 200.0
    score += (age - 20) / 200.0
    if score > 0.8:
        return "High risk — please consult a physician"
    elif score > 0.4:
        return "Moderate risk — monitor & consult if symptoms persist"
    else:
        return "Low risk"


# App layout

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None

def login_page():
    st.markdown("<h1 class='title'>Personalized Healthcare System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Please log in to continue</p>", unsafe_allow_html=True)
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
           
            st.rerun()
        else:
            st.error("Invalid username or password")

# After login 
def main_app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select", ["Disease Prediction", "Medicine Recommendation", "Analytics Dashboard", "Logout"])
    st.sidebar.caption(f"User: {st.session_state.username}")

   
    
    if page == "Logout":
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

    if page == "Disease Prediction":
        st.header("Disease Prediction")
        st.markdown("<p class='hint'>Enter health parameters below </p>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1, key="age_input")
        with col2:
            bp = st.slider("Blood Pressure (mmHg)", 80, 200, 120)
        with col3:
            glucose = st.slider("Glucose (mg/dL)", 70, 250, 100)
        with col4:
            hr = st.slider("Heart Rate (bpm)", 40, 150, 75)

        if st.button("Predict Risk"):
            result = predict_disease_risk(age, bp, glucose, hr)
            if "High" in result:
                st.error(result)
            elif "Moderate" in result:
                st.warning(result)
            else:
                st.success(result)

    elif page == "Medicine Recommendation":
        st.header("Medicine Recommendation")
        st.markdown("<p class='hint'>Describe your symptoms in a sentence (e.g., 'fever and cough for 2 days')</p>", unsafe_allow_html=True)
        user_symptoms = st.text_area("Symptoms", height=120)

        if st.button("Recommend Medicines"):
            recs = recommend_medicines_ml(user_symptoms, top_k_templates=4, top_meds=3)
            if recs:
                st.markdown("### Recommended (most relevant):")
                for med in recs:
                    st.markdown(f"- **{med}**")
                log_recommendation(st.session_state.username, user_symptoms, recs)
            else:
                st.warning("No clear match found. Please provide more details or consult a physician.")

    elif page == "Analytics Dashboard":
        st.header("Analytics Dashboard")
        conn = get_connection()
        df = pd.read_sql_query("SELECT * FROM recommendations", conn)
        conn.close()
        if df.empty:
            st.info("No recommendation records yet.")
        else:
            st.subheader("Recent Recommendation Logs")
            st.dataframe(df.tail(10))

            # Bar chart: top symptoms 
            keywords = []
            for s in df['symptoms'].dropna():
                keywords += s.lower().split()
            kw_series = pd.Series(keywords).value_counts().head(15)
           

            fig1, ax1 = plt.subplots()
            kw_series.plot.bar(ax=ax1, color="#00E5A8")
            ax1.set_title("Top symptom words")
            st.pyplot(fig1)

            # Pie chart: top medicines
            meds = []
            for m in df['medicines'].dropna():
                meds += [x.strip() for x in m.split(",")]
            med_counts = pd.Series(meds).value_counts().head(10)
            fig2, ax2 = plt.subplots()
            ax2.pie(med_counts.values, labels=med_counts.index, autopct='%1.1f%%', colors=plt.cm.viridis(np.linspace(0,1,len(med_counts))))
            ax2.set_title("Top recommended medicines")
            st.pyplot(fig2)

            # Heatmap: synthetic vitals correlation 
            vitals = pd.DataFrame(np.random.randint(60,180,(15,4)), columns=["Age","BP","Glucose","HR"])
            fig3, ax3 = plt.subplots()
            sns.heatmap(vitals.corr(), annot=True, cmap="mako", ax=ax3)
            ax3.set_title("Sample vitals correlation")
            st.pyplot(fig3)


# Run

if not st.session_state.authenticated:
    login_page()
else:
    main_app()
