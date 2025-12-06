import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)

from xgboost import XGBClassifier
from scipy.stats import skew
from scipy.stats import gaussian_kde

# ======================
# Streamlit Page Setup
# ======================
st.set_page_config(page_title="Breast Cancer Classification Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("breast-cancer.csv")
    df.drop(columns=["id"], inplace=True)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    df["diagnosis"] = df["diagnosis"].astype(int)
    return df

df = load_data()
PLOTLY_TEMPLATE = "plotly_dark"

# ================================
# SIDEBAR MENU
# ================================
st.sidebar.title("Breast Cancer Dashboard")
st.sidebar.markdown("""
    - **Home**: Informasi umum dan preview data
    - **Deskripsi dan Visualisasi**: Analisis eksploratif data
    - **Model dan Prediksi**: Pembuatan, evaluasi model klasifikasi, dan prediksi
    """)
menu = st.sidebar.selectbox("Pilih Menu:", ["Home", "Deskripsi dan Visualisasi", "Model dan Prediksi"])
st.sidebar.markdown("---")
st.sidebar.markdown("""
    Dibuat oleh Kelompok 3:
    - Narendra Arshya Karnowo (5003231014)
    - Aldenka Rifqi Ganendra Murti (5003231033)
    - Moh. Nafri Rehanata (5003231044)
    """)

# ================================
# ========== HOME PAGE ===========
# ================================
if menu == "Home":
    st.title("Home — Analisis Breast Cancer")

    st.markdown("""
    ## Problem
    Kanker payudara adalah jenis kanker paling umum di kalangan wanita di seluruh dunia. Kanker ini menyumbang 
    25% dari semua kasus kanker, dan telah mempengaruhi lebih dari 2,1 juta orang pada tahun 2015 saja. Kanker 
    ini bermula ketika sel-sel di payudara mulai tumbuh secara tidak terkendali. Sel-sel ini biasanya membentuk 
    tumor yang dapat terlihat melalui sinar-X atau dirasakan sebagai benjolan di area payudara. Oleh karena itu, 
    Dataset *Breast Cancer Wisconsin* bertujuan untuk **memprediksi diagnosis tumor**:
    - **1 = Malignant** (Ganas)
    - **0 = Benign** (Jinak)  
    """)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    st.subheader("Proporsi Diagnosis")
    prop = df["diagnosis"].value_counts()
    prop_df = prop.reset_index()
    prop_df.columns = ["label", "count"]

    fig = px.pie(prop_df, names="label", values="count", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=["#65fbff", "#00bbff"],
                     title="Proporsi Diagnosis Tumor Payudara",)
    st.plotly_chart(fig, use_container_width=True)

# ================================
# ======= VISUALISASI PAGE =======
# ================================
elif menu == "Deskripsi dan Visualisasi":
    st.title("Visualisasi Data")

    st.subheader("Pilih Variabel untuk Visualisasi")
    num_cols = df.select_dtypes(include=['number']).drop(columns=["diagnosis"]).columns.tolist()
    col = st.selectbox("Pilih Variabel Numerik", num_cols)

    # === Statistika Deskriptif ===
    st.markdown(f"### Statistik Deskriptif: **{col}**")
    desc = df[col].describe().to_frame()
    st.dataframe(desc, use_container_width=False, width=400)

    # Korelasi dengan Diagnosis
    st.markdown(f"### Korelasi dengan Diagnosis: **{col}**")
    corr = df[col].corr(df["diagnosis"])
    st.write(f"Korelasi antara **{col}** dengan **diagnosis**: {corr:.3f}")

    # Strip Plot
    st.markdown(f"### Strip Plot Variabel: **{col}**")
    fig = px.strip(df, x="diagnosis", y=col, color="diagnosis",
                   color_discrete_map={0: "#65fbff", 1: "#00bbff"},
                   template=PLOTLY_TEMPLATE,
                   title=f"Strip Plot {col} berdasarkan Diagnosis")
    st.plotly_chart(fig, use_container_width=True)

    # === Histogram ===
    st.markdown(f"### Distribusi Variabel: **{col}**")
    skew_vals = df.groupby("diagnosis")[col].apply(skew)
    colors = {0: "#65fbff", 1: "#00bbff"}
    # Loop untuk tiap diagnosis, buat figure terpisah
    for diag in df["diagnosis"].unique():
        data = df[df["diagnosis"] == diag][col]
        fig = go.Figure()
        # Histogram
        fig.add_trace(go.Histogram(
            x=data,
            name=f"Diagnosis {diag}",
            opacity=0.6,
            marker_color=colors[diag],
            nbinsx=30,
            histnorm='probability density'
        ))
        # Density line (KDE)
        kde = gaussian_kde(data)
        x_range = pd.Series(data).sort_values()
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde(x_range),
            mode='lines',
            line=dict(color=colors[diag], width=2),
            name=f"KDE Diagnosis {diag}"
        ))
        fig.update_layout(
            title=f"Distribusi {col} untuk Diagnosis {diag}",
            annotations=[dict(
                x=0.5,
                y=1.1,
                xref="paper",
                yref="paper",
                text=f"**Skewness:** {skew_vals[diag]:.3f}",
                showarrow=False,
                font=dict(size=12)
            )],
            xaxis_title=col,
            yaxis_title="Density",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    # === Boxplot ===
    st.markdown(f"### Boxplot Variabel: **{col}**")
    fig = px.box(df, y=col, color="diagnosis",
                 template=PLOTLY_TEMPLATE,
                 color_discrete_sequence=["#65fbff", "#00bbff"],
                 title=f"Boxplot {col} berdasarkan Diagnosis")
    st.plotly_chart(fig, use_container_width=True)

# ================================
# ========= MODEL PAGE ===========
# ================================
elif menu == "Model dan Prediksi":
    st.title("Modeling Klasifikasi")
    # Training model
    model_choice = st.selectbox("Pilih Model:", ["Logistic Regression", "Random Forest", "XGBoost"])
    model_map = {
        "Logistic Regression": "LogisticRegression",
        "Random Forest": "RandomForest",
        "XGBoost": "XGBoost"
    }
    model_name = model_map[model_choice]

    # === Load pickle ===
    try:
        with open(f"model_{model_name}.pkl","rb") as f:
            pack = pickle.load(f)
    except:
        st.error(f"File model_{model_name}.pkl tidak ditemukan!")
        st.stop()
    pipe = pack["model_pipeline"]
    selected_features = pack["rfe_features"]
    rh_mean = pack["repeated_holdout"]
    rh_extra = pack["repeated_holdout_extra"]
    kf_mean = pack["kfold_mean"]
    kf_extra = pack["kfold_extra"]
    st.success(f"Model **{model_choice}** berhasil dimuat dari pickle.")
    
    st.markdown("### Fitur Terpilih oleh RFE:")
    st.write(f"Fitur terpilih ({len(selected_features)}):") 
    mid = len(selected_features) // 2 + len(selected_features) % 2 
    col1_features = selected_features[:mid] 
    col2_features = selected_features[mid:] 
    col1, col2 = st.columns(2)
    with col1:
        for f in col1_features:
            st.markdown(f"- <span style='color:#0487b6'>{f}</span>", unsafe_allow_html=True)
    with col2:
        for f in col2_features:
            st.markdown(f"- <span style='color:#0487b6'>{f}</span>", unsafe_allow_html=True)
    st.markdown("---")

    # ================================
    # Evaluasi Model pada Test Set
    # ================================
    st.subheader("Evaluasi Model pada Test Set")
    metrics = {
        "accuracy": kf_mean["accuracy"],
        "precision": kf_mean["precision"],
        "recall": kf_mean["recall"],
        "f1": kf_mean["f1"],
        "auc": kf_mean["auc"],
        "confusion": kf_extra["confusion"],
        "roc": kf_extra["roc"]
    }

    # ----------------------------
    # KPI Cards untuk Metrics
    # ----------------------------
    kpi_style = """
    <style>
    .kpi_card {
        background-color: transparent;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: none;
        border: 2px solid #0487b6;
        margin: 5px;
    }
    .kpi_title {
        font-size: 16px;
        margin-bottom: 5px;
    }
    .kpi_value {
        font-size: 24px;
        font-weight: bold;
        color: #0487b6;
    }
    </style>
    """
    st.markdown(kpi_style, unsafe_allow_html=True)
    # Buat kolom
    k1, k2, k3, k4 = st.columns(4)
    # Tampilkan metrics di dalam kotak
    k1.markdown(f"<div class='kpi_card'><div class='kpi_title'>Accuracy</div><div class='kpi_value'>{metrics['accuracy']*100:.2f}%</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi_card'><div class='kpi_title'>Precision</div><div class='kpi_value'>{metrics['precision']*100:.2f}%</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi_card'><div class='kpi_title'>Recall</div><div class='kpi_value'>{metrics['recall']*100:.2f}%</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi_card'><div class='kpi_title'>F1-score</div><div class='kpi_value'>{metrics['f1']*100:.2f}%</div></div>", unsafe_allow_html=True)
    # Tampilkan AUC jika ada
    if metrics.get("auc") is not None:
        st.markdown(f"**AUC:** {metrics['auc']:.3f}")
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix (Test Set)")
    cm = metrics["confusion"]
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred 0", "Pred 1"],
        y=["True 0", "True 1"],
        colorscale="Blues",
        texttemplate="%{z}",
        textfont=dict(size=20)
    ))
    cm_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        width=400,          # atur sama dengan height
        height=400,
        xaxis=dict(
            scaleanchor="y",
            side="top"
        ),
        yaxis=dict(
            scaleanchor="x",
            autorange='reversed'
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(cm_fig, use_container_width=False)
    
    # ROC Curve 
    if metrics.get("roc") is not None and metrics["roc"][0] is not None: 
        st.subheader("ROC Curve (test set)") 
        fpr, tpr, _ = metrics["roc"] 
        roc_fig = go.Figure() 
        roc_fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode='lines', 
                name='ROC Curve', line=dict(color='#0487b6', width=2))
            ) 
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='#9eaba2', width=2, dash='dash'))) 
        roc_fig.update_layout( 
            title="ROC Curve", 
            xaxis_title="False Positive Rate", 
            yaxis_title="True Positive Rate", 
            template=PLOTLY_TEMPLATE, 
            width=600, 
            height=500 
        ) 
        st.plotly_chart(roc_fig, use_container_width=False)

    # -------------------------
    # --- Prediksi Input Baru
    # -------------------------
    X = df.drop(columns=["diagnosis"])
    st.subheader("Prediksi Diagnosis Tumor Baru")
    
    metode = st.radio("Input data:", ("Form Input", "Upload CSV"))

    if metode == "Form Input":
        input_dict = {}
        for c in X.columns:
            val = st.number_input(c, value=float(X[c].mean()))
            input_dict[c] = val

        if st.button("Prediksi"):
            df_new = pd.DataFrame([input_dict])
            pred = pipe.predict(df_new)[0]
            label = "Malignant (Ganas)" if pred == 1 else "Benign (Jinak)"
            st.success(f"Hasil Prediksi: **{label}**")

    else:
        file = st.file_uploader("Upload CSV")
        if file:
            df_new = pd.read_csv(file)

            if not all(col in df_new.columns for col in selected_features):
                st.error("Kolom tidak sesuai fitur RFE!")
                st.stop()

            preds = pipe.predict(df_new)
            df_new["Prediksi"] = ["Malignant" if p==1 else "Benign" for p in preds]
            st.dataframe(df_new)
            st.success("Prediksi selesai!")
