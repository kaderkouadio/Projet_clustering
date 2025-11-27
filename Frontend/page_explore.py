# page_explore.py
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import os
from utils import load_all_artifacts

# ==================================================================
# CONFIGURATION GÉNÉRALE
# ==================================================================
st.set_page_config(page_title="Segmentation & Exploration", layout="wide")

# ==================================================================
# CONSTANTES
# ==================================================================
LABEL_MAP = {0: "Premium VIP", 1: "À réactiver", 2: "Équilibrés", 3: "Jeunes Potentiel"}
COLOR_MAP = {0: "#00cc96", 1: "#ff6b35", 2: "#1f77b4", 3: "#9467bd"}
COLOR_SEQUENCE = ["#00cc96", "#ff6b35", "#1f77b4", "#9467bd"]

# ==================================================================
# HEADER / FOOTER
# ==================================================================
def display_header():
    html_temp = """
    <div style="text-align:center; padding:30px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius:15px; margin-bottom:30px;">
        <h1 style="color:white; margin:0;">Visualisation des Segments Clients</h1>
        <p style="color:#f0f0f0; font-size:1.1rem; margin-top:10px;">
            Architecture full-stack complète : ML + FastAPI + Streamlit + (BDD future)
        </p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

def display_footer():
    footer_html = """
    <style>
    .footer-container {text-align:center; margin-top:40px; padding:20px 10px; color:#4a4a4a; font-family: 'Segoe UI', sans-serif;}
    .footer-name {font-size:22px; font-weight:700; color:#222;}
    .footer-badge {display:inline-block; background:#1a73e8; color:white; padding:3px 10px; border-radius:12px; font-size:13px; margin-left:8px; font-weight:600;}
    .footer-role {font-size:15px; margin-top:6px; color:#333;}
    .footer-sub {font-size:13px; margin-top:4px; color:#777;}
    </style>
    <div class="footer-container">
        <span class="footer-name">KOUADIO Kader</span>
        <span class="footer-badge">✔ Vérifié</span>
        <div class="footer-role">Économiste • Analyste Financier • Data Analyst • Développeur BI & Intelligence Artificielle</div>
        <div class="footer-sub">© 2025 – Projet complet open-source</div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# ==================================================================
# FONCTIONS UTILES
# ==================================================================
def load_local_artifacts():
    artifacts = load_all_artifacts()
    df = artifacts.get("pca_coords")
    if df is None or df.empty:
        raise ValueError("pca_coords.csv introuvable ou vide")
    df["Segment"] = df["cluster"].map(LABEL_MAP)
    return df

def load_api_data(api_url):
    resp = requests.get(f"{api_url.rstrip('/')}/pca", params={"limit": 3000}, timeout=20)
    resp.raise_for_status()
    data = resp.json()["data"]
    df = pd.DataFrame(data)
    df["Segment"] = df["cluster"].map(LABEL_MAP)
    return df

@st.cache_data(show_spinner=False)
def load_uploaded_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def display_pca(df):
    fig = px.scatter(df, x="PC1", y="PC2", color="Segment", color_discrete_map=COLOR_MAP,
                     hover_data={col: True for col in df.columns if col not in ["PC1","PC2"]},
                     size_max=8, opacity=0.8, title="Segmentation Client - Projection PCA")
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
    st.plotly_chart(fig, use_container_width=True)

def display_segment_distribution(df):
    counts = df["Segment"].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(values=counts.values, names=counts.index, color_discrete_sequence=COLOR_SEQUENCE,
                     title="Répartition des segments")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index, color_discrete_sequence=COLOR_SEQUENCE,
                     text=counts.values, title="Nombre par segment")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

def display_profiles(df):
    cols = [c for c in df.columns if c not in ["PC1","PC2","cluster","Segment"]]
    if cols:
        profile = df.groupby("Segment")[cols].mean().round(1)
        profile = profile.reindex(LABEL_MAP.values())
        st.dataframe(profile.style.background_gradient(cmap="Blues"), use_container_width=True)
    else:
        st.info("Variables descriptives non présentes (ajoute-les dans pca_coords.csv)")

# ==================================================================
# MODE SEGMENTATION
# ==================================================================
def mode_segmentation():
    st.markdown("<div class='medium-font'>Segmentation Clients</div>", unsafe_allow_html=True)
    submode = st.radio("Source des données", ["Local (artefacts)", "API FastAPI"], horizontal=True, key="seg_source")

    df = None
    if submode == "Local (artefacts)":
        try:
            with st.spinner("Chargement des artefacts locaux..."):
                df = load_local_artifacts()
                st.success(f"{len(df):,} clients chargés localement")
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.stop()
    else:
        api_url = st.text_input("URL FastAPI", value=os.getenv("FASTAPI_URL", "http://localhost:8001"), key="api_seg")
        if st.button("Charger depuis l'API", type="primary"):
            try:
                with st.spinner("Connexion API..."):
                    df = load_api_data(api_url)
                    st.session_state.seg_df = df
                    st.success(f"{len(df):,} clients chargés depuis l'API")
            except Exception as e:
                st.error(f"Erreur API : {e}")
                st.stop()
        if "seg_df" in st.session_state:
            df = st.session_state.seg_df

    if df is not None:
        tab1, tab2, tab3 = st.tabs(["Nuage PCA", "Répartition", "Profils moyens"])
        with tab1:
            display_pca(df)
        with tab2:
            display_segment_distribution(df)
        with tab3:
            display_profiles(df)

# ==================================================================
# MODE EXPLORATION LIBRE
# ==================================================================
def mode_exploration():
    st.markdown("<div class='medium-font'>Explorateur de Données Libre</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Chargez votre fichier CSV ou Excel", type=["csv", "xlsx", "xls"], help="Max 200 Mo")
    if uploaded_file:
        df = load_uploaded_file(uploaded_file)
        st.success(f"Chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

        # Nettoyage léger
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes('object').columns:
            if df[col].nunique() < 50:
                df[col] = df[col].astype('category')

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = [c for c in df.columns if df[c].dtype.name == 'category' or df[c].nunique() < 20]

        # Tabs
        tab_a, tab_b, tab_c, tab_d = st.tabs(["KPIs", "Vue d'ensemble", "Univariée", "Bivariée"])

        # ---------------- KPIs ----------------
        with tab_a:
            st.subheader("KPIs personnalisés")
            with st.form("kpi_form"):
                c1, c2, c3 = st.columns(3)
                metric = c1.selectbox("Métrique principale", [None] + numeric_cols)
                date_col = c2.selectbox("Colonne date", [None] + df.columns.tolist())
                dim = c3.selectbox("Dimension à compter", [None] + df.columns.tolist())
                submitted = st.form_submit_button("Afficher KPIs")

            if submitted and metric:
                cols = st.columns(4)
                cols[0].metric(f"Total {metric}", f"{df[metric].sum():,.0f}")
                cols[1].metric(f"Moyenne {metric}", f"{df[metric].mean():,.1f}")
                if dim:
                    cols[2].metric(f"{dim} uniques", f"{df[dim].nunique():,}")
                if date_col:
                    try:
                        temp = df[[date_col, metric]].copy()
                        temp[date_col] = pd.to_datetime(temp[date_col], errors='coerce')
                        temp = temp.dropna().set_index(date_col).sort_index()
                        monthly = temp.resample('M').sum()
                        fig = px.line(monthly, title=f"Évolution de {metric}")
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.warning("Impossible de créer l'évolution temporelle avec cette colonne date")

        # ---------------- Vue d'ensemble ----------------
        with tab_b:
            st.subheader("Vue d'ensemble")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Lignes", f"{len(df):,}")
            c2.metric("Colonnes", len(df.columns))
            c3.metric("Complétude", f"{(df.notna().sum().sum()/df.size*100):.1f}%")
            c4.metric("Doublons", f"{df.duplicated().sum():,}")

            col1, col2 = st.columns([1,2])
            with col1:
                types = df.dtypes.value_counts()
                fig = px.pie(values=types.values, names=types.index.astype(str), hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                miss = df.isnull().sum()
                miss = miss[miss>0].sort_values(ascending=False)
                if not miss.empty:
                    fig = px.bar(x=miss.index, y=miss.values, title="Valeurs manquantes")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("Aucune valeur manquante !")

        # ---------------- Univariée ----------------
        with tab_c:
            st.subheader("Analyse univariée")
            if numeric_cols:
                col = st.selectbox("Numérique", numeric_cols, key="uni_num")
                fig = px.histogram(df, x=col, marginal="box", title=f"Distribution de {col}")
                st.plotly_chart(fig, use_container_width=True)
            if cat_cols:
                col = st.selectbox("Catégorielle", cat_cols, key="uni_cat")
                top = df[col].value_counts().head(20)
                fig = px.bar(x=top.index, y=top.values, title=f"Top 20 - {col}")
                st.plotly_chart(fig, use_container_width=True)

        # ---------------- Bivariée ----------------
        with tab_d:
            st.subheader("Analyse bivariée")
            type_analysis = st.radio("Type", ["Num vs Num", "Num vs Cat"], horizontal=True)
            if type_analysis == "Num vs Num" and len(numeric_cols)>=2:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
                st.plotly_chart(fig, use_container_width=True)
                x = st.selectbox("X", numeric_cols)
                y = st.selectbox("Y", numeric_cols, index=1)
                fig = px.scatter(df, x=x, y=y, trendline="ols", marginal_x="histogram", marginal_y="histogram")
                st.plotly_chart(fig, use_container_width=True)
            elif type_analysis == "Num vs Cat" and numeric_cols and cat_cols:
                num = st.selectbox("Numérique", numeric_cols)
                cat = st.selectbox("Catégorielle", cat_cols)
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.box(df, x=cat, y=num, color=cat)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.violin(df, x=cat, y=num, color=cat)
                    st.plotly_chart(fig, use_container_width=True)

# ==================================================================
# MAIN
# ==================================================================
display_header()

mode = st.radio(
    "Choisissez votre mode d'utilisation",
    ["Segmentation Clients (modèle entraîné)", "Exploration libre (upload CSV/Excel)"],
    horizontal=True,
    key="main_mode"
)

if mode == "Segmentation Clients (modèle entraîné)":
    mode_segmentation()
else:
    mode_exploration()

st.markdown("---")
display_footer()
