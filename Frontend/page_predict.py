# pages/2_Prediction_&_Recommandations.py
import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
from utils import load_all_artifacts, get_data_dir

# ==================================================================
# STYLE & TITRE
# ==================================================================
html_temp = """
<div style="text-align:center; padding:30px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius:15px; margin-bottom:30px;">
    <h1 style="color:white; margin:0;">Prédiction du Segment Client & Recommandations Marketing</h1>
    <p style="color:#f0f0f0; font-size:1.1rem; margin-top:10px;">
        Architecture full-stack complète : ML + FastAPI + Streamlit + (BDD future)
    </p>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# ==================================================================
# CHOIX DU MODE
# ==================================================================
col1, col2 = st.columns([1, 2])
with col1:
    mode = st.radio(
        "Mode de prédiction",
        ["API FastAPI (recommandé)", "Modèle local (artefacts)"],
        horizontal=True,
        help="API = scalable en prod | Local = rapide en dev"
    )

api_url = ""
if mode == "API FastAPI (recommandé)":
    with col2:
        api_url = st.text_input(
            "URL de l'API FastAPI",
            value=st.session_state.get("api_url", "http://localhost:8001"),
            help="Ex: http://localhost:8001"
        ).strip().rstrip("/")
        st.session_state.api_url = api_url
        if not api_url:
            st.warning("Entre l’URL de ton API pour continuer")
            st.stop()
        if not api_url.startswith("http"):
            st.error("L’URL doit commencer par http:// ou https://")
            st.stop()

# Mapping des segments (centralisé !)
SEGMENTS = {
    0: {
        "label": "Premium VIP",
        "color": "#00cc96",
        "emoji": "crown",
        "priority": "ÉLEVÉE",
        "description": "Clients haut de gamme, très rentables, fidèles et dépensiers.",
        "strategy": "Fidélisation premium • Accès anticipé • Offres exclusives • Club VIP",
        "actions": ["Offre personnalisée haut de gamme", "Invitation événement exclusif", "Cadeau anniversaire luxe"],
        "warning": None
    },
    1: {
        "label": "À Réactiver",
        "color": "#ff6b35",
        "emoji": "alarm",
        "priority": "URGENT",
        "description": "Clients inactifs ou en perte de vitesse. Risque de churn élevé.",
        "strategy": "Campagne de réactivation • Promo flash • Relance personnalisée",
        "actions": ["Offre -30% valable 7 jours", "Email 'Vous nous manquez'", "Bonus fidélité doublé"],
        "warning": "Client dormant détecté"
    },
    2: {
        "label": "Équilibrés",
        "color": "#1f77b4",
        "emoji": "bar_chart",
        "priority": "MOYENNE",
        "description": "Clients réguliers, comportement stable et prévisible.",
        "strategy": "Croissance progressive • Cross-selling • Programme de fidélité",
        "actions": ["Suggestion produits complémentaires", "Points fidélité x2", "Offre groupée"],
        "warning": None
    },
    3: {
        "label": "Jeunes Potentiel",
        "color": "#9467bd",
        "emoji": "rocket",
        "priority": "ÉLEVÉE",
        "description": "Jeunes clients à fort potentiel de croissance future.",
        "strategy": "Acquisition & éducation • Offres attractives • Gamification",
        "actions": ["Bienvenue : 20% sur premier achat >100€", "Programme parrainage", "Contenu éducatif"],
        "warning": None
    }
}

# === FORMULAIRE CLIENT COMPLET (TOUTES LES FEATURES) ===
with st.form("client_form", clear_on_submit=True):
    st.markdown("### Informations du client")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Informations personnelles**")
        age = st.number_input("Âge", 18, 100, 40)
        year_birth = 2025 - age  # pour compatibilité ancien code
        education = st.selectbox("Niveau d'éducation", 
            ["Basic", "2n Cycle", "Graduation", "Master", "PhD"], index=2)
        marital = st.selectbox("Statut marital", 
            ["Single", "Together", "Married", "Divorced", "Widow", "Alone", "Absurd", "YOLO"], index=2)
        income = st.number_input("Revenu annuel (€)", 0, 200000, 60000, step=1000)

    with col2:
        st.markdown("**Composition foyer**")
        kidhome = st.selectbox("Nb enfants < 12 ans", [0, 1, 2, 3], 0)
        teenhome = st.selectbox("Nb ados à la maison", [0, 1, 2], 0)

        st.markdown("**Activité récente**")
        recency = st.slider("Jours depuis dernier achat", 0, 120, 30)
        web_visits = st.slider("Visites web / mois", 0, 20, 6)

    with col3:
        st.markdown("**Dépenses par catégorie (2 dernières années)**")
        mnt_wines = st.slider("Vin", 0, 2000, 300)
        mnt_fruits = st.slider("Fruits", 0, 300, 30)
        mnt_meat = st.slider("Viande", 0, 2000, 170)
        mnt_fish = st.slider("Poisson", 0, 300, 40)
        mnt_sweet = st.slider("Sucreries", 0, 300, 30)
        mnt_gold = st.slider("Produits or (bijoux)", 0, 500, 40)

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        web_purchases = st.slider("Achats sur le web", 0, 30, 8)
        catalog_purchases = st.slider("Achats catalogue", 0, 30, 2)
    with col2:
        store_purchases = st.slider("Achats en magasin", 0, 20, 6)
        deals_purchases = st.slider("Achats avec promo", 0, 20, 3)
    with col3:
        accepted_cmp1 = st.checkbox("Accepté campagne 1", False)
        accepted_cmp2 = st.checkbox("Accepté campagne 2", False)
        accepted_cmp3 = st.checkbox("Accepté campagne 3", True)
    with col4:
        accepted_cmp4 = st.checkbox("Accepté campagne 4", False)
        accepted_cmp5 = st.checkbox("Accepté campagne 5", False)
        complain = st.checkbox("A porté plainte", False)

    submitted = st.form_submit_button("Prédire le segment & recommandations", 
                                      type="primary", use_container_width=True)

if submitted:
    # === PAYLOAD COMPLET (24+ features) ===
    payload = {
        "Year_Birth": year_birth,
        "Education": education,
        "Marital_Status": marital,
        "Income": float(income),
        "Kidhome": int(kidhome),
        "Teenhome": int(teenhome),
        "Recency": int(recency),
        "MntWines": int(mnt_wines),
        "MntFruits": int(mnt_fruits),
        "MntMeatProducts": int(mnt_meat),
        "MntFishProducts": int(mnt_fish),
        "MntSweetProducts": int(mnt_sweet),
        "MntGoldProds": int(mnt_gold),
        "NumDealsPurchases": int(deals_purchases),
        "NumWebPurchases": int(web_purchases),
        "NumCatalogPurchases": int(catalog_purchases),
        "NumStorePurchases": int(store_purchases),
        "NumWebVisitsMonth": int(web_visits),
        "AcceptedCmp1": int(accepted_cmp1),
        "AcceptedCmp2": int(accepted_cmp2),
        "AcceptedCmp3": int(accepted_cmp3),
        "AcceptedCmp4": int(accepted_cmp4),
        "AcceptedCmp5": int(accepted_cmp5),
        "Complain": int(complain),
        # Features dérivées (si ton preprocessor les attend)
        "Age": age,
        "Total_Spending": mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweet + mnt_gold,
        "Total_Purchases": web_purchases + catalog_purchases + store_purchases,
        "Customer_Seniority": max(100, 2025 - year_birth - 18) * 10  # estimation
    }

    st.success("Payload complet généré (toutes les features requises)")
    st.json(payload, expanded=False)

    with st.spinner("Analyse en cours..."):
        predicted_cluster = None
        confidence = None

        # === PRÉDICTION ===
        if mode == "API FastAPI (recommandé)":
            try:
                endpoint = f"{api_url.rstrip('/')}/predict-cluster"
                r = requests.post(endpoint, json=payload, timeout=15)
                r.raise_for_status()
                result = r.json()
                predicted_cluster = result.get("predicted_cluster") or result.get("cluster")
                confidence = result.get("confidence", result.get("probability", 0.95))
            except Exception as e:
                st.error(f"Erreur API : {e}")
                if 'r' in locals():
                    st.code(r.text)
                st.stop()

        else:  # Mode local
            artifacts = load_all_artifacts(get_data_dir())
            if not artifacts.get("classifier") and not artifacts.get("kmeans"):
                st.error("Modèles locaux manquants")
                st.stop()

            df_client = pd.DataFrame([payload])
            preprocessor = artifacts["preprocessor"]
            X = preprocessor.transform(df_client)

            if artifacts.get("classifier"):
                pred = artifacts["classifier"].predict(X)[0]
                proba = artifacts["classifier"].predict_proba(X)[0]
                predicted_cluster = int(pred)
                confidence = float(proba.max())
            else:
                predicted_cluster = int(artifacts["kmeans"].predict(X)[0])
                confidence = 0.98  # approximation

        # === AFFICHAGE RÉSULTAT ===
        if predicted_cluster is not None:
            segment = SEGMENTS.get(predicted_cluster, SEGMENTS[3])
            st.balloons()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                <div style="background:{segment['color']}; padding:30px; border-radius:20px; text-align:center; box-shadow: 0 8px 20px rgba(0,0,0,0.2);">
                    <h1 style="color:white; margin:0; font-size:48px;">{segment['emoji']} {segment['label']}</h1>
                    <p style="color:white; font-size:20px; margin:10px 0 0;">Confiance : <strong>{confidence:.1%}</strong></p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"### Priorité : **{segment['priority']}**")
                st.write(segment["description"])
                st.success(f"**Stratégie recommandée** : {segment['strategy']}")

                st.markdown("#### Actions concrètes à lancer")
                for action in segment["actions"]:
                    st.markdown(f"- {action}")

                if segment.get("warning"):
                    st.error(f"ALERTE : {segment['warning']}")

            # === OPTION : Afficher le client dans le nuage PCA (si artefacts locaux) ===
            if mode == "Modèle local (artefacts)":
                try:
                    pca_coords = artifacts.get("pca_coords")
                    pca_model = artifacts.get("pca")
                    if pca_coords is not None and pca_model is not None:
                        client_pca = pca_model.transform(X)
                        df_plot = pd.DataFrame(pca_coords)
                        df_client_plot = pd.DataFrame({
                            "PC1": [client_pca[0][0]],
                            "PC2": [client_pca[0][1]],
                            "cluster": [predicted_cluster]
                        })

                        import plotly.express as px
                        fig = px.scatter(df_plot, x="PC1", y="PC2", color="cluster", opacity=0.5,
                                         color_discrete_map={k: v["color"] for k, v in SEGMENTS.items()})
                        fig.add_scatter(x=df_client_plot["PC1"], y=df_client_plot["PC2"],
                                        mode="markers", marker=dict(size=20, color=segment["color"], symbol="star"),
                                        name="Nouveau client")

                        st.plotly_chart(fig, use_container_width=True)
                except:
                    pass



st.markdown("---")

footer_html = """
<style>
.footer-container {
    text-align:center;
    margin-top:40px;
    padding:20px 10px;
    color:#4a4a4a;
    font-family: 'Segoe UI', sans-serif;
}

.footer-name {
    font-size:22px;
    font-weight:700;
    color:#222;
}

.footer-badge {
    display:inline-block;
    background:#1a73e8;
    color:white;
    padding:3px 10px;
    border-radius:12px;
    font-size:13px;
    margin-left:8px;
    font-weight:600;
}

.footer-role {
    font-size:15px;
    margin-top:6px;
    color:#333;
}

.footer-sub {
    font-size:13px;
    margin-top:4px;
    color:#777;
}
</style>

<div class="footer-container">
    <span class="footer-name"> KOUADIO Kader</span>
    <span class="footer-badge">✔ Vérifié</span>
    <div class="footer-role">
        Économiste • Analyste Financier • Data Analyst • Développeur BI & Intelligence Artificielle
    </div>
    <div class="footer-sub">© 2025 – Projet complet open-source</div>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)

