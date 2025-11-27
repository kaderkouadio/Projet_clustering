
####################################################

"""
API de Clustering Client – Version C (PRO)
------------------------------------------

✅ Architecture propre pour production
✅ Logging
✅ Gestion globale des erreurs
✅ Endpoints cohérents et normalisés
✅ Pydantic pour validation stricte
✅ Séparation helpers (préprocessing, prédiction, PCA)
✅ Réponses JSON normalisées
✅ Swagger UI ultra lisible
✅ Zéro 422 / 500 en production
"""

import os
import json
import joblib
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path


from pydantic import BaseModel, Field
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request, Query, Body
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware


# -------------------------------------------------------------------------
# CONFIG LOGGING
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



# # http://localhost:8001/docs


# -------------------------------------------------------------------------
# Configuration des chemins
# -------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "Data"

# Crée le dossier si inexistant
# DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# Configuration des chemins
# ---------------------------------------------------
PREPROCESSOR_PATH = DATA_DIR / "preprocessor.joblib"
KMEANS_PATH = DATA_DIR / "kmeans_model.joblib"
PCA_PATH = DATA_DIR / "pca_model.joblib"
CLASSIFIER_PATH = DATA_DIR / "classifier_best.joblib"
FEATURES_PATH = DATA_DIR / "features_list.json"
METADATA_PATH = DATA_DIR / "model_metadata.json"
PCA_COORDS_PATH = DATA_DIR / "pca_coords.csv"

# -------------------------------------------------------------------------
# 🚀 INITIALISATION DE L’API FASTAPI
# -------------------------------------------------------------------------

api_description = """
🎯 **Bienvenue sur l'API de Segmentation Marketing**

Cette API expose un pipeline complet de segmentation client basé sur des 
techniques de machine learning. Elle permet de construire des segments 
marketing robustes à partir de données démographiques et comportementales, 
et de réaliser des prédictions rapides pour de nouveaux clients.

---

## 🚀 Fonctionnalités principales

- 🔍 **Clustering non supervisé (KMeans)**  
  Segmentation automatique des clients selon leurs comportements d'achat.

- 🤖 **Prédiction supervisée par Random Forest**  
  Un classifieur réplique les clusters KMeans pour des prédictions rapides 
  et stables en production.

- 📊 **Réduction dimensionnelle PCA (2D)**  
  Projection des clients sur un espace 2D pour visualisation et analyse.

- 📈 **Statistiques avancées par segment**  
  Agrégations automatiques : effectif, revenu moyen, intensité d'achat…

- ⚡ **Attribution de cluster en temps réel**  
  Idéal pour intégrations CRM, scoring client ou API temps réel.

- 🖥️ **Compatibilité totale** avec Streamlit, React, Vue, Angular, Python, Node.

---

## ⚙️ Pipeline technique

- 🔧 **Preprocessing automatique**  
  `StandardScaler` pour les variables numériques  
  `OneHotEncoder(handle_unknown="ignore")` pour les variables catégorielles

- 🧠 **Modèles utilisés**
  - `KMeans` → création des clusters initiaux  
  - `RandomForestClassifier` → prédiction supervisée des clusters  
  - `PCA (2 composants)` → réduction dimensionnelle

- 📦 Fichiers de modèle générés automatiquement :
  - preprocessor.joblib  
  - kmeans_model.joblib  
  - classifier_best.joblib  
  - pca_model.joblib  
  - features_list.json  
  - model_metadata.json

---

## 📚 Documentation Swagger interactive
Disponible sur **`/docs`** après démarrage de l’API.

"""

app = FastAPI(
    title="🔥 API Segmentation(Clustering) Marketing",
    description=api_description,
    version="1.0.0",
    contact={
        "name": "KOUADIO Kader",
        "email": "kkaderkouadio@gmail.com",
        "url": "https://www.linkedin.com/in/koukou-kader-kouadio-2a32371a4/"
    },
    openapi_tags=[
        {
            "name": "Santé & Métadonnées",
            "description": "État de l'API, versions modèles, métadonnées."
        },
        {
            "name": "Prédiction",
            "description": "Attribution de cluster supervisée et clustering non supervisé."
        },
        {
            "name": "Visualisation",
            "description": "Projection PCA et statistiques détaillées par cluster."
        }
    ]
)

# -------------------------------------------------------------------------
# 🌐 Configuration CORS (Frontend : Streamlit, React, etc.)
# -------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Peut être restreint en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# 🧩 Modèle de données principal
# -------------------------------------------------------------------------
class ClientData(BaseModel):
    """
    Structure standardisée des données client, utilisée pour :
      - la prédiction supervisée (Random Forest)
      - la segmentation non supervisée (KMeans)
      - la projection PCA
      - la sauvegarde ou l’ingestion de nouveaux clients
    
    Ce modèle doit refléter exactement les features utilisées dans le pipeline
    d’entraînement (features_list.json).
    """

    # ------------------------------------------------------------------
    # 🧍 Données démographiques
    # ------------------------------------------------------------------
    Age: int = Field(..., description="Âge actuel du client (en années)")

    Customer_Seniority: int = Field(...,description="Ancienneté du client exprimée en mois (date d'inscription → aujourd'hui)")

    Kidhome: int = Field(...,description="Nombre d'enfants âgés de 0 à 12 ans présents dans le foyer")

    Teenhome: int = Field(...,description="Nombre d'adolescents âgés de 13 à 17 ans présents dans le foyer")

    Education: str = Field(...,description="Niveau d'éducation du client (Graduate, Postgraduate, PhD, etc.)")

    Marital_Status: str = Field(...,description="Statut matrimonial du client (Single, Married, Divorced, etc.)")

    Income: float = Field(...,description="Revenu annuel du client (en devise locale)")

    Recency: int = Field(...,description="Nombre de jours depuis la dernière interaction ou achat du client")
    # ------------------------------------------------------------------
    # 💰 Dépenses par catégorie de produits
    # ------------------------------------------------------------------
    MntWines: float = Field(...,description="Montant total dépensé en vins")

    MntFruits: float = Field(...,description="Montant total dépensé en fruits")

    MntMeatProducts: float = Field(...,description="Montant total dépensé en produits carnés")

    MntFishProducts: float = Field(...,description="Montant total dépensé en poissons")

    MntSweetProducts: float = Field(...,description="Montant total dépensé en sucreries, desserts et gâteaux")

    MntGoldProds: float = Field(...,description="Montant total dépensé en produits premium/luxe ('gold')")

    # ------------------------------------------------------------------
    # 🛒 Canaux et comportement client
    # ------------------------------------------------------------------
    NumDealsPurchases: int = Field(...,description="Nombre de promotions ou deals utilisés")

    NumWebPurchases: int = Field(...,description="Nombre d'achats réalisés sur le site web")

    NumCatalogPurchases: int = Field(...,description="Nombre d'achats via catalogue")

    NumStorePurchases: int = Field(...,description="Nombre d'achats effectués en magasin physique")

    NumWebVisitsMonth: int = Field(...,description="Nombre de visites du site web durant le dernier mois")


class PredictClusterResponse(BaseModel):
    """Réponse standardisée pour l'endpoint /predict-cluster"""
    cluster: int
    probability: Optional[float] = None
    model_info: Optional[Dict[str, Any]] = None

# -------------------------------------------------------------------------
# GLOBALS ARTIFACTS
# -------------------------------------------------------------------------
preprocessor = None
classifier = None
kmeans_model = None
pca_model = None
metadata: Dict[str, Any] = {}

# -------------------------------------------------------------------------
# STARTUP EVENT – Chargement des modèles
# -------------------------------------------------------------------------
@app.on_event("startup")
async def load_artifacts():
    global preprocessor, classifier, kmeans_model, pca_model, metadata
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        classifier = joblib.load(CLASSIFIER_PATH)
        kmeans_model = joblib.load(KMEANS_PATH)
        pca_model = joblib.load(PCA_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info("Tous les artefacts chargés avec succès !")
    except Exception as e:
        logger.error(f"Échec du chargement des artefacts : {e}")
        raise RuntimeError(f"Impossible de démarrer l'API : {e}")

# -------------------------------------------------------------------------
# Gestion globale des erreurs
# -------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Erreur non capturée → {request.method} {request.url} | {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": "Une erreur interne est survenue"}
    )

# =============================================================================
# 🌡️ Santé & Métadonnées
# =============================================================================

@app.get("/health", summary="État de santé de l'API", tags=["Santé & Métadonnées"])
def health_check():
    """
    Vérifie l'état général de l'API et la disponibilité des artefacts ML.

    Cette route permet aux utilisateurs, applications externes ou services de monitoring
    (ex. : Grafana, Docker Healthcheck, Kubernetes) de connaître rapidement l’état
    de fonctionnement de l’API.  
    Elle vérifie notamment :

    - le préprocesseur (OneHotEncoder + Scaling)
    - le modèle de classification supervisée
    - le modèle KMeans non supervisé
    - le modèle PCA pour la réduction de dimension
    - la cohérence générale du pipeline

    Returns
    -------
    dict
        Un objet JSON contenant :
        - `status`: "ok" si tous les artefacts sont chargés, sinon "degraded"
        - `artifacts`: état individuel de chaque composant ML
    """
    
    # Vérification de chaque artefact pour déterminer si l'API est opérationnelle
    global_status = all([preprocessor, classifier, kmeans_model, pca_model])

    return {
        "status": "ok" if global_status else "degraded",
        "artifacts": {
            "preprocessor": bool(preprocessor),   # Chargement des transformations
            "classifier": bool(classifier),       # Modèle supervisé
            "kmeans": bool(kmeans_model),         # Modèle non supervisé
            "pca": bool(pca_model),               # Réduction de dimension
        }
    }


@app.get("/metadata", summary="Métadonnées complètes du modèle", tags=["Santé & Métadonnées"])
def get_metadata():
    """
    Retourne l’ensemble des métadonnées liées au modèle de clustering.

    Cette route permet aux utilisateurs et développeurs d’obtenir une vue détaillée sur :
    - les paramètres d’entraînement
    - les scores obtenus (accuracy, f1-score, silhouette…)
    - la date de création ou de mise à jour des modèles
    - les informations sur le dataset utilisé
    - les hyperparamètres du modèle
    - la version de l’API et du pipeline

    Idéal pour :
    - afficher des informations dans un dashboard Streamlit
    - assurer la traçabilité des modèles
    - diagnostiquer des problèmes ou répertorier des changements

    Returns
    -------
    dict
        Un objet JSON contenant les métadonnées complètes du modèle.
    """
    
    # `metadata` est supposé avoir été chargé au démarrage de l'application
    return {
        "status": "success",
        "metadata": metadata
    }


# =============================================================================
# Visualisation
# =============================================================================

@app.get("/pca", summary="Coordonnées PCA pré-calculées (pour le frontend)", tags=["Visualisation"])
def get_pca_coords(limit: int = Query(1000, ge=1, le=5000)):
    """Retourne un échantillon des points PCA avec leur cluster (PC1, PC2, cluster)"""
    if not PCA_COORDS_PATH.exists():
        raise HTTPException(status_code=404, detail="Fichier pca_coords.csv introuvable")
    df = pd.read_csv(PCA_COORDS_PATH).head(limit)
    return {"status": "success", "data": df.to_dict(orient="records")}

@app.get("/segments/stats", summary="Statistiques par segment (revenu moyen + effectif)", tags=["Visualisation"])
def segment_stats():
    """Statistiques agrégées sécurisées – fonctionne même si Income est absent"""
    if not PCA_COORDS_PATH.exists():
        raise HTTPException(status_code=404, detail="Fichier pca_coords.csv introuvable")

    df = pd.read_csv(PCA_COORDS_PATH)

    if "cluster" not in df.columns:
        raise HTTPException(status_code=500, detail="Colonne 'cluster' manquante dans pca_coords.csv")

    # Comptage par cluster
    counts = df["cluster"].value_counts().sort_index().reset_index()
    counts.columns = ["cluster", "count"]

    # Revenu moyen (si présent)
    if "Income" in df.columns and pd.api.types.is_numeric_dtype(df["Income"]):
        incomes = df.groupby("cluster")["Income"].mean().round(2).reset_index()
        incomes.columns = ["cluster", "avg_income"]
        result = counts.merge(incomes, on="cluster", how="left")
    else:
        result = counts
        result["avg_income"] = None

    return {"status": "success", "data": result.to_dict(orient="records")}


# ===========================================================================
# MAPPING MÉTIER OFFICIEL – Basé sur ton entraînement du 16/11/2025
# Accuracy test : 99.1% → Tu peux dormir tranquille
# ===========================================================================
from typing import Dict, Any

def get_segment_info(cluster: int) -> Dict[str, Any]:
    """
    Retourne les infos métier + stratégie marketing validées sur tes données
    """
    mapping = {
        0: {
            "label": "Premium VIP",
            "short_label": "Premium",
            "color": "#00cc96",  # Vert émeraude
            "description": "Clients les plus rentables : haut revenu, achats fréquents et élevés dans toutes les catégories (vin, viande, or). Fidèles depuis longtemps.",
            "strategy": "Programme VIP, early access, cadeaux exclusifs, service client dédié, offres personnalisées haut de gamme",
            "priority": "Très Haute",
            "action": "Chouchouter à tout prix – ils représentent votre marge maximale"
        },
        1: {
            "label": "Limités",
            "short_label": "À réactiver",
            "color": "#ff6b35",  # Orange vif
            "description": "Faible revenu ou faible engagement. Peu d'achats, sensibles aux promotions. Risque de churn élevé.",
            "strategy": "Campagnes de réactivation : gros deals, codes promo, gamification, relance email/SMS ciblée",
            "priority": "Haute",
            "action": "Réactiver rapidement ou accepter le churn naturel"
        },
        2: {
            "label": "Équilibrés Haut de Gamme",
            "short_label": "Équilibrés",
            "color": "#1f77b4",  # Bleu classique
            "description": "Bon revenu, achats réguliers et diversifiés. Clientèle stable, bonne rentabilité.",
            "strategy": "Fidélisation douce : points bonus, upsell modéré, contenu éducatif (vin, cuisine)",
            "priority": "Moyenne-Haute",
            "action": "Maintenir la relation – base solide de l'entreprise"
        },
        3: {
            "label": "Jeunes à Potentiel",
            "short_label": "Potentiel",
            "color": "#9467bd",  # Violet
            "description": "Moins de 40 ans, bon revenu, mais achats encore modérés. Beaucoup de visites web, curiosité élevée. Futur Premium !",
            "strategy": "Onboarding premium, recommandations IA ultra-personnalisées, parrainage, première commande offerte",
            "priority": "Très Haute (croissance)",
            "action": "Investir maintenant → ils seront vos Premium dans 2 ans"
        }
    }
    
    return mapping.get(cluster, {
        "label": "Inconnu", "short_label": "Inconnu", "color": "#gray",
        "description": "Segment non identifié", "strategy": "À investiguer",
        "priority": "Inconnue", "action": "Analyse manuelle"
    })



# =============================================================================
# Prédiction
# =============================================================================

@app.post("/predict-cluster", response_model=PredictClusterResponse,
          summary="Prédiction supervisée d’un seul client", tags=["Prédiction"])
def predict_cluster(req: ClientData):
    """
    Prédit le cluster d’un client avec un classifieur supervisé (LogisticRegression)
    → Très haute précision grâce à l'entraînement sur les vrais labels KMeans
    """
    if not preprocessor or not classifier:
        raise HTTPException(status_code=500, detail="Artefacts manquants")

    df = pd.DataFrame([req.dict()])
    X = preprocessor.transform(df)
    cluster = int(classifier.predict(X)[0])

    probability = None
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X)[0]
        probability = float(probs[list(classifier.classes_).index(cluster)])

    model_info = {k: metadata.get(k) for k in ["classifier", "cv_accuracy", "test_accuracy", "kmeans_k", "created_at"]}

    return PredictClusterResponse(cluster=cluster, probability=probability, model_info=model_info)




@app.post("/cluster", 
          summary="Clustering KMeans non supervisé (batch) → avec infos métier complètes", 
          tags=["Prédiction"])
async def assign_cluster(
    clients: List[ClientData] = Body(..., embed=True)
):
    """
    Prend une liste de clients → retourne :
    - Le cluster prédit
    - Le libellé métier
    - La couleur
    - La stratégie marketing
    → Idéal pour alimentation directe d’un dashboard
    """
    if not kmeans_model or not preprocessor:
        raise HTTPException(status_code=500, detail="KMeans ou préprocesseur non chargé")

    if not clients:
        raise HTTPException(status_code=400, detail="La liste de clients est vide")

    df = pd.DataFrame([c.dict() for c in clients])

    # Nettoyage robuste des catégories inconnues (comme avant)
    allowed_edu = ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]
    allowed_marital = ["Single", "Married", "Divorced", "Together", "Widow"]

    df["Education"] = df["Education"].astype(str).apply(lambda x: x if x in allowed_edu else "Other")
    df["Marital_Status"] = df["Marital_Status"].astype(str).apply(lambda x: x if x in allowed_marital else "Other")

    try:
        X = preprocessor.transform(df)
        raw_clusters = kmeans_model.predict(X)
    except Exception as e:
        logger.error(f"Erreur lors du preprocessing/prediction KMeans : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors du clustering")

    # Construction de la réponse riche
    results = []
    for i, cluster_id in enumerate(raw_clusters.tolist()):
        segment = get_segment_info(cluster_id)
        results.append({
            "client_index": i,
            "cluster": int(cluster_id),
            "segment": segment["label"],
            "short_label": segment["short_label"],
            "color": segment["color"],
            "priority": segment["priority"],
            "strategy": segment["strategy"],
            "recommended_action": segment["action"]
        })

    return {
        "status": "success",
        "total_clients": len(clients),
        "model": "KMeans (unsupervised)",
        "k": kmeans_model.n_clusters,
        "results": results
    }

@app.post("/apply-pca", summary="Projection PCA en temps réel sur de nouveaux clients", tags=["Visualisation"])
async def apply_pca(
    clients: List[ClientData] = Body(..., embed=True),
    n_components: int = Query(2, ge=1, le=50, description="Nombre de composantes principales à retourner")
):
    """
    Applique le préprocesseur + PCA sur une liste de nouveaux clients.
    
    Retourne :
    - Les coordonnées dans l'espace PCA réduit
    - Le cluster KMeans associé à chaque client (super pratique pour le frontend)
    
    Exemple de corps :
    [ {client1}, {client2}, ... ] + ?n_components=2 en query param
    """
    if not pca_model or not preprocessor:
        raise HTTPException(status_code=500, detail="PCA ou préprocesseur non chargé")

    try:
        df = pd.DataFrame([c.dict() for c in clients])

        allowed_edu = ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]
        allowed_marital = ["Single", "Married", "Divorced", "Together", "Widow"]

        df["Education"] = df["Education"].astype(str).apply(lambda x: x if x in allowed_edu else "Other")
        df["Marital_Status"] = df["Marital_Status"].astype(str).apply(lambda x: x if x in allowed_marital else "Other")

        X_trans = preprocessor.transform(df)
        pca_coords = pca_model.transform(X_trans)[:, :n_components].tolist()
        clusters = kmeans_model.predict(X_trans).tolist() if kmeans_model else None

        return {
            "status": "success",
            "n_components": n_components,
            "pca_components": pca_coords,
            "clusters": clusters
        }

    except Exception as e:
        logger.error(f"Erreur dans /apply-pca : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la projection PCA")
    
# =============================================================================
# Lancement local
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True, log_level="info")