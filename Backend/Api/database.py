

# ============================================================
# database.py
# Création DB + tables automatiquement

# pip install psycopg2
# ============================================================

# import os
# import psycopg2
# from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
# from sqlalchemy import create_engine, Column, Integer, Float, String
# from sqlalchemy.orm import declarative_base, sessionmaker

# # -----------------------------
# # Configuration PostgreSQL
# # -----------------------------
# POSTGRES_USER = os.getenv("POSTGRES_USER", "KOUADIO_KADER")
# POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "Nantessou88")
# POSTGRES_DB = os.getenv("POSTGRES_DB", "clustering_db")
# POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
# POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# # -----------------------------
# # Création DB si elle n'existe pas
# # -----------------------------
# def create_database_if_not_exists():
#     try:
#         conn = psycopg2.connect(
#             dbname="postgres",
#             user=POSTGRES_USER,
#             password=POSTGRES_PASSWORD,
#             host=POSTGRES_HOST,
#             port=POSTGRES_PORT
#         )
#         conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
#         cursor = conn.cursor()
#         cursor.execute(f"SELECT 1 FROM pg_database WHERE datname='{POSTGRES_DB}';")
#         if not cursor.fetchone():
#             cursor.execute(f"CREATE DATABASE {POSTGRES_DB};")
#             print(f"Base '{POSTGRES_DB}' créée avec succès.")
#         else:
#             print(f"Base '{POSTGRES_DB}' existe déjà.")
#         cursor.close()
#         conn.close()
#     except Exception as e:
#         print(f"Erreur lors de la création de la base : {e}")

# # ------------------------------------------------------------
# # Création de la base avant SQLAlchemy
# # ------------------------------------------------------------
# create_database_if_not_exists()

# # ------------------------------------------------------------
# # Chaîne de connexion SQLAlchemy
# # ------------------------------------------------------------
# DATABASE_URL = (
#     f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
#     f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
# )

# # ------------------------------------------------------------
# # Moteur SQLAlchemy
# # ------------------------------------------------------------
# engine = create_engine(
#     DATABASE_URL,
#     echo=False,   # True pour debug SQL
#     future=True
# )

# # ------------------------------------------------------------
# # Session
# # ------------------------------------------------------------
# SessionLocal = sessionmaker(
#     bind=engine,
#     autocommit=False,
#     autoflush=False,
#     future=True
# )

# # ------------------------------------------------------------
# # Base pour les modèles
# # ------------------------------------------------------------
# Base = declarative_base()


# # -----------------------------
# # Modèle Client
# # -----------------------------
# class Client(Base):
#     __tablename__ = "clients"
    
#     id = Column(Integer, primary_key=True, index=True)
#     Age = Column(Float, nullable=False)
#     Customer_Seniority = Column(Float, nullable=False)
#     Income = Column(Float, nullable=False)
#     Kidhome = Column(Integer, nullable=False)
#     Teenhome = Column(Integer, nullable=False)
#     Recency = Column(Integer, nullable=False)
#     MntWines = Column(Float, nullable=False)
#     MntFruits = Column(Float, nullable=False)
#     MntMeatProducts = Column(Float, nullable=False)
#     MntFishProducts = Column(Float, nullable=False)
#     MntSweetProducts = Column(Float, nullable=False)
#     MntGoldProds = Column(Float, nullable=False)
#     NumDealsPurchases = Column(Integer, nullable=False)
#     NumWebPurchases = Column(Integer, nullable=False)
#     NumCatalogPurchases = Column(Integer, nullable=False)
#     NumStorePurchases = Column(Integer, nullable=False)
#     NumWebVisitsMonth = Column(Integer, nullable=False)
#     Education = Column(String, nullable=True)
#     Marital_Status = Column(String, nullable=True)
#     cluster = Column(Integer, nullable=True)

# # -----------------------------
# # Création automatique des tables
# # -----------------------------
# def create_tables():
#     Base.metadata.create_all(engine)
#     print("Toutes les tables ont été créées (si elles n'existaient pas).")

# create_tables()


# # ------------------------------------------------------------
# # Dépendance FastAPI
# # ------------------------------------------------------------
# def get_db():
#     """Ouvre une session DB et la ferme automatiquement."""
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()



##############################################################


# ============================================================
# database.py
# Création DB + tables automatiquement (si nécessaire)
## PostgreSQL hébergé sur Render
# ============================================================

import os
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import declarative_base, sessionmaker

# ------------------------------------------------------------
# Configuration PostgreSQL via variables d'environnement Render
# ------------------------------------------------------------
POSTGRES_USER = os.getenv("POSTGRES_USER", "clustering_db_b60a_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "5GO6AO4Io5PpcIV2ebHsxVjSBqhHNirO")
POSTGRES_DB = os.getenv("POSTGRES_DB", "clustering_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "dpg-d4k9mee3jp1c738mrqvg-a.frankfurt-postgres.render.com")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# ------------------------------------------------------------
# Chaîne de connexion SQLAlchemy
# ------------------------------------------------------------
DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"
    f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ------------------------------------------------------------
# Moteur SQLAlchemy
# ------------------------------------------------------------
engine = create_engine(
    DATABASE_URL,
    echo=False,   # True pour debug SQL
    future=True
)

# ------------------------------------------------------------
# Session
# ------------------------------------------------------------
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    future=True
)

# ------------------------------------------------------------
# Base pour les modèles
# ------------------------------------------------------------
Base = declarative_base()

# ------------------------------------------------------------
# Modèle Client
# ------------------------------------------------------------
class Client(Base):
    __tablename__ = "clients"
    
    id = Column(Integer, primary_key=True, index=True)
    Age = Column(Float, nullable=False)
    Customer_Seniority = Column(Float, nullable=False)
    Income = Column(Float, nullable=False)
    Kidhome = Column(Integer, nullable=False)
    Teenhome = Column(Integer, nullable=False)
    Recency = Column(Integer, nullable=False)
    MntWines = Column(Float, nullable=False)
    MntFruits = Column(Float, nullable=False)
    MntMeatProducts = Column(Float, nullable=False)
    MntFishProducts = Column(Float, nullable=False)
    MntSweetProducts = Column(Float, nullable=False)
    MntGoldProds = Column(Float, nullable=False)
    NumDealsPurchases = Column(Integer, nullable=False)
    NumWebPurchases = Column(Integer, nullable=False)
    NumCatalogPurchases = Column(Integer, nullable=False)
    NumStorePurchases = Column(Integer, nullable=False)
    NumWebVisitsMonth = Column(Integer, nullable=False)
    Education = Column(String, nullable=True)
    Marital_Status = Column(String, nullable=True)
    cluster = Column(Integer, nullable=True)

# ------------------------------------------------------------
# Création automatique des tables
# ------------------------------------------------------------
def create_tables():
    Base.metadata.create_all(engine)
    print("Toutes les tables ont été créées (si elles n'existaient pas).")

create_tables()

# ------------------------------------------------------------
# Dépendance FastAPI
# ------------------------------------------------------------
def get_db():
    """Ouvre une session DB et la ferme automatiquement."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
