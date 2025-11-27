# """
# test_db.py
# -----------

# Script de vérification de la base PostgreSQL :

# 1. Connexion à la base
# 2. Vérification des tables
# 3. Aperçu des données insérées
# """

# from database import engine, Base
# from sqlalchemy import inspect, text
# from sqlalchemy.orm import sessionmaker
# from database import Client

# # Création session
# SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
# session = SessionLocal()

# print("⏳ Test de connexion à PostgreSQL...")

# try:
#     with engine.connect() as conn:
#         # Test simple
#         now = conn.execute(text("SELECT NOW()")).scalar()
#         print(f"✅ Connexion OK ! Horodatage PostgreSQL : {now}")
# except Exception as e:
#     print("❌ Erreur de connexion :", e)


# # -----------------------------
# # Vérification des tables
# # -----------------------------
# inspector = inspect(engine)
# tables = inspector.get_table_names()
# print("\n📋 Tables existantes dans la base :")
# if tables:
#     for t in tables:
#         print(" -", t)
# else:
#     print("Aucune table trouvée !")

# # -----------------------------
# # Vérification du contenu d'une table
# # -----------------------------
# def show_table_sample(table_class, limit=5):
#     print(f"\n🔹 Aperçu des données de la table '{table_class.__tablename__}' (max {limit} lignes)")
#     rows = session.query(table_class).limit(limit).all()
#     if not rows:
#         print("Aucune donnée trouvée.")
#         return
#     for row in rows:
#         print(row.__dict__)

# # Exemple pour table Client
# show_table_sample(Client, limit=5)

# # -----------------------------
# # Compte total de lignes
# # -----------------------------
# def count_rows(table_class):
#     total = session.query(table_class).count()
#     print(f"\nNombre total de lignes dans '{table_class.__tablename__}' : {total}")

# count_rows(Client)

# # Fermeture session
# session.close()





"""
full_test_db.py
----------------

Script complet de vérification de la base PostgreSQL :

- Connexion à la DB
- Vérification de toutes les tables et colonnes
- Aperçu des données
- Compte total des lignes
"""

from database import engine, Base, SessionLocal
from sqlalchemy import inspect, text



# -----------------------------
# Test de connexion
# -----------------------------
print("⏳ Test de connexion à PostgreSQL...")
try:
    # with engine.connect() as conn:
    #     now = conn.execute("SELECT NOW()").scalar()
    #     print(f"✅ Connexion OK ! Horodatage PostgreSQL : {now}")
    with engine.connect() as conn:
        now = conn.execute(text("SELECT NOW()")).scalar()
        print(f"✅ Connexion OK ! Horodatage PostgreSQL : {now}")
except Exception as e:
    print("❌ Erreur de connexion :", e)

# -----------------------------
# Inspecteur SQLAlchemy
# -----------------------------
inspector = inspect(engine)

# Récupération des tables déclarées dans SQLAlchemy
declared_tables = Base.metadata.tables.keys()
existing_tables = inspector.get_table_names()

print("\n📋 Tables existantes dans la base :")
for t in declared_tables:
    if t in existing_tables:
        print(f"✅ Table '{t}' trouvée")
    else:
        print(f"❌ Table '{t}' manquante !")

# -----------------------------
# Vérification colonnes et données
# -----------------------------
def check_table(table_class, limit=5):
    print(f"\n🔹 Vérification de la table '{table_class.__tablename__}'")
    table_name = table_class.__tablename__

    # Colonnes
    columns = [c.name for c in table_class.__table__.columns]
    print("Colonnes :", columns)

    # Données
    session = SessionLocal()
    rows = session.query(table_class).limit(limit).all()
    if rows:
        print(f"Aperçu des {len(rows)} premières lignes :")
        for row in rows:
            row_dict = {k: v for k, v in row.__dict__.items() if k != "_sa_instance_state"}
            print(row_dict)
    else:
        print("Aucune donnée trouvée dans cette table.")

    # Nombre total de lignes
    total = session.query(table_class).count()
    print(f"Nombre total de lignes : {total}")
    session.close()


# -----------------------------
# Test de toutes les tables SQLAlchemy
# -----------------------------
for table_class in Base.__subclasses__():
    check_table(table_class, limit=5)

print("\n✅ Test complet terminé !")
