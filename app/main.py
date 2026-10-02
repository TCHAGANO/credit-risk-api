
# API DE PRÉDICTION DU RISQUE DE DÉFAUT


from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel



# 1. LOCALISATION DU MODÈLE


# __file__ = emplacement de app/main.py
# parent = dossier app/
# parent.parent = dossier FastAPI/
DOSSIER_PROJET = Path(__file__).resolve().parent.parent

CHEMIN_MODELE = DOSSIER_PROJET / "models" / "meilleur_modele.pkl"



# 2. CHARGEMENT DU MODÈLE


modele = joblib.load(CHEMIN_MODELE)



# 3. CRÉATION DE L'APPLICATION FASTAPI


app = FastAPI(
    title="API de prédiction du défaut de crédit",
    description="API permettant d'estimer le risque de défaut d'un emprunteur.",
    version="1.0.0"
)


# 4. STRUCTURE DES DONNÉES D'ENTRÉE

class Emprunteur(BaseModel):

    age: int

    revenu_annuel: float

    statut_logement: str

    anciennete_emploi: float

    motif_pret: str

    niveau_de_risque: str

    montant_pret: float

    taux_interet: float

    ratio_pret_revenu: float

    defaut_historique_credit: str

    duree_historique_credit: float


# 5. ROUTE D'ACCUEIL

@app.get("/")
def accueil():

    return {
        "message": "API de prédiction du défaut de crédit opérationnelle"
    }

# 6. ROUTE DE PRÉDICTION
@app.post("/predict")
def predire(emprunteur: Emprunteur):

    # Transformer les données reçues en dictionnaire
    donnees = emprunteur.model_dump()

    # Transformer le dictionnaire en DataFrame
    donnees_df = pd.DataFrame([donnees])

    # Prédiction de la classe
    prediction = modele.predict(donnees_df)[0]

    # Probabilité de défaut = probabilité de la classe 1
    probabilite_defaut = modele.predict_proba(
        donnees_df
    )[0, 1]

    # Retourner les résultats
    return {
        "prediction": int(prediction),
        "probabilite_defaut": round(
            float(probabilite_defaut),
            4
        )
    }