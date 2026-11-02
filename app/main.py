from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "meilleur_modele.pkl"

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API de prédiction du risque de défaut de crédit.",
    version="3.0.0",
)

model = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modèle introuvable : {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)


class Borrower(BaseModel):
    age: int = Field(ge=18, le=100)
    revenu_annuel: float = Field(ge=0)
    statut_logement: str
    anciennete_emploi: float = Field(ge=0)
    motif_pret: str
    niveau_de_risque: str
    montant_pret: float = Field(ge=0)
    taux_interet: float = Field(ge=0)
    ratio_pret_revenu: float = Field(ge=0)
    defaut_historique_credit: str
    duree_historique_credit: int = Field(ge=0)


@app.get("/")
def root():
    return {"message": "Credit Risk API is running", "version": app.version}


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(borrower: Borrower):
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé.")

    data = pd.DataFrame([borrower.model_dump()])

    try:
        prediction = int(model.predict(data)[0])
        probabilities = model.predict_proba(data)[0]
        # Cette API suppose que la classe positive (défaut) est la classe 1.
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("La classe 1 (défaut) est absente du modèle.")
        default_probability = float(probabilities[classes.index(1)])

        return {
            "prediction": prediction,
            "probabilite_defaut": round(default_probability, 6),
            "score_risque_pourcentage": round(default_probability * 100, 2),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {exc}") from exc
