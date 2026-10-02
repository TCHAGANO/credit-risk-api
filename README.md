# Prédiction du risque de défaut de crédit

## 1. Présentation du projet

Ce projet consiste à développer un modèle de Machine Learning permettant
d'estimer le risque de défaut d'un emprunteur à partir de ses caractéristiques
financières et personnelles.

Le modèle retenu est un Random Forest.

Le modèle est ensuite déployé sous forme d'API REST avec FastAPI et
conteneurisé avec Docker.

---

## 2. Objectif

L'objectif est de construire une chaîne complète de Data Science :

- exploration des données ;
- nettoyage des données ;
- préparation des variables ;
- entraînement de modèles de classification ;
- validation croisée ;
- évaluation sur un jeu de test ;
- sauvegarde du modèle ;
- création d'une API de prédiction ;
- déploiement avec Docker.

---

## 3. Données

La variable cible est :

- `statut_pret = 0` : pas de défaut ;
- `statut_pret = 1` : défaut.

Les variables explicatives comprennent notamment :

- âge ;
- revenu annuel ;
- statut de logement ;
- ancienneté professionnelle ;
- motif du prêt ;
- niveau de risque ;
- montant du prêt ;
- taux d'intérêt ;
- ratio prêt/revenu ;
- historique de défaut ;
- durée de l'historique de crédit.

---

## 4. Modélisation

Deux modèles ont été étudiés :

- Régression logistique ;
- Random Forest.

Une validation croisée stratifiée en 5 folds a été utilisée.

Les métriques étudiées sont :

- Accuracy ;
- Precision ;
- Recall ;
- F1-score ;
- ROC-AUC.

Le Random Forest a ensuite été sauvegardé dans :

`models/meilleur_modele.pkl`

---

## 5. API

L'API a été développée avec FastAPI.

Endpoint principal :

`POST /predict`

L'API reçoit les caractéristiques d'un emprunteur et retourne :

- la classe prédite ;
- la probabilité estimée de défaut.

Documentation interactive :

`http://localhost:8000/docs`

---

## 6. Docker

L'application est conteneurisée avec Docker.

Docker Compose permet de construire et lancer l'application.

Commande :

```bash
docker compose up --build