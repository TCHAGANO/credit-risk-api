# Credit Risk Scoring --- FastAPI & Streamlit

> **Projet de Data Science \| Classification binaire du risque de défaut
> de crédit**
>
> [Application de
> démonstration](https://credit-risk-api-tchagnao.streamlit.app/) ·
> [Documentation interactive de
> l'API](https://credit-risk-api-ihx8.onrender.com/docs) · [Dépôt
> GitHub](https://github.com/TCHAGANO/credit-risk-api)

------------------------------------------------------------------------

## 1. Contexte et problématique

L'octroi d'un crédit nécessite d'évaluer la capacité d'un emprunteur à
rembourser le montant accordé. Pour un établissement financier, une
mauvaise estimation du risque peut entraîner deux types d'erreurs :
accorder un financement à un emprunteur qui fera défaut, ou écarter un
profil pourtant solvable.

Ce projet consiste à concevoir une solution de **scoring de crédit**
fondée sur les caractéristiques personnelles, professionnelles et
financières des emprunteurs. L'objectif est de prédire le risque de
défaut à partir de données historiques, puis de rendre le modèle
accessible au moyen d'une application web.

La démarche couvre l'ensemble d'un cycle de projet Data Science :
préparation des données, analyse exploratoire, comparaison de modèles,
évaluation, sauvegarde du pipeline, développement d'une API et création
d'un tableau de bord interactif.

Deux dimensions structurent le travail :

-   **Dimension analytique :** comparer plusieurs algorithmes de
    classification et différentes stratégies de traitement du
    déséquilibre des classes afin d'évaluer leur capacité à détecter les
    défauts de paiement.
-   **Dimension applicative :** exposer le modèle avec FastAPI et
    proposer une interface Streamlit permettant de soumettre des
    dossiers, consulter les résultats et traiter un portefeuille à
    partir d'un fichier CSV.

> **Périmètre d'utilisation :** cette application est une démonstration
> technique et pédagogique. Elle ne constitue pas un système bancaire
> homologué et ne doit pas être utilisée seule pour prendre une décision
> réelle d'octroi de crédit.

## 2. Objectifs du projet

-   Explorer et préparer un jeu de données de risque de crédit issu de
    Kaggle.
-   Identifier et traiter les valeurs manquantes et les valeurs
    aberrantes.
-   Étudier les relations entre les caractéristiques des emprunteurs et
    le défaut de paiement.
-   Comparer trois algorithmes : Régression Logistique, Random Forest et
    XGBoost.
-   Évaluer trois stratégies de gestion du déséquilibre des classes :
    Baseline, pondération et SMOTE.
-   Comparer les modèles à l'aide de métriques adaptées à la
    classification du risque.
-   Retenir un modèle, sauvegarder son pipeline de prétraitement et de
    prédiction, puis l'exposer via une API.
-   Développer un dashboard permettant l'évaluation individuelle et le
    traitement de plusieurs dossiers.

## 3. Données et variables

Le projet utilise le jeu de données **`credit_risk_dataset`**, issu de
Kaggle. Le jeu de données préparé pour l'analyse comprend **32 416
observations et 12 colonnes**.

Les variables exploitées par l'API décrivent le profil de l'emprunteur
et les caractéristiques du crédit :

  -----------------------------------------------------------------------
  Variable                            Description
  ----------------------------------- -----------------------------------
  `age`                               Âge de l'emprunteur

  `revenu_annuel`                     Revenu annuel de l'emprunteur

  `statut_logement`                   Situation résidentielle (`RENT`,
                                      `OWN`, `MORTGAGE`, etc.)

  `anciennete_emploi`                 Ancienneté professionnelle

  `motif_pret`                        Motif du financement (`PERSONAL`,
                                      `EDUCATION`, `MEDICAL`, etc.)

  `niveau_de_risque`                  Grade du prêt, représenté par les
                                      catégories `A` à `G`

  `montant_pret`                      Montant du crédit sollicité

  `taux_interet`                      Taux d'intérêt associé au prêt

  `ratio_pret_revenu`                 Rapport entre le montant du prêt et
                                      le revenu annuel

  `defaut_historique_credit`          Indicateur d'un antécédent de
                                      défaut (`N` ou `Y`)

  `duree_historique_credit`           Ancienneté de l'historique de
                                      crédit
  -----------------------------------------------------------------------

La variable cible correspond au statut de défaut de paiement. Dans le
cadre de la classification binaire, **0 représente un crédit sans défaut
et 1 représente un défaut**.

## 4. Préparation et nettoyage des données

La préparation des données constitue une étape essentielle du projet.
Elle vise à améliorer leur cohérence et à limiter l'apprentissage de
relations artificielles par les modèles.

### 4.1 Contrôle de la qualité des données

L'analyse initiale porte notamment sur :

-   les dimensions et les types des variables ;
-   la présence de doublons ;
-   le taux de valeurs manquantes par colonne ;
-   les distributions des variables numériques ;
-   les modalités des variables catégorielles ;
-   les valeurs impossibles ou incohérentes ;
-   la répartition de la variable cible.

Ces contrôles permettent de distinguer les erreurs de saisie des
observations rares mais plausibles. Une valeur extrême n'est pas
automatiquement une erreur : son traitement doit tenir compte de sa
cohérence avec le contexte du crédit.

### 4.2 Traitement des valeurs manquantes

Les valeurs manquantes sont traitées en tenant compte de la nature de
chaque variable. Une suppression systématique des lignes peut entraîner
une perte d'information et réduire la taille de l'échantillon disponible
pour l'apprentissage.

Une attention particulière est portée à la variable `taux_interet`. Les
taux d'intérêt sont liés au profil de risque du prêt. Pour préserver les
observations incomplètes, une **imputation conditionnelle** s'appuie sur
le `niveau_de_risque` et le `ratio_pret_revenu`. Cette approche tient
compte du contexte de l'emprunt plutôt que d'appliquer une valeur unique
à toutes les observations.

### 4.3 Traitement des valeurs aberrantes

Les variables numériques sont examinées afin d'identifier les valeurs
extrêmes et les incohérences susceptibles de perturber l'apprentissage.
Les contrôles concernent notamment l'âge, le revenu annuel, l'ancienneté
professionnelle, le montant du prêt, le taux d'intérêt et le ratio
prêt/revenu.

Les valeurs physiquement impossibles ou manifestement incohérentes sont
traitées lors du nettoyage. Les valeurs extrêmes mais plausibles
doivent, quant à elles, être interprétées avec prudence : leur
suppression automatique pourrait éliminer des profils réels importants
pour l'analyse du risque.

### 4.4 Prétraitement et cohérence entre entraînement et prédiction

Les variables catégorielles et numériques doivent être préparées dans un
format compatible avec les algorithmes. Le pipeline sauvegardé avec le
modèle regroupe le prétraitement nécessaire à la prédiction.

Cette organisation permet à l'API de recevoir les caractéristiques d'un
emprunteur et de leur appliquer les transformations attendues avant
l'inférence. Elle contribue à conserver la cohérence entre les données
utilisées lors de l'entraînement et celles reçues par l'application.

## 5. Modélisation et gestion du déséquilibre des classes

Trois algorithmes de classification ont été comparés :

-   **Régression Logistique :** modèle de référence, apprécié pour sa
    simplicité et son interprétabilité. Il permet d'établir une première
    performance de comparaison.
-   **Random Forest :** ensemble d'arbres de décision capable de
    représenter des relations non linéaires et des interactions entre
    variables.
-   **XGBoost :** algorithme de boosting fondé sur une succession
    d'arbres, chaque étape cherchant à améliorer les prédictions
    précédentes.

Pour chacun de ces algorithmes, trois stratégies ont été étudiées, soit
neuf configurations distinctes.

  -----------------------------------------------------------------------
  Stratégie               Principe                Objectif
  ----------------------- ----------------------- -----------------------
  Baseline                Entraînement sans       Établir une référence
                          rééquilibrage           
                          spécifique              

  Pondération             Augmentation du poids   Accroître l'importance
                          attribué à la classe    des erreurs sur les
                          minoritaire             défauts

  SMOTE                   Génération              Améliorer
                          d'observations          l'apprentissage de la
                          synthétiques de la      classe minoritaire
                          classe minoritaire      
  -----------------------------------------------------------------------

La pondération modifie le coût associé aux erreurs pendant
l'apprentissage. SMOTE modifie la composition de l'échantillon
d'entraînement en générant des observations synthétiques. Ces stratégies
répondent au même problème, mais agissent de façons différentes.

## 6. Évaluation et comparaison des configurations

Trois métriques principales sont utilisées pour comparer les
performances :

-   **ROC-AUC :** mesure la capacité du modèle à distinguer les deux
    classes pour différents seuils de décision. Une valeur élevée
    indique une meilleure capacité de classement.
-   **Recall (Défaut) :** mesure la proportion de défauts réels
    correctement détectés. Un rappel élevé signifie que le modèle manque
    moins de dossiers qui présentent effectivement un défaut.
-   **F1-score :** moyenne harmonique de la précision et du rappel. Il
    permet d'évaluer leur compromis.

Dans le contexte du risque de crédit, le rappel de la classe défaut est
particulièrement important, car un défaut non détecté peut représenter
un coût financier. Toutefois, augmenter le rappel peut également
entraîner davantage de fausses alertes. Le choix du modèle doit donc
tenir compte de plusieurs métriques et du coût relatif des erreurs.

### 6.1 Tableau comparatif des neuf configurations

  Modèle                  Stratégie           ROC-AUC   Recall (Défaut)     F1-score
  ----------------------- -------------- ------------ ----------------- ------------
  **XGBoost**             **Baseline**     **0,9475**            0,7489   **0,8349**
  XGBoost                 Pondération          0,9443        **0,7976**       0,8067
  XGBoost                 SMOTE                0,9428            0,7489       0,8336
  Random Forest           Baseline             0,9274            0,7264       0,8260
  Random Forest           Pondération          0,9279            0,7532       0,8153
  Random Forest           SMOTE                0,9236            0,7433       0,8148
  Régression Logistique   Baseline             0,8696            0,5776       0,6618
  Régression Logistique   Pondération          0,8695            0,6206       0,6672
  Régression Logistique   SMOTE                0,8468            0,6382       0,6362

### 6.2 Interprétation et choix du modèle

Les résultats montrent que **XGBoost Baseline** obtient le meilleur
ROC-AUC (0,9475) et le meilleur F1-score (0,8349) parmi les neuf
configurations présentées. Il offre ainsi le meilleur compromis global
selon ces deux critères.

**XGBoost avec pondération** obtient le meilleur rappel de la classe
défaut (0,7976). Cette configuration détecte une proportion plus
importante des défauts réels, mais présente un F1-score inférieur à
celui de XGBoost Baseline.

Le choix du modèle dépend donc de la priorité retenue :

-   si l'objectif est de privilégier la capacité globale de classement
    et le compromis précision-rappel, XGBoost Baseline est le meilleur
    choix selon les résultats obtenus ;
-   si l'objectif prioritaire est de détecter davantage de défauts,
    XGBoost avec pondération constitue une alternative intéressante.

Le pipeline XGBoost retenu pour l'application est sauvegardé dans
**`models/meilleur_modele.pkl`**. La configuration Baseline présente les
meilleurs résultats globaux selon le ROC-AUC et le F1-score du tableau
comparatif.

### 6.3 Pourquoi ces modèles et ces stratégies ?

La Régression Logistique constitue une référence simple, utile pour
mesurer le gain apporté par des modèles plus flexibles. Ses résultats
sont inférieurs à ceux de Random Forest et XGBoost sur les métriques
présentées.

Random Forest permet de représenter des relations non linéaires, mais
son meilleur ROC-AUC (0,9279) reste inférieur à celui de XGBoost.

XGBoost obtient les meilleures performances globales dans cette
comparaison. Les résultats montrent également que SMOTE n'améliore pas
systématiquement les performances : son intérêt dépend de l'algorithme
et de la métrique observée. La pondération apporte, dans le cas de
XGBoost, un gain de rappel au prix d'une baisse du F1-score.

Cette comparaison illustre pourquoi il est nécessaire de tester
plusieurs configurations plutôt que de choisir une stratégie de
rééquilibrage a priori.

## 7. Architecture applicative

L'application sépare l'interface utilisateur du moteur de prédiction. Le
dashboard transmet les caractéristiques du dossier à l'API, qui valide
les données, exécute le pipeline sauvegardé et renvoie le résultat au
frontend.

``` text
                 UTILISATEUR
                      |
                      v
          DASHBOARD STREAMLIT
              (Frontend)
                      |
              Requête HTTP JSON
               POST /predict
                      |
                      v
              API FASTAPI
               (Backend)
                      |
             Validation Pydantic
                      |
                      v
           Pipeline XGBoost sauvegardé
             meilleur_modele.pkl
                      |
                      v
              Prédiction et
          probabilité de défaut
                      |
                      v
           Résultat dans Streamlit
```

### 7.1 Backend --- FastAPI

FastAPI expose les endpoints suivants :

  -----------------------------------------------------------------------
  Endpoint                Méthode                 Rôle
  ----------------------- ----------------------- -----------------------
  `/`                     GET                     Vérifier que l'API
                                                  répond

  `/health`               GET                     Contrôler l'état du
                                                  service

  `/predict`              POST                    Obtenir une prédiction
                                                  à partir d'un dossier
                                                  emprunteur
  -----------------------------------------------------------------------

La validation des données entrantes est assurée par **Pydantic**. Les
caractéristiques reçues sont vérifiées avant d'être transmises au
pipeline de Machine Learning. L'API renvoie notamment la classe prédite,
la probabilité de défaut et le score de risque en pourcentage.

### 7.2 Frontend --- Streamlit

Le dashboard propose plusieurs espaces fonctionnels :

-   **Vue d'ensemble :** présentation du système et des indicateurs de
    la session.
-   **Nouvelle prédiction :** saisie des caractéristiques d'un
    emprunteur et affichage du résultat.
-   **Analyse portefeuille :** import d'un fichier CSV, exécution des
    prédictions et téléchargement des résultats.
-   **Historique :** consultation des prédictions réalisées pendant la
    session.
-   **Paramètres :** affichage de la configuration et de l'état de
    connexion à l'API.

L'interface utilise **Plotly** pour les visualisations interactives et
des composants Streamlit pour les formulaires, les tableaux et la
navigation.

## 8. Technologies utilisées

  Domaine                    Technologies
  -------------------------- ----------------------------
  Langage                    Python
  Manipulation des données   Pandas, NumPy
  Machine Learning           Scikit-learn, XGBoost
  Rééquilibrage              SMOTE
  API Backend                FastAPI, Pydantic, Uvicorn
  Frontend                   Streamlit
  Visualisation              Plotly
  Communication API          HTTP, JSON, Requests
  Sérialisation du modèle    Joblib / Pickle
  Conteneurisation           Docker
  Versionnement              Git, GitHub
  Déploiement Backend        Render
  Déploiement Frontend       Streamlit Community Cloud

## 9. Déploiement et démonstration

Le code source est versionné sur GitHub. Le backend FastAPI est hébergé
sur Render et le dashboard Streamlit est publié sur Streamlit Community
Cloud.

-   **Dashboard :** https://credit-risk-api-tchagnao.streamlit.app/
-   **API FastAPI :** https://credit-risk-api-ihx8.onrender.com/
-   **Documentation interactive :**
    https://credit-risk-api-ihx8.onrender.com/docs
-   **Dépôt GitHub :** https://github.com/TCHAGANO/credit-risk-api

Cette architecture permet de tester l'application depuis un navigateur
sans avoir à exécuter localement le modèle.

## 10. Installation et exécution en local

### Prérequis

-   Python ;
-   Git ;
-   un environnement virtuel Python.

### Cloner le dépôt

``` bash
git clone https://github.com/TCHAGANO/credit-risk-api.git
cd credit-risk-api
```

### Créer l'environnement et installer les dépendances

``` bash
# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement sous Windows PowerShell
.venv\Scripts\Activate.ps1

# Installer les dépendances du projet
pip install -r requirements.txt
```

### Lancer l'API FastAPI

``` bash
uvicorn app.main:app --reload
```

L'API est accessible à l'adresse `http://127.0.0.1:8000`. La
documentation interactive est disponible sur
`http://127.0.0.1:8000/docs`.

### Lancer le dashboard Streamlit

Dans un second terminal :

``` powershell
# Définir l'adresse de l'API locale
$env:CREDIT_RISK_API_URL = "http://127.0.0.1:8000"

# Démarrer l'application Streamlit
python -m streamlit run Frontend/app_streamlit.py
```

La variable `CREDIT_RISK_API_URL` permet de configurer l'adresse du
backend utilisé par le dashboard.

## 11. Limites et perspectives d'amélioration

Plusieurs axes peuvent renforcer la robustesse et la maturité de cette
solution :

-   **Validation croisée stratifiée :** mesurer la stabilité des
    performances sur plusieurs découpages des données.
-   **Optimisation des hyperparamètres :** rechercher des configurations
    plus performantes pour les trois algorithmes.
-   **Optimisation du seuil de décision :** choisir un seuil adapté au
    compromis entre défauts non détectés et fausses alertes.
-   **Calibration des probabilités :** vérifier que les probabilités
    estimées sont cohérentes avec les fréquences de défaut observées.
-   **Interprétabilité :** utiliser SHAP pour expliquer les facteurs qui
    contribuent aux prédictions individuelles et globales.
-   **Analyse des erreurs :** étudier les matrices de confusion et les
    profils des faux positifs et faux négatifs.
-   **Monitoring :** suivre l'évolution des données reçues et des
    performances du modèle dans le temps.
-   **Persistance de l'historique :** ajouter une base de données pour
    conserver les évaluations au-delà d'une session Streamlit.
-   **Tests automatisés et CI/CD :** renforcer la qualité logicielle et
    automatiser les vérifications lors des mises à jour.
-   **Équité et conformité :** évaluer les biais potentiels et prendre
    en compte les exigences réglementaires applicables aux systèmes de
    décision de crédit.

## 12. Conclusion

Ce projet met en œuvre une chaîne de Data Science allant de la
préparation des données à la mise à disposition d'une application web de
scoring de crédit. La comparaison de neuf configurations a permis
d'étudier l'effet des algorithmes et des stratégies de gestion du
déséquilibre des classes.

Sur les résultats présentés, XGBoost Baseline obtient le meilleur
ROC-AUC et le meilleur F1-score, tandis que XGBoost avec pondération
maximise le rappel de la classe défaut. Le choix du modèle met ainsi en
évidence l'importance de relier les métriques d'évaluation aux objectifs
du problème métier.

Enfin, l'intégration du modèle dans une API FastAPI et sa mise à
disposition via un dashboard Streamlit illustrent les différentes étapes
nécessaires pour transformer une expérimentation de Machine Learning en
application accessible et testable.
