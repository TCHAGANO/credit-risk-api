# Credit Risk Scoring --- API FastAPI & Dashboard Streamlit

> **Projet de Data Science --- classification binaire du risque de
> défaut de crédit**\
> Application de démonstration : [ouvrir le dashboard
> Streamlit](https://credit-risk-api-tchagnao.streamlit.app/) ·
> [consulter la documentation de
> l'API](https://credit-risk-api-ihx8.onrender.com/docs) · [dépôt
> GitHub](https://github.com/TCHAGANO/credit-risk-api)

------------------------------------------------------------------------

## 1. Contexte et problématique

L'octroi d'un crédit implique d'évaluer la capacité d'un emprunteur à
rembourser son prêt. Pour un établissement financier, une mauvaise
estimation du risque peut entraîner des pertes : un emprunteur
présentant un risque élevé peut être identifié à tort comme peu risqué.
À l'inverse, un profil solvable peut être considéré comme trop risqué.

L'objectif de ce projet est de construire une solution de **scoring de
crédit** à partir de données décrivant les caractéristiques financières
et personnelles des emprunteurs. Le travail ne s'arrête pas à
l'entraînement d'un modèle dans un notebook : le modèle retenu est
intégré à une API et rendu accessible au moyen d'une interface web.

Le projet répond ainsi à deux questions complémentaires :

1.  **Question analytique :** parmi plusieurs modèles et stratégies de
    traitement du déséquilibre des classes, quelle configuration offre
    le meilleur compromis pour détecter les défauts de paiement ?
2.  **Question applicative :** comment rendre ce modèle utilisable
    depuis une interface, avec une prédiction retournée par une API ?

L'application est une **démonstration pédagogique et technique**. Elle
ne constitue pas un outil bancaire homologué et ne doit pas être
utilisée seule pour accorder ou refuser un crédit.

## 2. Objectifs

Les objectifs du projet sont les suivants :

-   explorer et préparer un jeu de données de risque de crédit ;
-   traiter les valeurs manquantes et examiner les valeurs aberrantes ;
-   étudier les relations entre les variables explicatives et le défaut
    de paiement ;
-   comparer trois algorithmes de classification ;
-   mesurer l'effet de trois stratégies face au déséquilibre des classes
    ;
-   sélectionner une configuration à partir de métriques adaptées au
    risque de crédit ;
-   sauvegarder le modèle et l'exposer avec FastAPI ;
-   créer un dashboard Streamlit pour effectuer des évaluations
    individuelles et des analyses par fichier CSV.

## 3. Données et variables

Le projet utilise le jeu de données **`credit_risk_dataset`**, récupéré
sur Kaggle. Après la préparation mentionnée dans le projet, le jeu de
données utilisé comporte **32 416 lignes et 12 colonnes**.

Les variables transmises au modèle via l'API sont :

  -----------------------------------------------------------------------
  Variable                            Description
  ----------------------------------- -----------------------------------
  `age`                               Âge de l'emprunteur

  `revenu_annuel`                     Revenu annuel

  `statut_logement`                   Situation de logement (`RENT`,
                                      `OWN`, `MORTGAGE`, etc.)

  `anciennete_emploi`                 Ancienneté professionnelle

  `motif_pret`                        Motif du prêt

  `niveau_de_risque`                  Grade du prêt

  `montant_pret`                      Montant demandé

  `taux_interet`                      Taux d'intérêt associé au prêt

  `ratio_pret_revenu`                 Rapport entre le montant du prêt et
                                      le revenu annuel

  `defaut_historique_credit`          Indicateur d'un défaut antérieur
                                      (`N` ou `Y`)

  `duree_historique_credit`           Ancienneté de l'historique de
                                      crédit
  -----------------------------------------------------------------------

La variable cible représente le statut de défaut de paiement. Il faut
conserver la convention exacte de codage utilisée pendant l'entraînement
: **la signification de 0 et de 1 doit être vérifiée dans le notebook et
dans le code de préparation** avant toute interprétation métier.

## 4. Préparation et nettoyage des données

La qualité des données est essentielle : un modèle peut apprendre des
relations erronées si les valeurs sont incohérentes, si les valeurs
manquantes sont mal traitées ou si des informations issues de la cible
contaminent les variables explicatives.

### 4.1 Contrôles initiaux

Avant la modélisation, les contrôles à effectuer comprennent :

-   dimensions du jeu de données et types des colonnes ;
-   doublons éventuels ;
-   valeurs manquantes par variable ;
-   valeurs numériques impossibles ou incohérentes ;
-   modalités des variables catégorielles ;
-   distribution de la variable cible et proportion de chaque classe.

Ces vérifications permettent de distinguer une véritable anomalie d'une
valeur simplement rare. Par exemple, un revenu élevé peut être
inhabituel sans être nécessairement erroné.

### 4.2 Valeurs manquantes

Les valeurs manquantes ne doivent pas être remplacées automatiquement
par zéro ou par une moyenne globale. La méthode dépend du sens de la
variable et de la raison probable de l'absence.

Dans ce projet, le **taux d'intérêt** a fait l'objet d'un traitement
spécifique. Plutôt que de supprimer systématiquement les lignes
concernées, une imputation conditionnelle a été envisagée à partir du
`niveau_de_risque` et du `ratio_pret_revenu`. Cette approche cherche à
préserver les observations tout en tenant compte du contexte du prêt.

Le principe est de ne pas attribuer la même valeur à tous les
emprunteurs lorsque le taux dépend de leur profil. La méthode exacte de
calcul (médiane par groupe, modèle d'imputation ou autre règle) doit
être documentée à partir du notebook d'entraînement.

**Point de vigilance :** les paramètres d'imputation doivent être appris
sur les données d'entraînement uniquement, puis appliqués aux données de
validation et de test. Cela évite une fuite d'information (*data
leakage*).

### 4.3 Valeurs aberrantes et incohérences

Les valeurs aberrantes ont été examinées avant la modélisation. Il est
important de distinguer :

-   **une erreur de saisie ou une valeur impossible**, qui doit être
    corrigée ou exclue selon une règle justifiée ;
-   **une valeur extrême mais plausible**, qui peut représenter un
    profil réel et ne doit pas être supprimée automatiquement.

Les contrôles métier portent notamment sur l'âge, le revenu, le montant
du prêt, le taux d'intérêt, l'ancienneté professionnelle et le ratio
prêt/revenu. Les valeurs négatives ou les combinaisons manifestement
incohérentes doivent être investiguées.

Les seuils précis, le nombre de lignes corrigées ou supprimées et la
méthode statistique utilisée pour détecter les valeurs extrêmes doivent
être renseignés à partir du notebook. Ils ne sont pas déduits des seuls
résultats de modèles présentés ici.

### 4.4 Encodage et prévention des fuites de données

Les variables catégorielles doivent être transformées dans un format
exploitable par les algorithmes. Les transformations et l'encodage sont
intégrés au pipeline sauvegardé, de sorte que l'API puisse recevoir les
valeurs brutes attendues et appliquer le même prétraitement que pendant
l'entraînement.

Le prétraitement doit être ajusté sur l'échantillon d'entraînement, puis
réutilisé sans réapprentissage sur les échantillons de validation et de
test. La variable cible ne doit jamais être utilisée comme variable
explicative.

## 5. Comparaison des modèles et des stratégies

Trois algorithmes ont été comparés :

-   **Régression logistique :** modèle de référence (*baseline*), simple
    à interpréter et utile pour établir un premier niveau de performance
    ;
-   **Random Forest :** ensemble d'arbres de décision capable de
    représenter des relations non linéaires et des interactions ;
-   **XGBoost :** méthode de boosting qui construit successivement des
    arbres afin d'améliorer les prédictions.

Pour chacun des trois algorithmes, trois stratégies de gestion du
déséquilibre des classes ont été étudiées, soit **neuf configurations
théoriques**.

  -----------------------------------------------------------------------
  Stratégie               Principe                Intérêt
  ----------------------- ----------------------- -----------------------
  Baseline                Entraînement sans       Mesurer les
                          correction spécifique   performances de
                          du déséquilibre         référence

  Pondération             Donner davantage de     Pénaliser davantage les
                          poids à la classe       erreurs sur les défauts
                          minoritaire dans la     
                          fonction de coût        

  SMOTE                   Créer des observations  Aider le modèle à
                          synthétiques de la      apprendre la classe
                          classe minoritaire à    minoritaire
                          partir de ses voisins   
  -----------------------------------------------------------------------

La pondération agit sur le coût des erreurs pendant l'apprentissage.
SMOTE agit sur la composition de l'échantillon d'entraînement en
générant des observations synthétiques. Ces méthodes ne sont donc pas
équivalentes et leurs effets doivent être évalués sur un jeu de test qui
n'a pas été rééquilibré.

**Précaution méthodologique :** SMOTE doit être appliqué uniquement dans
les données d'entraînement, idéalement à l'intérieur d'un pipeline de
validation croisée. Il ne faut pas appliquer SMOTE avant la séparation
entraînement/test, car cela peut contaminer l'évaluation.

### 5.1 Critères d'évaluation

Trois métriques sont présentées dans les résultats :

-   **ROC-AUC :** mesure la capacité du modèle à classer les défauts
    devant les non-défauts pour différents seuils de décision. Plus la
    valeur est proche de 1, meilleure est la séparation des classes.
-   **Recall (Défaut) :** parmi les emprunteurs ayant réellement fait
    défaut, proportion de ceux détectés par le modèle. Un rappel élevé
    signifie moins de défauts manqués (*faux négatifs*, si la classe
    défaut est bien la classe positive).
-   **F1-score :** moyenne harmonique de la précision et du rappel. Il
    permet de rechercher un compromis entre les défauts détectés et les
    fausses alertes.

Dans le risque de crédit, manquer un emprunteur qui fera défaut peut
être coûteux. Le rappel de la classe défaut est donc important.
Toutefois, augmenter le rappel peut aussi augmenter le nombre de faux
positifs : il faut examiner la précision, la matrice de confusion et le
coût métier des erreurs avant de fixer un seuil.

### 5.2 Résultats comparatifs

Le tableau ci-dessous reprend les neuf configurations distinctes
correspondant aux trois modèles et aux trois stratégies.

  Modèle                  Stratégie          ROC-AUC   Recall (Défaut)     F1-score
  ----------------------- ------------- ------------ ----------------- ------------
  XGBoost                 Baseline        **0,9475**            0,7489   **0,8349**
  XGBoost                 Pondération         0,9443        **0,7976**       0,8067
  XGBoost                 SMOTE               0,9428            0,7489       0,8336
  Random Forest           Baseline            0,9274            0,7264       0,8260
  Random Forest           Pondération         0,9279            0,7532       0,8153
  Random Forest           SMOTE               0,9236            0,7433       0,8148
  Régression logistique   Baseline            0,8696            0,5776       0,6618
  Régression logistique   Pondération         0,8695            0,6206       0,6672
  Régression logistique   SMOTE               0,8468            0,6382       0,6362

*Note sur les résultats transmis : le tableau source comportait 14
lignes, avec plusieurs lignes dupliquées et une seconde valeur pour la
configuration Régression logistique -- Baseline (ROC-AUC = 0,8659 ;
Recall = 0,2786 ; F1 = 0,4270). Cette seconde valeur n'est pas intégrée
au tableau des neuf configurations ci-dessus, car elle contredit l'autre
résultat Baseline. Il faut vérifier le notebook ou les résultats
exportés pour identifier s'il s'agit d'un autre essai, d'un seuil
différent ou d'une erreur de copie. Les lignes dupliquées Random Forest
et Régression logistique--SMOTE ont été ramenées à une seule ligne.*

### 5.3 Choix de la configuration retenue

Sur la base des résultats présentés, **XGBoost Baseline** est la
configuration qui obtient le ROC-AUC le plus élevé (0,9475) et le
meilleur F1-score (0,8349). Elle constitue donc le choix cohérent si la
priorité est la qualité globale de classement et le compromis
précision-rappel mesuré par le F1-score.

Cependant, **XGBoost avec pondération** obtient le meilleur rappel de la
classe défaut (0,7976). Cette configuration détecte une plus grande
proportion des défauts observés, au prix d'un F1-score plus faible
(0,8067) et d'un ROC-AUC légèrement inférieur (0,9443).

Le choix final dépend donc de l'objectif métier :

-   privilégier XGBoost Baseline si l'on retient le meilleur ROC-AUC et
    le meilleur F1-score parmi les résultats fournis ;
-   privilégier XGBoost avec pondération si la priorité est de réduire
    les défauts manqués, sous réserve d'accepter et de mesurer les
    fausses alertes supplémentaires.

Le modèle retenu dans l'application est XGBoost. Le fichier sauvegardé
est `models/meilleur_modele.pkl`. Le critère de sélection exact doit
être aligné sur la décision réellement prise dans le notebook : les
métriques seules ne permettent pas de connaître le seuil de
classification choisi ni le coût attribué à chaque type d'erreur.

### 5.4 Pourquoi ne pas utiliser une autre approche ?

La régression logistique est un excellent modèle de référence, mais ses
résultats sont inférieurs à ceux de Random Forest et XGBoost sur les
métriques fournies. Elle reste intéressante pour sa simplicité et son
interprétabilité.

Random Forest obtient de bons résultats, mais son ROC-AUC maximal
(0,9279) reste inférieur à celui de XGBoost dans les essais présentés.

SMOTE améliore le rappel de la régression logistique par rapport à sa
baseline retenue, mais diminue son ROC-AUC et son F1-score. Pour
XGBoost, la baseline obtient un meilleur ROC-AUC et un meilleur F1-score
que SMOTE, tandis que la pondération obtient un meilleur rappel. Il
n'existe donc pas de stratégie universellement supérieure : le résultat
dépend de l'algorithme et de la métrique prioritaire.

D'autres approches pourraient être étudiées (réglage des
hyperparamètres, validation croisée stratifiée, calibration des
probabilités, ajustement du seuil, méthodes de rééquilibrage
alternatives). Elles n'ont pas à être présentées comme réalisées tant
qu'elles n'ont pas été exécutées et évaluées.

## 6. Architecture applicative et déploiement

Le projet sépare le moteur de prédiction de l'interface utilisateur.

``` text
Utilisateur
    |
    v
Dashboard Streamlit
    |
    | requête HTTP POST /predict
    v
API FastAPI
    |
    v
Pipeline XGBoost sauvegardé
    |
    v
Prédiction + probabilité estimée de défaut
    |
    v
Résultat affiché dans Streamlit
```

### Backend --- FastAPI

FastAPI expose notamment :

-   `GET /` : vérification de la disponibilité générale de l'API ;
-   `GET /health` : contrôle de santé utilisé par le dashboard ;
-   `POST /predict` : réception des caractéristiques d'un emprunteur et
    retour de la prédiction.

La validation des données entrantes est assurée par Pydantic. Les types
et contraintes définis dans le schéma permettent de rejeter les requêtes
invalides avant l'appel au modèle. Le pipeline sauvegardé est chargé par
le backend et utilisé pour produire les résultats.

### Frontend --- Streamlit

Le dashboard permet :

-   de saisir les caractéristiques d'un emprunteur ;
-   de demander une prédiction à l'API ;
-   de visualiser la probabilité estimée et le niveau de risque
    indicatif ;
-   de consulter l'historique des prédictions de la session ;
-   d'importer un fichier CSV pour lancer des prédictions en lot et
    télécharger les résultats.

### Hébergement

-   **GitHub** : versionnement du code source ;
-   **Render** : hébergement du backend FastAPI ;
-   **Streamlit Community Cloud** : hébergement du dashboard.

Liens du projet :

-   Dashboard : https://credit-risk-api-tchagnao.streamlit.app/
-   API : https://credit-risk-api-ihx8.onrender.com/
-   Documentation interactive :
    https://credit-risk-api-ihx8.onrender.com/docs
-   Code source : https://github.com/TCHAGANO/credit-risk-api

## 7. Installation et exécution en local

### Prérequis

-   Python installé ;
-   Git ;
-   un environnement virtuel recommandé.

### Installation

``` bash
# Cloner le dépôt
git clone https://github.com/TCHAGANO/credit-risk-api.git
cd credit-risk-api

# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement sous Windows PowerShell
.venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt
```

### Démarrer l'API

``` bash
uvicorn app.main:app --reload
```

L'API est alors accessible localement à l'adresse
`http://127.0.0.1:8000` et sa documentation à
`http://127.0.0.1:8000/docs`.

### Démarrer Streamlit

Dans un second terminal :

``` powershell
# Utiliser l'API locale
$env:CREDIT_RISK_API_URL = "http://127.0.0.1:8000"

# Démarrer le dashboard
python -m streamlit run Frontend/app_streamlit.py
```

Pour utiliser le backend hébergé sur Render depuis une exécution locale,
définir `CREDIT_RISK_API_URL` avec l'adresse publique de l'API.

## 8. Limites et perspectives

Ce projet constitue une première chaîne complète de scoring, mais
plusieurs améliorations sont nécessaires avant d'envisager un usage
opérationnel.

1.  **Validation plus robuste :** répéter les expériences avec
    validation croisée stratifiée et conserver un jeu de test
    indépendant.
2.  **Analyse des erreurs :** examiner les matrices de confusion, la
    précision de la classe défaut et les faux négatifs/faux positifs.
3.  **Optimisation du seuil :** comparer plusieurs seuils de décision
    selon le coût relatif des erreurs, au lieu de supposer qu'un seuil
    standard convient au contexte bancaire.
4.  **Calibration des probabilités :** vérifier que les probabilités
    annoncées correspondent aux fréquences observées de défaut.
5.  **Interprétabilité :** utiliser des outils comme les importances de
    variables et SHAP pour expliquer les facteurs influençant les
    prédictions.
6.  **Robustesse et équité :** contrôler la stabilité du modèle et
    rechercher d'éventuels écarts de performance entre groupes, dans le
    respect du cadre légal applicable.
7.  **Suivi en production :** journaliser les requêtes de manière
    sécurisée, surveiller les dérives de données et de performance,
    gérer les erreurs et versionner les modèles.
8.  **Persistance de l'historique :** remplacer l'historique limité à la
    session Streamlit par un stockage adapté si un suivi durable est
    nécessaire.

## 9. Conclusion

Ce projet met en œuvre une démarche de Data Science de bout en bout :
préparation des données, exploration, comparaison de modèles,
évaluation, sauvegarde du pipeline, création d'une API et développement
d'un dashboard web.

Les résultats fournis montrent que XGBoost obtient les meilleures
performances globales selon le ROC-AUC et le F1-score pour la
configuration Baseline, tandis que la pondération améliore le rappel de
la classe défaut. Ce résultat illustre l'importance de choisir un modèle
en fonction du problème métier et du coût des erreurs, plutôt que de se
limiter à une seule métrique.

Le déploiement avec FastAPI, Render et Streamlit rend le travail
consultable et testable depuis un navigateur. Il constitue une
démonstration concrète de la transformation d'un modèle de Machine
Learning en application accessible.

------------------------------------------------------------------------

## Informations à compléter avant publication

Pour que ce README décrive exactement le travail réalisé, vérifier dans
le notebook :

-   la méthode exacte d'imputation du taux d'intérêt et les autres
    valeurs manquantes ;
-   la règle et les seuils utilisés pour traiter les valeurs aberrantes
    ;
-   le découpage entraînement/validation/test et la graine aléatoire ;
-   le nom exact de la classe correspondant au défaut (0 ou 1) ;
-   les hyperparamètres et le seuil de classification de chaque
    configuration ;
-   la raison de la seconde ligne de résultats Régression logistique --
    Baseline (0,8659 / 0,2786 / 0,4270) ;
-   la métrique effectivement utilisée pour sélectionner le fichier
    `meilleur_modele.pkl`.

Ne pas publier de méthode, de seuil ou de résultat comme réalisé tant
qu'il n'a pas été confirmé dans le code ou les sorties
d'expérimentation.
