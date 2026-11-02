
# Image de base : Python
FROM python:3.13-slim

# Dossier de travail dans le conteneur Docker
WORKDIR /app

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'API
COPY app ./app

# Copier le modèle de Machine Learning
COPY models ./models

# Indiquer le port utilisé par Render
EXPOSE 10000

# Démarrer FastAPI sur le port fourni par Render
# Si PORT n'existe pas, utiliser 10000 par défaut
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
