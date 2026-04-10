# Image Python légère — on utilise slim pour réduire la taille
FROM python:3.11-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# Copie des dépendances d'abord (optimise le cache Docker)
COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code de l'API
COPY main.py .

# Copie du modèle et de la configuration
COPY ../models/xgboost_fruits_legumes.pkl ./models/
COPY ../models/features.json ./models/

# Port exposé par l'API
EXPOSE 8000

# Commande de démarrage
# --host 0.0.0.0 : accessible depuis l'extérieur du conteneur
# --port 8000    : port d'écoute
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]