# text-search-engine-m1

Implémentation complète d’un moteur de recherche textuel basé sur un index inversé et un scoring TF-IDF. Le projet contient les briques de nettoyage, d’indexation, de recherche, d’analyse exploratoire et prépare une interface Streamlit pour tester le moteur plus facilement.

Dataset du projet :
https://www.kaggle.com/datasets/karnikakapoor/ml-in-healthcare-patent-data

## Installation

Le projet utilise Python 3.13. Il est recommandé de créer un environnement dédié avant d’installer les dépendances.

Avec Conda :

```bash
conda create -n TF-IDF python=3.13
conda activate TF-IDF
pip install -r requirements.txt
```

Avec `venv` :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ressources NLTK

Certaines fonctionnalités de traitement du texte utilisent NLTK. Après l’installation des dépendances, télécharger les ressources nécessaires :

```bash
python -m nltk.downloader wordnet
python -m nltk.downloader omw-1.4
```

## Lancement

Pour exécuter le script actuel de nettoyage des données :

```bash
python main.py
```

L’objectif du projet est d’utiliser Streamlit pour l’interface. Dès que l’application Streamlit sera disponible, elle pourra être lancée avec :

```bash
streamlit run app.py
```

## Notebooks

Les notebooks d’exemple et d’analyse se trouvent dans `src/exemple/` et `src/analyse exploratoire/`. Pour les ouvrir :

```bash
jupyter lab
```
