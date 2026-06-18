# text-search-engine-m1

Moteur de recherche textuel basé sur un index inversé. Le projet compare plusieurs méthodes de classement des résultats :

- TF-IDF pur
- BM25 pur
- TF-IDF avec synonymes WordNet
- BM25 avec synonymes WordNet
- TF-IDF avec lemmatisation
- BM25 avec lemmatisation
- SPLADE local (vecteurs creux)

Une interface Streamlit permet de lancer des recherches sur un dataset de brevets et de comparer la robustesse des méthodes face à différentes formulations de requête.

Dataset du projet :
https://www.kaggle.com/datasets/karnikakapoor/ml-in-healthcare-patent-data

## Fonctionnement

Le moteur commence par construire un index inversé à partir des mots présents dans les documents. Cet index stocke, pour chaque terme, les documents dans lesquels il apparaît et sa fréquence dans chaque document.

Ensuite, les méthodes de ranking utilisent cet index :

- TF-IDF donne plus de poids aux mots fréquents dans un document mais rares dans le corpus.
- BM25 ajoute une normalisation plus avancée par fréquence et longueur de document.
- La lemmatisation construit aussi un index lemmatisé pour rapprocher des formes comme `robot` et `robots`.
- Les synonymes sont ajoutés côté requête avec WordNet, pour éviter de gonfler inutilement l’index.

## Installation

Le projet utilise Python 3.13. Créer un environnement dédié avant d’installer les dépendances.

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

Pour installer explicitement les dépendances SPLADE avant de lancer l’application :

```bash
pip install "pinecone-text[splade]"
pip install torch
```

## Ressources NLTK

Les onglets avec synonymes et lemmatisation utilisent WordNet. Après l’installation des dépendances, télécharger les ressources NLTK :

```bash
python -m nltk.downloader wordnet
python -m nltk.downloader omw-1.4
```

Ces ressources sont nécessaires pour que les variantes avec synonymes et lemmatisation soient pleinement actives. Le projet ne télécharge pas ces ressources automatiquement au lancement de Streamlit.

## Lancement Streamlit

Depuis la racine du projet :

```bash
python3 -m streamlit run main.py
```

Selon l’environnement, la commande suivante peut aussi fonctionner :

```bash
streamlit run main.py
```

L’application expose plusieurs onglets :

- recherche TF-IDF pure
- recherche BM25 pure
- recherche TF-IDF avec synonymes
- recherche BM25 avec synonymes
- recherche TF-IDF avec lemmatisation
- recherche BM25 avec lemmatisation
- dashboard de comparaison des méthodes

## Dashboard de comparaison

L’onglet `Comparaison` permet de tester un brevet cible contre plusieurs formulations de requête. Il affiche :

- les meilleures méthodes
- le taux de réussite
- le taux de présence dans le top N
- le rang moyen du brevet cible
- une matrice requête x méthode
- le détail des scores et rangs

Le brevet cible par défaut est :

```text
Artificial intelligence robot cleaner and robot cleaning system
```

## Notebooks

Les notebooks d’exemple et d’analyse se trouvent dans :

- `src/exemple/`
- `src/analyse exploratoire/`

Pour les ouvrir :

```bash
jupyter lab
```
