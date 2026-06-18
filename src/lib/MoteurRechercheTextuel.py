import math
import pickle
import time


class MoteurRechercheTextuel:
    """
    Moteur de recherche textuel du projet.

    Le moteur reçoit un corpus sous forme `{doc_id: texte}`, construit plusieurs
    structures de recherche, puis renvoie des couples `(doc_id, score)` triés par
    pertinence décroissante.

    Structures utilisées :
    - index inversé classique pour TF-IDF et BM25 ;
    - index lemmatisé pour rapprocher des formes comme `robots` et `robot` ;
    - vecteurs creux SPLADE pour la recherche sparse neuronale.
    """

    def __init__(self):
        self.index = {}  # {terme: [(doc_id, fréquence), ...]}
        self.documents = {}  # {doc_id: texte}
        self.tailles_documents = {}  # {doc_id: nombre_de_mots}
        self.longueur_moyenne_documents = 0.0

        self.index_lemmatise = None
        self.tailles_documents_lemmatise = None
        self.longueur_moyenne_documents_lemmatise = 0.0

        self.corpus_vectors = {}
        self.splade_encoder = None

        self._wordnet = None
        self._wordnet_disponible = None
        self._lemmatizer = None

    def _nettoyer_texte(self, texte):
        """
        Normalise un texte en liste de mots exploitables.

        La fonction met en minuscules, remplace la ponctuation par des espaces,
        puis retire des stopwords anglais peu discriminants. Exemple :
        `"The robot-cleaner, using AI!"` devient environ `["robot", "cleaner", "ai"]`.
        """
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
            "the", "to", "was", "were", "with", "will", "we", "you", "your",
            "this", "these", "those", "they", "their", "them", "than", "then",
            "our", "us", "do", "does", "did", "not", "but", "if", "so", "also",
            "can", "could", "should", "would", "about", "how", "what", "when",
            "where", "why", "which", "all", "any", "each", "both", "either",
            "neither", "such", "more", "most", "less", "few", "many", "some",
            "into", "over", "under", "between", "after", "before", "during",
            "while", "within", "without", "using", "used", "through", "because",
        }

        texte_nettoye = texte.lower()
        ponctuation = [
            ".", ",", "!", "?", "'", "-", "«", "»", ":", ";", "(", ")",
            "[", "]", "{", "}", "/", "\\", "*", "_", "#", "@", "&", "%",
            "$", "~", "^", "=", "+", "<", ">", "|", "`", '"',
        ]
        for caractere in ponctuation:
            texte_nettoye = texte_nettoye.replace(caractere, " ")

        return [mot for mot in texte_nettoye.split() if mot not in stopwords]

    def indexer_corpus(self, corpus):
        """
        Construit l'index inversé principal à partir du corpus.

        Streamlit charge le CSV, assemble `Title + Abstract`, puis transmet ici
        un dictionnaire `{doc_id: texte_du_brevet}`. Pour chaque document, le
        moteur nettoie le texte, compte les fréquences locales des mots, puis
        remplit l'index inversé `{terme: [(doc_id, fréquence), ...]}`.

        Exemple : si `robot` apparaît 3 fois dans le document 0, l'index contient
        une entrée de type `"robot": [(0, 3), ...]`.
        """
        self.documents = corpus
        self.index = {}
        self.tailles_documents = {}
        self.index_lemmatise = None
        self.tailles_documents_lemmatise = None
        self.longueur_moyenne_documents_lemmatise = 0.0
        mots_par_document = {}

        for doc_id in sorted(corpus.keys()):
            mots = self._nettoyer_texte(corpus[doc_id])
            mots_par_document[doc_id] = mots
            self.tailles_documents[doc_id] = len(mots)
            if not mots:
                continue

            frequences_locales = {}
            for mot in mots:
                frequences_locales[mot] = frequences_locales.get(mot, 0) + 1

            for mot, freq in frequences_locales.items():
                if mot not in self.index:
                    self.index[mot] = []
                self.index[mot].append((doc_id, freq))

        self.longueur_moyenne_documents = self._calculer_longueur_moyenne_documents()
        self.corpus_vectors = {}
        self._indexer_corpus_lemmatise(mots_par_document)

    def indexer_corpus_splade(self, corpus, encoder=None, batch_size=32):
        """
        Indexe le corpus avec SPLADE sous forme de vecteurs creux.

        SPLADE ne travaille pas directement avec les fréquences de mots comme
        TF-IDF/BM25. Il encode chaque document avec un modèle de langage et produit
        un vecteur sparse `{dimension: poids}`. La recherche SPLADE compare ensuite
        le vecteur de la requête aux vecteurs des documents par produit scalaire.
        """
        if encoder is not None:
            self.splade_encoder = encoder

        if self.splade_encoder is None:
            try:
                import torch
                from pinecone_text.sparse import SpladeEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "La bibliothèque pinecone-text avec SPLADE n'est pas installée. "
                    "Installez avec `pip install \"pinecone-text[splade]\"` et `pip install torch`."
                ) from exc

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.splade_encoder = SpladeEncoder(device=device)

        self.corpus_vectors = {}
        ordered_doc_ids = sorted(corpus.keys())
        total_docs = len(ordered_doc_ids)
        start_time = time.perf_counter()

        for start in range(0, total_docs, batch_size):
            batch_doc_ids = ordered_doc_ids[start:start + batch_size]
            batch_texts = [corpus[doc_id] for doc_id in batch_doc_ids]
            batch_vectors = self.splade_encoder.encode_documents(batch_texts)

            for doc_id, vector in zip(batch_doc_ids, batch_vectors):
                if not vector:
                    self.corpus_vectors[doc_id] = {}
                    continue

                indices = vector.get("indices", [])
                values = vector.get("values", [])
                self.corpus_vectors[doc_id] = {
                    int(index): float(value)
                    for index, value in zip(indices, values)
                    if value != 0
                }

            processed_docs = min(start + batch_size, total_docs)
            elapsed = time.perf_counter() - start_time
            progress = processed_docs / total_docs if total_docs else 1.0
            percentage = progress * 100
            speed = processed_docs / elapsed if elapsed > 0 else 0.0
            if processed_docs % 100 == 0 or processed_docs == total_docs:
                print(
                    f"Indexation SPLADE en cours... {processed_docs}/{total_docs} "
                    f"documents traités ({percentage:.1f}%) - "
                    f"{speed:.2f} doc/s - {elapsed:.1f}s"
                )

        return self.corpus_vectors

    def sauvegarder_index(self, chemin):
        """Sauvegarde l'index SPLADE calculé pour éviter de le reconstruire."""
        with open(chemin, "wb") as fichier:
            pickle.dump(self.corpus_vectors, fichier)

    def charger_index(self, chemin):
        """Charge un index SPLADE sauvegardé. Retourne False si le fichier est absent ou invalide."""
        try:
            with open(chemin, "rb") as fichier:
                self.corpus_vectors = pickle.load(fichier)
            return True
        except (FileNotFoundError, EOFError, pickle.PickleError):
            self.corpus_vectors = {}
            return False

    def _indexer_corpus_lemmatise(self, mots_par_document):
        """
        Construit l'index lemmatisé au moment de l'indexation.

        Si WordNet n'est pas disponible, on réutilise l'index brut pour que
        l'application reste utilisable.
        """
        if self._obtenir_lemmatizer() is None:
            self.index_lemmatise = self.index
            self.tailles_documents_lemmatise = self.tailles_documents
            self.longueur_moyenne_documents_lemmatise = self.longueur_moyenne_documents
            return

        mots_lemmatises_par_document = {
            doc_id: [self._lemmatiser_mot(mot) for mot in mots]
            for doc_id, mots in mots_par_document.items()
        }
        self.index_lemmatise, self.tailles_documents_lemmatise = self._indexer_mots(
            mots_lemmatises_par_document
        )
        self.longueur_moyenne_documents_lemmatise = (
            self._calculer_longueur_moyenne_depuis_tailles(
                self.tailles_documents_lemmatise
            )
        )

    def _intersecter_deux_listes(self, liste1, liste2):
        """
        Intersecte deux posting lists triées par `doc_id`.

        Cette méthode sert à appliquer une requête AND : un document ne reste
        candidat que s'il apparaît dans toutes les listes de termes recherchés.
        """
        intersection = []
        i, j = 0, 0
        while i < len(liste1) and j < len(liste2):
            if liste1[i][0] == liste2[j][0]:
                intersection.append(liste1[i])
                i += 1
                j += 1
            elif liste1[i][0] < liste2[j][0]:
                i += 1
            else:
                j += 1
        return intersection

    def requete_and(self, requete_texte):
        """
        Retourne les documents contenant tous les mots de la requête.

        Exemple : pour `robot cleaner`, le moteur intersecte les documents contenant
        `robot` avec ceux contenant `cleaner`. Cette étape ne classe pas encore les
        documents ; elle filtre seulement les candidats.
        """
        mots_requete = self._nettoyer_texte(requete_texte)
        if not mots_requete:
            return []
        for mot in mots_requete:
            if mot not in self.index:
                return []
        resultats = self.index[mots_requete[0]]
        for mot in mots_requete[1:]:
            resultats = self._intersecter_deux_listes(resultats, self.index[mot])
        return resultats

    def _calculer_idf(self, terme):
        """
        Calcule l'IDF TF-IDF du terme dans l'index principal.

        Exemple : si `robot` apparaît dans peu de brevets, son IDF sera élevé ;
        s'il apparaît presque partout, son IDF sera plus faible.
        """
        return self._calculer_idf_depuis_index(terme, self.index)

    def _calculer_longueur_moyenne_documents(self):
        """
        Calcule la longueur moyenne des documents non vides.

        Exemple : si trois documents contiennent 100, 150 et 200 mots, la longueur
        moyenne vaut 150. BM25 utilise cette valeur pour corriger les documents
        très courts ou très longs.
        """
        tailles_non_vides = [
            taille
            for taille in self.tailles_documents.values()
            if taille > 0
        ]
        if not tailles_non_vides:
            return 0.0
        return sum(tailles_non_vides) / len(tailles_non_vides)

    def _calculer_idf_bm25(self, terme):
        """
        Calcule l'IDF BM25 du terme dans l'index principal.

        Comme pour TF-IDF, un mot rare pèse plus qu'un mot fréquent, mais BM25
        utilise une formule d'IDF légèrement différente.
        """
        return self._calculer_idf_bm25_depuis_index(terme, self.index)

    def _calculer_idf_depuis_index(self, terme, index):
        """
        Calcule l'IDF TF-IDF dans l'index fourni.

        `df_t` est le nombre de documents contenant le terme. Plus ce nombre est
        faible, plus le terme est rare et donc discriminant.

        Exemple simplifié : si le corpus contient 1 000 documents et que `robot`
        apparaît dans 10 documents, `robot` est considéré comme plus informatif
        que `system`, qui apparaîtrait dans 700 documents.
        """
        N = len(self.documents)
        df_t = len(index.get(terme, []))

        if df_t == 0:
            return 0.0

        return math.log((N / (df_t + 1))) + 1

    def _calculer_idf_bm25_depuis_index(self, terme, index):
        """
        Calcule l'IDF spécifique à BM25 dans l'index fourni.

        Exemple : le terme `cleaner` reçoit un poids plus fort s'il apparaît dans
        peu de documents, car il aide davantage à distinguer un brevet précis.
        """
        N = len(self.documents)
        df_t = len(index.get(terme, []))

        if N == 0 or df_t == 0:
            return 0.0

        return math.log(1 + ((N - df_t + 0.5) / (df_t + 0.5)))

    def _frequence_terme_document(self, terme, doc_id, index=None):
        """
        Retourne la fréquence brute d'un terme dans un document.

        Exemple : si `robot` apparaît trois fois dans le document 42, la fonction
        retourne 3 pour `_frequence_terme_document("robot", 42)`.
        """
        if index is None:
            index = self.index

        for d_id, freq in index.get(terme, []):
            if d_id == doc_id:
                return freq
        return 0

    def _obtenir_wordnet(self):
        """Charge WordNet si la ressource est disponible localement."""
        if self._wordnet_disponible is False:
            return None
        if self._wordnet_disponible is True:
            return self._wordnet

        try:
            from nltk.corpus import wordnet

            wordnet.synsets("test")
        except Exception:
            self._wordnet = None
            self._wordnet_disponible = False
            return None

        self._wordnet = wordnet
        self._wordnet_disponible = True
        return self._wordnet

    def _obtenir_lemmatizer(self):
        """Retourne le lemmatizer WordNet si NLTK/WordNet sont disponibles."""
        if self._obtenir_wordnet() is None:
            return None
        if self._lemmatizer is not None:
            return self._lemmatizer

        try:
            from nltk.stem import WordNetLemmatizer
        except Exception:
            return None

        self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer

    def _lemmatiser_mot(self, mot):
        """Lemmatise un mot anglais, ou le retourne tel quel si WordNet manque."""
        lemmatizer = self._obtenir_lemmatizer()
        if lemmatizer is None:
            return mot

        try:
            lemme_nom = lemmatizer.lemmatize(mot, pos="n")
            return lemmatizer.lemmatize(lemme_nom, pos="v")
        except Exception:
            return mot

    def _indexer_mots(self, mots_par_document):
        """Construit un index inversé à partir d'un dictionnaire `{doc_id: [mots]}`."""
        index = {}
        tailles_documents = {}

        for doc_id in sorted(mots_par_document.keys()):
            mots = mots_par_document[doc_id]
            tailles_documents[doc_id] = len(mots)
            if len(mots) == 0:
                continue

            frequences_locales = {}
            for mot in mots:
                frequences_locales[mot] = frequences_locales.get(mot, 0) + 1

            for mot, freq in frequences_locales.items():
                if mot not in index:
                    index[mot] = []
                index[mot].append((doc_id, freq))

        return index, tailles_documents

    def _calculer_longueur_moyenne_depuis_tailles(self, tailles_documents):
        """
        Calcule une longueur moyenne à partir d'un dictionnaire de tailles.

        Cette version sert aussi pour l'index lemmatisé, où les tailles peuvent
        être recalculées après transformation des mots.
        """
        tailles_non_vides = [
            taille
            for taille in tailles_documents.values()
            if taille > 0
        ]
        if not tailles_non_vides:
            return 0.0
        return sum(tailles_non_vides) / len(tailles_non_vides)

    def _obtenir_index_lemmatise(self):
        """Renvoie l'index lemmatisé, en le construisant si nécessaire."""
        if self.index_lemmatise is not None:
            return (
                self.index_lemmatise,
                self.tailles_documents_lemmatise,
                self.longueur_moyenne_documents_lemmatise,
            )

        if self._obtenir_lemmatizer() is None:
            self.index_lemmatise = self.index
            self.tailles_documents_lemmatise = self.tailles_documents
            self.longueur_moyenne_documents_lemmatise = self.longueur_moyenne_documents
            return (
                self.index_lemmatise,
                self.tailles_documents_lemmatise,
                self.longueur_moyenne_documents_lemmatise,
            )

        mots_par_document = {}
        for doc_id, texte in self.documents.items():
            mots_par_document[doc_id] = [
                self._lemmatiser_mot(mot)
                for mot in self._nettoyer_texte(texte)
            ]

        self.index_lemmatise, self.tailles_documents_lemmatise = self._indexer_mots(
            mots_par_document
        )
        self.longueur_moyenne_documents_lemmatise = (
            self._calculer_longueur_moyenne_depuis_tailles(
                self.tailles_documents_lemmatise
            )
        )

        return (
            self.index_lemmatise,
            self.tailles_documents_lemmatise,
            self.longueur_moyenne_documents_lemmatise,
        )

    def _synonymes_mot(self, mot, limite=12):
        """
        Retourne le mot initial et ses synonymes WordNet en un seul mot.

        La limite évite d'élargir excessivement la requête, car trop de synonymes
        peuvent augmenter le bruit.
        """
        synonymes = [mot]
        wordnet = self._obtenir_wordnet()
        if wordnet is None:
            return synonymes

        try:
            for synset in wordnet.synsets(mot):
                for lemme in synset.lemma_names():
                    synonyme = lemme.lower().replace("_", " ")
                    mots_synonyme = self._nettoyer_texte(synonyme)
                    if len(mots_synonyme) != 1:
                        continue
                    synonyme = mots_synonyme[0]
                    if synonyme not in synonymes:
                        synonymes.append(synonyme)
                    if len(synonymes) >= limite:
                        return synonymes
        except Exception:
            return [mot]

        return synonymes

    def _groupes_termes_synonymes(self, requete_texte):
        """
        Transforme une requête en groupes de synonymes.

        Exemple : `smart robot` peut devenir
        `[["smart", "intelligent"], ["robot", "automaton"]]`.
        """
        groupes = []
        for mot in self._nettoyer_texte(requete_texte):
            lemme = self._lemmatiser_mot(mot)
            groupes.append([
                self._lemmatiser_mot(synonyme)
                for synonyme in self._synonymes_mot(lemme)
            ])
        return groupes

    def _groupes_termes_lemmatises(self, requete_texte):
        """Transforme chaque mot de la requête en un groupe contenant son lemme."""
        return [
            [self._lemmatiser_mot(mot)]
            for mot in self._nettoyer_texte(requete_texte)
        ]

    def _documents_candidats_groupes(self, groupes_termes, index):
        """
        Filtre les documents candidats pour les recherches par groupes.

        La logique est un AND entre groupes et un OR dans chaque groupe. Pour
        `[["smart", "intelligent"], ["robot"]]`, un document doit contenir
        `smart` OU `intelligent`, ET contenir `robot`.
        """
        ensembles_par_groupe = []
        for groupe in groupes_termes:
            doc_ids_groupe = set()
            for terme in groupe:
                for doc_id, _ in index.get(terme, []):
                    doc_ids_groupe.add(doc_id)
            if not doc_ids_groupe:
                return []
            ensembles_par_groupe.append(doc_ids_groupe)

        doc_ids = set.intersection(*ensembles_par_groupe)
        return sorted(doc_ids)

    def _chercher_tfidf_groupes(self, groupes_termes, index, tailles_documents):
        """
        Recherche TF-IDF avec groupes de synonymes ou de lemmes.

        Chaque groupe représente une position de la requête. Si un groupe contient
        `["smart", "intelligent"]`, on calcule le score TF-IDF possible de chaque
        terme présent dans le document, puis on garde seulement le meilleur score
        du groupe. Cela évite de compter double deux synonymes du même mot.

        Exemple court : dans un document de 100 mots, `smart` apparaît 2 fois avec
        IDF=3, donc son score vaut `(2/100) * 3 = 0.06`. Si `intelligent` vaut
        0.04 dans le même groupe, le groupe ajoute 0.06 au score total.
        """
        groupes_termes = [groupe for groupe in groupes_termes if groupe]
        if not groupes_termes:
            return []

        doc_ids_filtres = self._documents_candidats_groupes(groupes_termes, index)
        if not doc_ids_filtres:
            return []

        idfs = {
            terme: self._calculer_idf_depuis_index(terme, index)
            for groupe in groupes_termes
            for terme in groupe
        }
        scores_documents = {}

        for doc_id in doc_ids_filtres:
            score_total = 0.0
            taille_doc = tailles_documents[doc_id]
            if taille_doc == 0:
                continue

            for groupe in groupes_termes:
                meilleurs_scores = []
                for terme in groupe:
                    frequence_brute = self._frequence_terme_document(
                        terme, doc_id, index
                    )
                    if frequence_brute == 0:
                        continue
                    tf = frequence_brute / taille_doc
                    meilleurs_scores.append(tf * idfs[terme])
                if meilleurs_scores:
                    score_total += max(meilleurs_scores)

            scores_documents[doc_id] = score_total

        return sorted(scores_documents.items(), key=lambda x: x[1], reverse=True)

    def _chercher_bm25_groupes(
        self,
        groupes_termes,
        index,
        tailles_documents,
        longueur_moyenne_documents,
        k1=1.5,
        b=0.75,
    ):
        """
        Recherche BM25 avec groupes de synonymes ou de lemmes.

        La logique des groupes est la même que pour TF-IDF : un seul score est
        retenu par groupe. La différence vient de la formule BM25, qui combine
        IDF, fréquence saturée (`k1`) et correction de longueur (`b`).
        """
        groupes_termes = [groupe for groupe in groupes_termes if groupe]
        if not groupes_termes or longueur_moyenne_documents == 0:
            return []

        doc_ids_filtres = self._documents_candidats_groupes(groupes_termes, index)
        if not doc_ids_filtres:
            return []

        idfs = {
            terme: self._calculer_idf_bm25_depuis_index(terme, index)
            for groupe in groupes_termes
            for terme in groupe
        }
        scores_documents = {}

        for doc_id in doc_ids_filtres:
            score_total = 0.0
            taille_doc = tailles_documents[doc_id]

            for groupe in groupes_termes:
                meilleurs_scores = []
                for terme in groupe:
                    frequence_brute = self._frequence_terme_document(
                        terme, doc_id, index
                    )
                    if frequence_brute == 0:
                        continue

                    normalisation_longueur = 1 - b + b * (
                        taille_doc / longueur_moyenne_documents
                    )
                    numerateur = frequence_brute * (k1 + 1)
                    denominateur = frequence_brute + k1 * normalisation_longueur
                    meilleurs_scores.append(
                        idfs[terme] * (numerateur / denominateur)
                    )
                if meilleurs_scores:
                    score_total += max(meilleurs_scores)

            scores_documents[doc_id] = score_total

        return sorted(scores_documents.items(), key=lambda x: x[1], reverse=True)

    def chercher_tfidf(self, requete_texte):
        """
        Recherche TF-IDF pure.

        La requête est nettoyée, puis traitée comme un AND : seuls les documents
        contenant tous les mots sont candidats. Le score d'un document est la somme
        des contributions `TF * IDF` de chaque mot de la requête.
        """
        mots_requete = self._nettoyer_texte(requete_texte)

        documents_candidats = self.requete_and(requete_texte)
        if not documents_candidats:
            return []

        doc_ids_filtrés = [couple[0] for couple in documents_candidats]
        scores_documents = {}
        idfs_requete = {mot: self._calculer_idf(mot) for mot in mots_requete}

        for doc_id in doc_ids_filtrés:
            score_total = 0.0

            for mot in mots_requete:
                frequence_brute = 0
                for d_id, freq in self.index[mot]:
                    if d_id == doc_id:
                        frequence_brute = freq
                        break

                taille_doc = self.tailles_documents[doc_id]
                tf = frequence_brute / taille_doc
                score_total += tf * idfs_requete[mot]

            scores_documents[doc_id] = score_total

        return sorted(scores_documents.items(), key=lambda x: x[1], reverse=True)

    def chercher_bm25(self, requete_texte, k1=1.5, b=0.75):
        """
        Recherche BM25 pure.

        BM25 utilise les mêmes documents candidats que TF-IDF pur, mais son score
        limite l'effet des répétitions avec `k1` et corrige la longueur des documents
        avec `b`.
        """
        mots_requete = self._nettoyer_texte(requete_texte)
        if not mots_requete or self.longueur_moyenne_documents == 0:
            return []

        documents_candidats = self.requete_and(requete_texte)
        if not documents_candidats:
            return []

        doc_ids_filtres = [couple[0] for couple in documents_candidats]
        idfs_requete = {mot: self._calculer_idf_bm25(mot) for mot in mots_requete}
        scores_documents = {}

        for doc_id in doc_ids_filtres:
            score_total = 0.0
            taille_doc = self.tailles_documents[doc_id]

            for mot in mots_requete:
                frequence_brute = self._frequence_terme_document(mot, doc_id)
                if frequence_brute == 0:
                    continue

                normalisation_longueur = 1 - b + b * (
                    taille_doc / self.longueur_moyenne_documents
                )
                numerateur = frequence_brute * (k1 + 1)
                denominateur = frequence_brute + k1 * normalisation_longueur
                score_total += idfs_requete[mot] * (numerateur / denominateur)

            scores_documents[doc_id] = score_total

        return sorted(scores_documents.items(), key=lambda x: x[1], reverse=True)

    def wordnet_est_disponible(self):
        """Indique si les ressources WordNet locales sont disponibles."""
        return self._obtenir_wordnet() is not None

    def chercher_tfidf_lemmatise(self, requete_texte):
        """
        Recherche TF-IDF sur l'index lemmatisé.

        La lemmatisation rapproche des variantes grammaticales comme `robots` et
        `robot`. Elle change les termes comparés, mais pas la formule TF-IDF.
        """
        donnees_index = self._obtenir_index_lemmatise()
        index, tailles_documents, _ = donnees_index
        groupes_termes = self._groupes_termes_lemmatises(requete_texte)

        return self._chercher_tfidf_groupes(groupes_termes, index, tailles_documents)

    def chercher_bm25_lemmatise(self, requete_texte, k1=1.5, b=0.75):
        """
        Recherche BM25 sur l'index lemmatisé.

        Les mots des documents et de la requête sont comparés sous forme de lemmes,
        puis les documents candidats sont classés avec BM25.
        """
        donnees_index = self._obtenir_index_lemmatise()
        index, tailles_documents, longueur_moyenne_documents = donnees_index
        groupes_termes = self._groupes_termes_lemmatises(requete_texte)

        return self._chercher_bm25_groupes(
            groupes_termes,
            index,
            tailles_documents,
            longueur_moyenne_documents,
            k1=k1,
            b=b,
        )

    def chercher_tfidf_synonymes(self, requete_texte):
        """
        Recherche TF-IDF avec expansion par synonymes WordNet.

        Chaque mot de la requête devient un groupe de termes équivalents. Le document
        doit valider chaque groupe, et un seul score est retenu par groupe pour éviter
        qu'un synonyme compte double.
        """
        donnees_index = self._obtenir_index_lemmatise()
        index, tailles_documents, _ = donnees_index
        groupes_termes = self._groupes_termes_synonymes(requete_texte)

        return self._chercher_tfidf_groupes(groupes_termes, index, tailles_documents)

    def chercher_bm25_synonymes(self, requete_texte, k1=1.5, b=0.75):
        """
        Recherche BM25 avec expansion par synonymes WordNet.

        La logique des groupes est identique à TF-IDF + synonymes ; seul le calcul
        du score change, avec la formule BM25.
        """
        donnees_index = self._obtenir_index_lemmatise()
        index, tailles_documents, longueur_moyenne_documents = donnees_index
        groupes_termes = self._groupes_termes_synonymes(requete_texte)

        return self._chercher_bm25_groupes(
            groupes_termes,
            index,
            tailles_documents,
            longueur_moyenne_documents,
            k1=k1,
            b=b,
        )
