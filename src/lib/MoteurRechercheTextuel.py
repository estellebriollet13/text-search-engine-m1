import math
import pickle
import time

class MoteurRechercheTextuel:
    def __init__(self):
        self.index = {}  # { terme -> [(doc_id, freq), ...] }
        self.documents = {}  # { doc_id -> "texte" }
        self.tailles_documents = {}  # { doc_id -> nombre_de_mots }
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
        Transforme une chaîne de caractères en une liste de mots nettoyés.
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
            "while", "within", "without", "using", "used", "through", "because"
        }

        texte_nettoye = texte.lower()
        for caractere in [".", ",", "!", "?", "'", "-", "«", "»", ":", ";", "(", ")", "[", "]", "{", "}", "/", "\\", "*", "_", "#", "@", "&", "%", "$", "~", "^", "=", "+", "<", ">", "|", "`", '"']:
            texte_nettoye = texte_nettoye.replace(caractere, " ")

        mots = [mot for mot in texte_nettoye.split() if mot not in stopwords]
        return mots

    def indexer_corpus(self, corpus):
        """
        Prend en entrée un dictionnaire { doc_id (int): "texte du document" (str) }
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
            if len(mots) == 0:
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
        Indexe le corpus en vecteurs creux SPLADE en mémoire.
        Chaque vecteur est stocké sous la forme {index: poids}.
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
        """
        Sauvegarde le dictionnaire des vecteurs SPLADE sur disque.
        """
        with open(chemin, "wb") as fichier:
            pickle.dump(self.corpus_vectors, fichier)

    def charger_index(self, chemin):
        """
        Charge un dictionnaire de vecteurs SPLADE depuis le disque.
        Retourne True si le chargement a réussi, sinon False.
        """
        try:
            with open(chemin, "rb") as fichier:
                self.corpus_vectors = pickle.load(fichier)
            return True
        except (FileNotFoundError, EOFError, pickle.PickleError):
            self.corpus_vectors = {}
            return False

    def _indexer_corpus_lemmatise(self, mots_par_document):
        """
        Construit l'index lemmatise au moment de l'indexation.
        Si WordNet n'est pas disponible, on reutilise l'index brut.
        """
        if self._obtenir_lemmatizer() is None:
            self.index_lemmatise = self.index
            self.tailles_documents_lemmatise = self.tailles_documents
            self.longueur_moyenne_documents_lemmatise = self.longueur_moyenne_documents
            return

        mots_lemmatises_par_document = {
            doc_id: [
                self._lemmatiser_mot(mot)
                for mot in mots
            ]
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
        Algorithme à deux pointeurs pour intersecter deux listes triées.
        Complexité optimale : O(|liste1| + |liste2|)
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
        Traite une requête multi-mots comme un AND logique.
        Exemple : "hachage inversé" -> cherche les docs avec les DEUX mots.
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
        """Calcule l'IDF d'un terme avec l'ajustement pour éviter les divisions par 0."""
        return self._calculer_idf_depuis_index(terme, self.index)

    def _calculer_longueur_moyenne_documents(self):
        """Calcule la longueur moyenne des documents indexés."""
        tailles_non_vides = [
            taille
            for taille in self.tailles_documents.values()
            if taille > 0
        ]
        if not tailles_non_vides:
            return 0.0
        return sum(tailles_non_vides) / len(tailles_non_vides)

    def _calculer_idf_bm25(self, terme):
        """Calcule l'IDF utilise par BM25."""
        return self._calculer_idf_bm25_depuis_index(terme, self.index)

    def _calculer_idf_depuis_index(self, terme, index):
        """Calcule l'IDF TF-IDF dans l'index fourni."""
        N = len(self.documents)
        df_t = len(index.get(terme, []))

        if df_t == 0:
            return 0.0

        return math.log((N / (df_t + 1))) + 1

    def _calculer_idf_bm25_depuis_index(self, terme, index):
        """Calcule l'IDF BM25 dans l'index fourni."""
        N = len(self.documents)
        df_t = len(index.get(terme, []))

        if N == 0 or df_t == 0:
            return 0.0

        return math.log(1 + ((N - df_t + 0.5) / (df_t + 0.5)))

    def _frequence_terme_document(self, terme, doc_id, index=None):
        """Retourne la frequence brute d'un terme dans un document."""
        if index is None:
            index = self.index

        for d_id, freq in index.get(terme, []):
            if d_id == doc_id:
                return freq
        return 0

    def _obtenir_wordnet(self):
        """Retourne WordNet si la ressource locale est disponible."""
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
        """Retourne le lemmatizer WordNet si NLTK et WordNet sont disponibles."""
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
        """Construit un index inverse depuis {doc_id: [mots]}."""
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
        """Calcule la longueur moyenne depuis un dictionnaire de tailles."""
        tailles_non_vides = [
            taille
            for taille in tailles_documents.values()
            if taille > 0
        ]
        if not tailles_non_vides:
            return 0.0
        return sum(tailles_non_vides) / len(tailles_non_vides)

    def _obtenir_index_lemmatise(self):
        """Construit puis met en cache un index base sur les lemmes."""
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
        """Retourne le mot et ses synonymes WordNet presents dans l'index."""
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
        """Prepare les groupes de termes equivalents pour une requete."""
        groupes = []
        for mot in self._nettoyer_texte(requete_texte):
            lemme = self._lemmatiser_mot(mot)
            groupes.append([
                self._lemmatiser_mot(synonyme)
                for synonyme in self._synonymes_mot(lemme)
            ])
        return groupes

    def _groupes_termes_lemmatises(self, requete_texte):
        """Prepare les termes lemmatises de la requete."""
        return [
            [self._lemmatiser_mot(mot)]
            for mot in self._nettoyer_texte(requete_texte)
        ]

    def _documents_candidats_groupes(self, groupes_termes, index):
        """
        Retourne les documents qui contiennent au moins un terme de chaque groupe.
        Chaque groupe represente un mot de requete et ses variantes.
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
        """Recherche TF-IDF avec des groupes de termes equivalents."""
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
        """Recherche BM25 avec des groupes de termes equivalents."""
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
        Exécute la requête AND, calcule le score TF-IDF des documents correspondants
        et les renvoie triés par pertinence décroissante.
        """
        mots_requete = self._nettoyer_texte(requete_texte)
        
        # 1. On récupère d'abord les documents candidats (ceux qui contiennent TOUS les mots)
        documents_candidats = self.requete_and(requete_texte)
        if not documents_candidats:
            return []

        # Pour faciliter le calcul, on extrait juste la liste des doc_ids uniques filtrés
        doc_ids_filtrés = [couple[0] for couple in documents_candidats]

        # 2. Calcul du score pour chaque document candidat
        scores_documents = {}
        
        # On calcule à l'avance l'IDF de chaque mot de la requête pour gagner du temps
        idfs_requete = {mot: self._calculer_idf(mot) for mot in mots_requete}

        for doc_id in doc_ids_filtrés:
            score_total = 0.0
            
            for mot in mots_requete:
                # On doit retrouver la fréquence brute du mot dans ce document précis
                # On cherche dans la posting list du mot
                frequence_brute = 0
                for d_id, freq in self.index[mot]:
                    if d_id == doc_id:
                        frequence_brute = freq
                        break
                
                # Calcul du TF normalisé pour ce document
                taille_doc = self.tailles_documents[doc_id]
                tf = frequence_brute / taille_doc
                
                # Ajout au score cumulé du document
                score_total += tf * idfs_requete[mot]
                
            scores_documents[doc_id] = score_total

        # 3. Tri des documents par score décroissant
        # sorted renvoie une liste de couples (doc_id, score)
        resultats_tries = sorted(scores_documents.items(), key=lambda x: x[1], reverse=True)
        
        return resultats_tries

    def chercher_bm25(self, requete_texte, k1=1.5, b=0.75):
        """
        Exécute la requête AND, calcule le score BM25 des documents correspondants
        et les renvoie triés par pertinence décroissante.
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

        resultats_tries = sorted(scores_documents.items(), key=lambda x: x[1], reverse=True)

        return resultats_tries

    def wordnet_est_disponible(self):
        """Indique si les ressources WordNet locales sont disponibles."""
        return self._obtenir_wordnet() is not None

    def chercher_tfidf_lemmatise(self, requete_texte):
        """
        Calcule le score TF-IDF sur une version lemmatisée de l'index et de la requête.
        """
        donnees_index = self._obtenir_index_lemmatise()
        index, tailles_documents, _ = donnees_index
        groupes_termes = self._groupes_termes_lemmatises(requete_texte)

        return self._chercher_tfidf_groupes(groupes_termes, index, tailles_documents)

    def chercher_bm25_lemmatise(self, requete_texte, k1=1.5, b=0.75):
        """
        Calcule le score BM25 sur une version lemmatisée de l'index et de la requête.
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
        Calcule le score TF-IDF après expansion de chaque terme avec ses synonymes.
        """
        donnees_index = self._obtenir_index_lemmatise()
        index, tailles_documents, _ = donnees_index
        groupes_termes = self._groupes_termes_synonymes(requete_texte)

        return self._chercher_tfidf_groupes(groupes_termes, index, tailles_documents)

    def chercher_bm25_synonymes(self, requete_texte, k1=1.5, b=0.75):
        """
        Calcule le score BM25 après expansion de chaque terme avec ses synonymes.
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
