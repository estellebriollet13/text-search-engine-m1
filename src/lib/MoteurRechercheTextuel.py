import math

class MoteurRechercheTextuel:
    def __init__(self):
        self.index = {}  # { terme -> [(doc_id, freq), ...] }
        self.documents = {}  # { doc_id -> "texte" }
        self.tailles_documents = {}  # { doc_id -> nombre_de_mots }
        self.longueur_moyenne_documents = 0.0

    def _nettoyer_texte(self, texte):
        """
        Transforme une chaîne de caractères en une liste de mots nettoyés.
        """
        texte_nettoye = texte.lower()
        for caractere in [".", ",", "!", "?", "'", "-", "«", "»", ":", ";"]:
            texte_nettoye = texte_nettoye.replace(caractere, " ")
        return texte_nettoye.split()

    def indexer_corpus(self, corpus):
        """
        Prend en entrée un dictionnaire { doc_id (int): "texte du document" (str) }
        """
        self.documents = corpus
        self.index = {}
        self.tailles_documents = {}
        for doc_id in sorted(corpus.keys()):
            mots = self._nettoyer_texte(corpus[doc_id])
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
        N = len(self.documents)
        # df_t est la longueur de la posting list pour ce terme
        df_t = len(self.index.get(terme, []))
        
        if df_t == 0:
            return 0.0
            
        # Formule lissée classique
        return math.log((N / (df_t + 1))) + 1

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
        N = len(self.documents)
        df_t = len(self.index.get(terme, []))

        if N == 0 or df_t == 0:
            return 0.0

        return math.log(1 + ((N - df_t + 0.5) / (df_t + 0.5)))

    def _frequence_terme_document(self, terme, doc_id):
        """Retourne la frequence brute d'un terme dans un document."""
        for d_id, freq in self.index.get(terme, []):
            if d_id == doc_id:
                return freq
        return 0

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
