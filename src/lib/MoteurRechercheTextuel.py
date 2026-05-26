class MoteurRechercheTextuel:
    def __init__(self):
        # L'index inversé : un dictionnaire où la clé est un mot (str)
        # et la valeur est la liste des IDs de documents (list of int)
        self.index = {}
       
        # Un dictionnaire pour conserver le texte original des documents
        self.documents = {}
 
    def _nettoyer_texte(self, texte):
        """
        Transforme une chaîne de caractères en une liste de mots nettoyés.
        """
        # 1. Passage en minuscules
        texte_nettoye = texte.lower()
       
        # 2. Remplacement des signes de ponctuation par des espaces
        # (indispensable pour gérer les apostrophes comme dans "l'index")
        pour_remplacement = [".", ",", "!", "?", "'", "-", "«", "»", ":", ";"]
        for caractere in pour_remplacement:
            texte_nettoye = texte_nettoye.replace(caractere, " ")
           
        # 3. Découpage par espace (tokenisation)
        mots = texte_nettoye.split()
        return mots
   
    def indexer_corpus(self, corpus):
        """
        Prend en entrée un dictionnaire { doc_id (int): "texte du document" (str) }
        """
        self.documents = corpus
       
        # On parcourt les documents dans l'ordre de leurs IDs
        for doc_id in sorted(corpus.keys()):
            texte = corpus[doc_id]
            mots = self._nettoyer_texte(texte)
           
            # Utiliser set(mots) évite de dupliquer l'ID si un mot
            # apparaît plusieurs fois dans le MÊME document.
            for mot in set(mots):
                if mot not in self.index:
                    self.index[mot] = []
               
                # Comme on trie les clés du corpus, doc_id est inséré
                # de manière strictement croissante !
                self.index[mot].append(doc_id)
   
    def _intersecter_deux_listes(self, liste1, liste2):
        """
        Algorithme à deux pointeurs pour intersecter deux listes triées.
        Complexité optimale : O(|liste1| + |liste2|)
        """
        intersection = []
        i, j = 0, 0  # Nos deux pointeurs
 
        while i < len(liste1) and j < len(liste2):
            if liste1[i] == liste2[j]:
                intersection.append(liste1[i])
                i += 1
                j += 1
            elif liste1[i] < liste2[j]:
                i += 1  # On avance le pointeur de la liste 1 car sa valeur est trop petite
            else:
                j += 1  # On avance le pointeur de la liste 2
               
        return intersection
 
    def requete_and(self, requete_texte):
        """
        Traite une requête multi-mots comme un AND logique.
        Exemple : "hachage inversé" -> cherche les docs avec les DEUX mots.
        """
        mots_requete = self._nettoyer_texte(requete_texte)
        if not mots_requete:
            return []
 
        # Si l'un des mots de la requête n'est pas du tout dans l'index,
        # l'intersection AND est forcément vide.
        for mot in mots_requete:
            if mot not in self.index:
                return []
 
        # On initialise nos résultats avec la liste du premier mot
        resultats = self.index[mots_requete[0]]
 
        # On intersecte successivement avec les listes des autres mots
        for mot in mots_requete[1:]:
            liste_suivante = self.index[mot]
            resultats = self._intersecter_deux_listes(resultats, liste_suivante)
 
        return resultats