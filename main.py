import csv
import math
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.lib.MoteurRechercheTextuel import MoteurRechercheTextuel


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "patent_analysis_data.csv"
KAGGLE_DATASET_HANDLE = "karnikakapoor/ml-in-healthcare-patent-data"
DATASET_REQUIRED_COLUMNS = {"Patent ID", "Title", "Abstract"}
TEXT_COLUMNS = ("Title", "Abstract")
TARGET_PATENT_TITLE = "Artificial intelligence robot cleaner and robot cleaning system"
VISIBILITY_TOP_RANK = 10
PATENTSBERTA_MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"
PATENTSBERTA_INDEX_PATH = BASE_DIR / "index_patentsberta.pkl"
DEFAULT_COMPARISON_QUERIES = [
    "Artificial intelligence robot cleaner",
    "ARTIFICIAL INTELLIGENCE ROBOT CLEANER",
    "Artificial Intelligence Robot Cleaner",
    "artificial-intelligence robot-cleaner",
    "robot cleaner artificial intelligence",
    "artificial intelligence robot cleaner and robot cleaning system",
    "robot cleaner and robot cleaning system",
    "robot cleaning system artificial intelligence",
    "artificial intelligence robot cleaner system",
    "artificial intelligence robot cleaning system",
    "system",
    "intelligence",
    "artificial",
    "cleaning",
    "robot",
    "artificial intelligence",
    "intelligence system",
    "artificial system",
    "artificial intelligence system",
    "cleaning system",
    "robot system",
    "robots cleaning system",
    "intelligent robot cleaner",
    "AI robot vacuum",
    "automated robot cleaner",
    "automaton cleaner",
    "automaton cleaning system",
    "automaton cleaner system",
    "automaton system",
    "robots cleaners",
    "robots cleaning systems",
    "robots system",
    "cleaners system",
    "cleanser system",
    "golem cleaner",
    "robot cleansing system",
    "robot cleanser system",
    "robot cleanser",
    "robotic cleaning apparatus",
    "autonomous cleaning robot",
    "intelligent cleaning robot system",
    "AI based robot cleaning system",
    "artificial intelligence cleaning apparatus",
    "robot cleaner control method",
    "robot cleaner navigation system",
    "robot cleaning device using artificial intelligence",
    "autonomous floor cleaning robot",
    "smart robotic vacuum cleaner",
    "self moving cleaning robot",
    "method for controlling a robot cleaner",
    "system and method for robot cleaning",
    "cleaning robot using AI",
    "artificial intelligence based cleaning robot",
    "robotic cleaner and cleaning control system",
    "patent robot cleaner artificial intelligence cleaning system",
    "prior art AI robot cleaner navigation cleaning control",
    "autonomous robot cleaner cleaning route determination patent",
    "robotic vacuum cleaner artificial intelligence obstacle detection",
    "AI based cleaning robot control system patent search",
    "robot cleaner sensor control unit cleaning path",
    "smart robot cleaner mapping obstacle avoidance cleaning",
    "invention robot cleaner automatically decides where to clean",
    "robot vacuum that uses artificial intelligence to clean a room",
    "cleaning robot with sensors and automatic route planning",
    "machine that cleans floors by itself using AI",
    "robot cleaner that learns the home and controls cleaning",
]


def main():
    st.set_page_config(
        page_title="Moteur de recherche textuel",
        layout="wide",
    )

    st.title("Moteur de recherche textuel")
    try:
        with st.spinner("Vérification du dataset..."):
            dataset_path = ensure_dataset_available()
    except Exception as error:
        st.error(
            "Impossible de préparer le dataset. L'application s'arrête ici pour éviter de lancer "
            "Streamlit sans données."
        )
        st.info(
            "Vérifiez d'abord les dépendances avec `python3 -m pip install -r requirements.txt`. "
            "Si Kaggle demande une authentification, configurez aussi vos accès Kaggle, puis relancez Streamlit."
        )
        st.warning(str(error))
        st.stop()

    st.caption(f"Dataset chargé depuis : `{dataset_path}`")

    search_methods = get_search_methods()
    tab_names = [method_name for method_name, _ in search_methods]
    tab_objects = st.tabs(tab_names + ["Comparaison"])

    for tab, (method_name, _) in zip(tab_objects[:-1], search_methods):
        with tab:
            render_search_method_tab(method_name)

    with tab_objects[-1]:
        render_comparison_tab()


def render_search_method_tab(method_name):
    tab_renderers = {
        "TF-IDF pur": render_tfidf_tab,
        "BM25 pur": render_bm25_tab,
        "TF-IDF + synonymes": render_tfidf_synonyms_tab,
        "BM25 + synonymes": render_bm25_synonyms_tab,
        "TF-IDF + lemmatisation": render_tfidf_lemmatization_tab,
        "BM25 + lemmatisation": render_bm25_lemmatization_tab,
        "SPLADE": render_splade_tab,
        "BM25 synonymes + SPLADE": render_hybrid_bm25_splade_tab,
        "TF-IDF synonymes + SPLADE": render_hybrid_tfidf_splade_tab,
        "TF-IDF/BM25 synonymes + SPLADE": render_hybrid_tfidf_bm25_splade_tab,
        "PatentSBERTa": render_patentsberta_tab,
        "BM25 synonymes + PatentSBERTa": render_hybrid_bm25_patentsberta_tab,
        "TF-IDF synonymes + PatentSBERTa": render_hybrid_tfidf_patentsberta_tab,
        "TF-IDF/BM25 synonymes + PatentSBERTa": render_hybrid_tfidf_bm25_patentsberta_tab,
    }

    if method_name not in tab_renderers:
        raise ValueError(f"Aucun rendu défini pour la méthode de recherche : {method_name}")

    tab_renderers[method_name]()


def render_tfidf_tab():
    render_search_tab(
        method_key="tfidf",
        score_name="TF-IDF",
        search_callback=lambda moteur, query: moteur.chercher_tfidf(query),
        score_description=(
            "Score : score TF-IDF calculé pour la requête ; plus il est élevé, "
            "plus le document contient fortement les termes recherchés."
        ),
        tfidf_breakdown_variant="pure",
    )


def render_tfidf_synonyms_tab():
    render_search_tab(
        method_key="tfidf_synonyms",
        score_name="TF-IDF + synonymes",
        search_callback=lambda moteur, query: moteur.chercher_tfidf_synonymes(query),
        score_description=(
            "Score : score TF-IDF calculé après enrichissement de la requête avec WordNet ; "
            "chaque mot peut être retrouvé via lui-même, son lemme ou un synonyme."
        ),
        requires_wordnet=True,
        tfidf_breakdown_variant="synonyms",
    )


def render_bm25_synonyms_tab():
    render_search_tab(
        method_key="bm25_synonyms",
        score_name="BM25 + synonymes",
        search_callback=lambda moteur, query, k1, b: moteur.chercher_bm25_synonymes(query, k1=k1, b=b),
        score_description=(
            "Score : score BM25 calculé après enrichissement de la requête avec WordNet ; "
            "chaque mot peut être retrouvé via lui-même, son lemme ou un synonyme."
        ),
        requires_wordnet=True,
        bm25_breakdown_variant="synonyms",
    )


def render_tfidf_lemmatization_tab():
    render_search_tab(
        method_key="tfidf_lemmatization",
        score_name="TF-IDF + lemmatisation",
        search_callback=lambda moteur, query: moteur.chercher_tfidf_lemmatise(query),
        score_description=(
            "Score : score TF-IDF calculé après normalisation des formes fléchies ; "
            "par exemple, des variantes comme robots et robot sont rapprochées."
        ),
        requires_wordnet=True,
        tfidf_breakdown_variant="lemmatization",
    )


def render_bm25_lemmatization_tab():
    render_search_tab(
        method_key="bm25_lemmatization",
        score_name="BM25 + lemmatisation",
        search_callback=lambda moteur, query, k1, b: moteur.chercher_bm25_lemmatise(query, k1=k1, b=b),
        score_description=(
            "Score : score BM25 calculé après normalisation des formes fléchies ; "
            "il tient aussi compte de la fréquence des termes et de la longueur du document."
        ),
        requires_wordnet=True,
        bm25_breakdown_variant="lemmatization",
    )


def render_bm25_tab():
    render_search_tab(
        method_key="bm25",
        score_name="BM25",
        search_callback=lambda moteur, query, k1, b: moteur.chercher_bm25(query, k1=k1, b=b),
        score_description=(
            "Score : score BM25 calculé pour la requête ; plus il est élevé, "
            "plus le document est pertinent en tenant compte de la fréquence des termes et de la longueur du document."
        ),
        bm25_breakdown_variant="pure",
    )


def render_splade_tab():
    render_search_tab(
        method_key="splade",
        score_name="SPLADE",
        search_callback=lambda moteur, query: search_splade(moteur, query),
        score_description=(
            "Score : score dot product sur les vecteurs creux SPLADE ; "
            "plus il est élevé, plus le document est compatible avec la requête."
        ),
    )


def render_hybrid_bm25_splade_tab():
    render_search_tab(
        method_key="hybrid_bm25_splade",
        score_name="BM25 synonymes + SPLADE",
        search_callback=lambda moteur, query, alpha: search_hybrid_bm25_splade(
            moteur,
            query,
            alpha=alpha,
        ),
        score_description=(
            "Score : fusion de rangs pondérée entre BM25 avec synonymes et SPLADE ; "
            "plus il est élevé, plus le document est bien placé par les deux méthodes."
        ),
        requires_wordnet=True,
        hybrid_alpha_enabled=True,
    )


def render_hybrid_tfidf_bm25_splade_tab():
    render_search_tab(
        method_key="hybrid_tfidf_bm25_splade",
        score_name="TF-IDF/BM25 synonymes + SPLADE",
        search_callback=lambda moteur, query: search_hybrid_tfidf_bm25_splade(
            moteur,
            query,
        ),
        score_description=(
            "Score : fusion de rangs entre TF-IDF avec synonymes, BM25 avec synonymes et SPLADE ; "
            "plus il est élevé, plus le document est bien placé par les trois méthodes."
        ),
        requires_wordnet=True,
    )


def render_hybrid_tfidf_splade_tab():
    render_search_tab(
        method_key="hybrid_tfidf_splade",
        score_name="TF-IDF synonymes + SPLADE",
        search_callback=lambda moteur, query: search_hybrid_tfidf_splade(
            moteur,
            query,
        ),
        score_description=(
            "Score : fusion de rangs entre TF-IDF avec synonymes et SPLADE ; "
            "plus il est élevé, plus le document est bien placé par les deux méthodes."
        ),
        requires_wordnet=True,
    )


def render_patentsberta_tab():
    render_search_tab(
        method_key="patentsberta",
        score_name="PatentSBERTa",
        search_callback=lambda moteur, query: search_patentsberta(moteur, query),
        score_description=(
            "Score : similarité cosinus sur des embeddings brevet PatentSBERTa ; "
            "plus il est élevé, plus le document est compatible avec la requête."
        ),
    )


def render_hybrid_bm25_patentsberta_tab():
    render_search_tab(
        method_key="hybrid_bm25_patentsberta",
        score_name="BM25 synonymes + PatentSBERTa",
        search_callback=lambda moteur, query, alpha: search_hybrid_bm25_patentsberta(
            moteur,
            query,
            alpha=alpha,
        ),
        score_description=(
            "Score : fusion de rangs pondérée entre BM25 avec synonymes et PatentSBERTa ; "
            "plus il est élevé, plus le document est bien placé par les deux méthodes."
        ),
        requires_wordnet=True,
        hybrid_alpha_enabled=True,
        hybrid_alpha_label="Poids PatentSBERTa",
        hybrid_alpha_target="PatentSBERTa",
    )


def render_hybrid_tfidf_patentsberta_tab():
    render_search_tab(
        method_key="hybrid_tfidf_patentsberta",
        score_name="TF-IDF synonymes + PatentSBERTa",
        search_callback=lambda moteur, query: search_hybrid_tfidf_patentsberta(
            moteur,
            query,
        ),
        score_description=(
            "Score : fusion de rangs entre TF-IDF avec synonymes et PatentSBERTa ; "
            "plus il est élevé, plus le document est bien placé par les deux méthodes."
        ),
        requires_wordnet=True,
    )


def render_hybrid_tfidf_bm25_patentsberta_tab():
    render_search_tab(
        method_key="hybrid_tfidf_bm25_patentsberta",
        score_name="TF-IDF/BM25 synonymes + PatentSBERTa",
        search_callback=lambda moteur, query: search_hybrid_tfidf_bm25_patentsberta(
            moteur,
            query,
        ),
        score_description=(
            "Score : fusion de rangs entre TF-IDF avec synonymes, BM25 avec synonymes et PatentSBERTa ; "
            "plus il est élevé, plus le document est bien placé par les trois méthodes."
        ),
        requires_wordnet=True,
    )


def render_comparison_tab():
    st.subheader("Comparaison des méthodes")
    st.write(
        "Ce tableau de bord teste si un brevet cible reste retrouvable quand la requête change "
        "de casse, d'ordre, de forme grammaticale ou de vocabulaire."
    )
    render_comparison_definitions()

    total_documents = count_searchable_patents()
    slider_max = max(1, total_documents)
    controls_col, target_col = st.columns([0.32, 0.68], gap="large")

    with controls_col:
        max_documents = st.slider(
            "Documents indexés pour la comparaison",
            min_value=1,
            max_value=slider_max,
            value=min(12000, slider_max),
            step=1,
            key="comparison_max_documents",
        )
        max_rank_display = st.slider(
            "Rang maximal considéré comme visible",
            min_value=1,
            max_value=500,
            value=10,
            step=1,
            key="comparison_max_rank",
        )

    with target_col:
        target_title = st.text_input(
            "Titre du brevet cible",
            value=TARGET_PATENT_TITLE,
            key="comparison_target_title",
        )
        raw_queries = st.text_area(
            "Variantes de requête à tester",
            value="\n".join(DEFAULT_COMPARISON_QUERIES),
            height=220,
            key="comparison_queries",
        )

    queries = [
        line.strip()
        for line in raw_queries.splitlines()
        if line.strip()
    ]
    patents = load_patents(max_documents)
    moteur = build_search_engine(patents)
    target_doc_id = find_patent_doc_id(patents, target_title)

    if target_doc_id is None:
        st.warning(
            "Le brevet cible n'est pas présent dans les documents indexés. "
            "Augmentez le nombre de documents indexés ou vérifiez le titre."
        )
        return

    target_patent = patents[target_doc_id]
    st.info(
        f"Brevet cible : Doc {target_doc_id} | {target_patent['patent_id']} | "
        f"{target_patent['title']}"
    )

    if moteur.wordnet_est_disponible():
        st.caption("WordNet est disponible : synonymes et lemmatisation sont actifs.")
    else:
        st.warning(
            "WordNet n'est pas disponible localement : les variantes synonymes/lemmatisation "
            "retomberont sur le comportement lexical simple."
        )

    if not st.button("Lancer la comparaison", key="run_comparison"):
        st.info("Lancez la comparaison pour générer les tableaux et graphiques.")
        return

    comparison_df = build_comparison_dataframe(moteur, target_doc_id, queries, target_title)
    if comparison_df.empty:
        st.warning("Aucune variante de requête à comparer.")
        return

    summary_df = build_comparison_summary(comparison_df, max_rank_display)
    top10_summary_df = build_comparison_summary(comparison_df, VISIBILITY_TOP_RANK)
    top10_summary_df = enrich_comparison_summary(top10_summary_df, comparison_df, VISIBILITY_TOP_RANK)
    matrix_df = build_presence_matrix(comparison_df, max_rank_display)
    family_matrix_df = build_family_matrix(comparison_df, VISIBILITY_TOP_RANK)
    quality_df = build_quality_metrics(comparison_df, VISIBILITY_TOP_RANK)
    problematic_queries = build_problematic_queries(comparison_df)

    render_comparison_dashboard(
        top10_summary_df,
        summary_df,
        comparison_df,
        family_matrix_df,
        quality_df,
        problematic_queries,
        matrix_df,
        max_rank_display,
    )


def render_search_tab(
    method_key,
    score_name,
    search_callback,
    score_description,
    requires_wordnet=False,
    tfidf_breakdown_variant=None,
    bm25_breakdown_variant=None,
    hybrid_alpha_enabled=False,
    hybrid_alpha_label="Poids SPLADE",
    hybrid_alpha_target="SPLADE",
):
    controls_col, results_col = st.columns([0.28, 0.72], gap="large")
    total_documents = count_searchable_patents()
    slider_max = max(1, total_documents)

    with controls_col:
        st.subheader("Paramètres")
        max_documents = st.slider(
            "Documents indexés",
            min_value=1,
            max_value=slider_max,
            value=min(12000, slider_max),
            step=1,
            key=f"{method_key}_max_documents",
        )
        max_results = st.slider(
            "Résultats affichés",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            key=f"{method_key}_max_results",
        )
        bm25_k1 = 1.5
        bm25_b = 0.75
        hybrid_alpha = 0.5
        if bm25_breakdown_variant is not None:
            bm25_k1 = st.slider(
                "BM25 k1",
                min_value=0.2,
                max_value=3.0,
                value=1.5,
                step=0.1,
                key=f"{method_key}_bm25_k1",
                help=(
                    "Contrôle la saturation de la fréquence d'un terme. "
                    "Plus k1 est élevé, plus les répétitions d'un mot peuvent augmenter le score."
                ),
            )
            bm25_b = st.slider(
                "BM25 b",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                key=f"{method_key}_bm25_b",
                help=(
                    "Contrôle la correction liée à la longueur du document. "
                    "0 désactive cette correction, 1 l'applique fortement."
                ),
            )
        if hybrid_alpha_enabled:
            hybrid_alpha = st.slider(
                hybrid_alpha_label,
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key=f"{method_key}_hybrid_alpha",
                help=(
                    f"0 utilise uniquement BM25, 1 utilise uniquement {hybrid_alpha_target}, "
                    "0.5 donne le même poids aux deux classements."
                ),
            )

    patents = load_patents(max_documents)
    moteur = build_search_engine(patents)

    with controls_col:
        st.metric("Documents", len(patents))
        st.metric("Documents disponibles", total_documents)
        st.metric("Termes indexés", len(moteur.index))
        if requires_wordnet and not moteur.wordnet_est_disponible():
            st.warning(
                "WordNet n'est pas disponible localement. Lancez "
                "`python -m nltk.downloader wordnet omw-1.4`, puis redémarrez Streamlit."
            )

    with results_col:
        query = st.text_input(
            "Recherche",
            value="Artificial Intelligence robot cleaner",
            placeholder="Exemple : medical image deep learning",
            key=f"{method_key}_query",
        )

        if not query.strip():
            render_empty_state(patents, score_name)
            return

        if hybrid_alpha_enabled:
            results = search_callback(moteur, query, hybrid_alpha)
        elif bm25_breakdown_variant is not None:
            results = search_callback(moteur, query, bm25_k1, bm25_b)
        else:
            results = search_callback(moteur, query)
        st.caption(f"{len(results)} résultat(s) trouvé(s)")

        if not results:
            st.warning("Aucun résultat trouvé pour cette requête.")
            return

        st.info(
            "Rang : position du résultat après tri par pertinence. "
            f"{score_description}"
        )
        if tfidf_breakdown_variant is not None:
            render_tfidf_score_breakdown(
                moteur,
                query,
                results[0],
                patents,
                tfidf_breakdown_variant,
            )
        if bm25_breakdown_variant is not None:
            render_bm25_score_breakdown(
                moteur,
                query,
                results[0],
                patents,
                bm25_breakdown_variant,
                bm25_k1,
                bm25_b,
            )
        render_results(results[:max_results], patents)


def render_empty_state(patents, score_name):
    st.info(f"Saisir une requête pour lancer une recherche {score_name}.")

    if not patents:
        return

    preview_rows = []
    for doc_id, patent in list(patents.items())[:5]:
        preview_rows.append(
            {
                "doc_id": doc_id,
                "patent_id": patent["patent_id"],
                "title": patent["title"],
                "publication_date": patent["publication_date"],
            }
        )

    st.dataframe(preview_rows, hide_index=True, use_container_width=True)


def render_results(results, patents):
    rows = []
    for rank, (doc_id, score) in enumerate(results, start=1):
        patent = patents[doc_id]
        rows.append(
            {
                "rang": rank,
                "score": round(score, 6),
                "patent_id": patent["patent_id"],
                "titre": patent["title"],
                "assignee": patent["assignee"],
                "publication": patent["publication_date"],
            }
        )

    st.dataframe(rows, hide_index=True, use_container_width=True)

    for rank, (doc_id, score) in enumerate(results, start=1):
        patent = patents[doc_id]
        title = patent["title"] or "Titre non disponible"
        with st.expander(f"{rank}. {title} - score {score:.4f}"):
            st.write(f"**Patent ID :** {patent['patent_id']}")
            st.write(f"**Assignee :** {patent['assignee'] or 'Non renseigné'}")
            st.write(f"**Publication :** {patent['publication_date'] or 'Non renseignée'}")
            if patent["result_link"]:
                st.link_button("Ouvrir le brevet", patent["result_link"])
            st.write(patent["abstract"])


def render_tfidf_score_breakdown(moteur, query, first_result, patents, variant):
    doc_id, score = first_result
    breakdown_df = build_tfidf_breakdown(moteur, query, doc_id, variant)

    if breakdown_df.empty:
        return

    patent_title = patents[doc_id]["title"] or f"Doc {doc_id}"
    st.subheader("Décomposition du score TF-IDF")
    st.caption(
        "Lecture du premier résultat uniquement, pour éviter de ralentir l'onglet. "
        "TF mesure la fréquence du mot dans ce document, IDF mesure la rareté du mot dans le corpus, "
        "et TF-IDF = TF × IDF. Le score total TF-IDF est la somme des contributions."
    )
    st.metric(
        "Score total TF-IDF du premier résultat",
        f"{score:.6f}",
        help="Somme des contributions TF × IDF des termes de la requête pour le document classé premier.",
    )
    st.caption(f"Document analysé : {patent_title}")

    st.dataframe(breakdown_df, hide_index=True, use_container_width=True)


def build_tfidf_breakdown(moteur, query, doc_id, variant):
    groupes_termes, index, tailles_documents = get_tfidf_breakdown_context(
        moteur,
        query,
        variant,
    )
    taille_doc = tailles_documents.get(doc_id, 0)
    if taille_doc == 0:
        return pd.DataFrame()

    rows = []
    for groupe in groupes_termes:
        best_row = build_best_tfidf_group_row(moteur, groupe, doc_id, index, taille_doc)
        if best_row is not None:
            rows.append(best_row)

    return pd.DataFrame(rows)


def get_tfidf_breakdown_context(moteur, query, variant):
    if variant == "synonyms":
        index, tailles_documents, _ = moteur._obtenir_index_lemmatise()
        groupes_termes = moteur._groupes_termes_synonymes(query)
    elif variant == "lemmatization":
        index, tailles_documents, _ = moteur._obtenir_index_lemmatise()
        groupes_termes = moteur._groupes_termes_lemmatises(query)
    else:
        index = moteur.index
        tailles_documents = moteur.tailles_documents
        groupes_termes = [
            [mot]
            for mot in moteur._nettoyer_texte(query)
        ]

    return groupes_termes, index, tailles_documents


def build_best_tfidf_group_row(moteur, groupe, doc_id, index, taille_doc):
    best_row = None

    for terme in groupe:
        frequence_brute = moteur._frequence_terme_document(terme, doc_id, index)
        idf = moteur._calculer_idf_depuis_index(terme, index)
        tf = frequence_brute / taille_doc if taille_doc else 0.0
        contribution = tf * idf
        row = {
            "terme affiché": terme,
            "termes équivalents testés": ", ".join(groupe),
            "fréquence brute": frequence_brute,
            "TF": round(tf, 6),
            "IDF": round(idf, 6),
            "Contribution TF-IDF": round(contribution, 6),
        }

        if best_row is None or row["Contribution TF-IDF"] > best_row["Contribution TF-IDF"]:
            best_row = row

    return best_row


def render_bm25_score_breakdown(moteur, query, first_result, patents, variant, k1, b):
    doc_id, score = first_result
    breakdown_df = build_bm25_breakdown(moteur, query, doc_id, variant, k1=k1, b=b)

    if breakdown_df.empty:
        return

    patent_title = patents[doc_id]["title"] or f"Doc {doc_id}"
    st.subheader("Décomposition du score BM25")
    st.caption(
        "Lecture du premier résultat uniquement, pour éviter de ralentir l'onglet. "
        "BM25 combine la rareté du terme, sa fréquence dans le document et une correction liée à la longueur du document."
    )
    st.info(
        f"Paramètres utilisés : `k1 = {k1:.1f}` et `b = {b:.2f}`. "
        "`k1` règle la saturation de la fréquence : répéter un mot aide, mais de moins en moins. "
        "`b` règle la correction par longueur du document. Par défaut, l'application propose `k1 = 1.5` "
        "et `b = 0.75`, des réglages classiques de BM25."
    )
    st.metric(
        "Score total BM25 du premier résultat",
        f"{score:.6f}",
        help="Somme des contributions BM25 des termes de la requête pour le document classé premier.",
    )
    st.caption(f"Document analysé : {patent_title}")
    st.dataframe(breakdown_df, hide_index=True, use_container_width=True)


def build_bm25_breakdown(moteur, query, doc_id, variant, k1=1.5, b=0.75):
    (
        groupes_termes,
        index,
        tailles_documents,
        longueur_moyenne_documents,
    ) = get_bm25_breakdown_context(moteur, query, variant)
    taille_doc = tailles_documents.get(doc_id, 0)
    if taille_doc == 0 or longueur_moyenne_documents == 0:
        return pd.DataFrame()

    rows = []
    for groupe in groupes_termes:
        best_row = build_best_bm25_group_row(
            moteur,
            groupe,
            doc_id,
            index,
            taille_doc,
            longueur_moyenne_documents,
            k1,
            b,
        )
        if best_row is not None:
            rows.append(best_row)

    return pd.DataFrame(rows)


def get_bm25_breakdown_context(moteur, query, variant):
    if variant == "synonyms":
        index, tailles_documents, longueur_moyenne_documents = moteur._obtenir_index_lemmatise()
        groupes_termes = moteur._groupes_termes_synonymes(query)
    elif variant == "lemmatization":
        index, tailles_documents, longueur_moyenne_documents = moteur._obtenir_index_lemmatise()
        groupes_termes = moteur._groupes_termes_lemmatises(query)
    else:
        index = moteur.index
        tailles_documents = moteur.tailles_documents
        longueur_moyenne_documents = moteur.longueur_moyenne_documents
        groupes_termes = [
            [mot]
            for mot in moteur._nettoyer_texte(query)
        ]

    return groupes_termes, index, tailles_documents, longueur_moyenne_documents


def build_best_bm25_group_row(
    moteur,
    groupe,
    doc_id,
    index,
    taille_doc,
    longueur_moyenne_documents,
    k1,
    b,
):
    best_row = None

    for terme in groupe:
        frequence_brute = moteur._frequence_terme_document(terme, doc_id, index)
        if frequence_brute == 0:
            contribution = 0.0
            normalisation_longueur = 0.0
            idf = moteur._calculer_idf_bm25_depuis_index(terme, index)
        else:
            idf = moteur._calculer_idf_bm25_depuis_index(terme, index)
            normalisation_longueur = 1 - b + b * (
                taille_doc / longueur_moyenne_documents
            )
            numerateur = frequence_brute * (k1 + 1)
            denominateur = frequence_brute + k1 * normalisation_longueur
            contribution = idf * (numerateur / denominateur)

        row = {
            "terme affiché": terme,
            "termes équivalents testés": ", ".join(groupe),
            "fréquence brute": frequence_brute,
            "IDF BM25": round(idf, 6),
            "longueur document": taille_doc,
            "longueur moyenne corpus": round(longueur_moyenne_documents, 2),
            "normalisation longueur": round(normalisation_longueur, 6),
            "k1": round(k1, 2),
            "b": round(b, 2),
            "Contribution BM25": round(contribution, 6),
        }

        if best_row is None or row["Contribution BM25"] > best_row["Contribution BM25"]:
            best_row = row

    return best_row


def render_comparison_definitions():
    with st.expander("Définitions des scores et métriques", expanded=False):
        st.write(
            "**Score** : valeur calculée par la méthode de ranking pour classer les documents. "
            "Un score plus haut signifie que le document est jugé plus pertinent pour la requête. "
            "Les scores servent surtout à comparer des documents au sein d'une même méthode ; "
            "un score TF-IDF et un score BM25 ne sont pas directement comparables entre eux."
        )
        st.write(
            "**TF-IDF** : combine la fréquence d'un terme dans un document avec sa rareté dans le corpus. "
            "Un mot fréquent dans le document mais rare ailleurs pèse davantage."
        )
        st.write(
            "**BM25** : proche de TF-IDF dans l'idée, mais avec une saturation de la fréquence des mots "
            "et une correction liée à la longueur du document."
        )
        st.write(
            "**Paramètres BM25 utilisés** : `k1 = 1.5` et `b = 0.75`. "
            "`k1` contrôle la saturation de la fréquence d'un mot : répéter un mot aide, mais de moins en moins. "
            "`b` contrôle la normalisation par longueur : avec `b = 0.75`, les documents très longs sont pénalisés "
            "partiellement pour éviter qu'ils gagnent seulement parce qu'ils contiennent plus de mots."
        )
        st.write(
            "**Taille des documents dans BM25** : elle est calculée automatiquement pendant l'indexation. "
            "Pour chaque document, on compte le nombre de mots nettoyés. La longueur moyenne est ensuite calculée "
            "sur les documents non vides. Les variantes avec lemmatisation utilisent la longueur de l'index lemmatisé."
        )
        st.write(
            "**Rang dashboard** : position de la méthode dans le classement synthétique du dashboard. "
            "On trie d'abord par taux de réussite, puis par présence dans le top 10, puis par rang moyen."
        )
        st.write(
            "**Requêtes trouvées** : nombre de variantes de requête pour lesquelles le brevet cible apparaît "
            "dans le rang maximal considéré."
        )
        st.write(
            "**Taux de réussite** : part des variantes où le brevet cible apparaît dans le rang maximal considéré. "
            "C'est le même calcul que la moyenne des `1` dans la matrice requête x méthode."
        )
        st.write(
            "**Top N** : nombre de variantes pour lesquelles le brevet cible apparaît dans les N premiers résultats. "
            "N correspond au réglage `Rang maximal considéré comme visible`."
        )
        st.write(
            "**Taux top N** : `nombre de requêtes où le brevet est dans le top N / nombre total de variantes * 100`."
        )
        st.write(
            "**Meilleur rang** : meilleure position obtenue par le brevet cible pour une méthode. "
            "Par exemple, `1` veut dire qu'au moins une variante place le brevet en premier."
        )
        st.write(
            "**Rang moyen** : position moyenne du brevet cible uniquement quand il apparaît dans le rang maximal considéré. "
            "Rang 1 signifie premier résultat ; plus le rang moyen est bas, meilleure est la méthode."
        )
        st.write(
            "**Matrice de robustesse** : `1` signifie que le brevet cible est retrouvé dans le top N "
            "pour une requête et une méthode ; `0` signifie qu'il ne l'est pas."
        )
        st.write(
            "**Recall@10** : part des requêtes où le brevet cible apparaît dans les 10 premiers résultats."
        )
        st.write(
            "**Precision@10** : part des 10 premières positions occupée par le brevet cible. "
            "Comme ce dashboard ne suit qu'un seul brevet pertinent, la valeur maximale est 10%."
        )
        st.write(
            "**MRR** : moyenne de `1 / rang`. Cette métrique récompense fortement les modèles qui placent "
            "le brevet cible très haut."
        )
        st.write(
            "**NDCG@10** : mesure de qualité du classement dans le top 10. Plus le brevet cible est haut, "
            "plus le score est proche de 100%."
        )


def render_comparison_dashboard(
    top10_summary_df,
    summary_df,
    comparison_df,
    family_matrix_df,
    quality_df,
    problematic_queries,
    matrix_df,
    max_rank_display,
):
    top10_rates = top10_summary_df.set_index("méthode")[f"taux top {VISIBILITY_TOP_RANK}"]
    dashboard_summary = enrich_comparison_summary(summary_df, comparison_df, max_rank_display)
    dashboard_summary[f"taux top {VISIBILITY_TOP_RANK}"] = dashboard_summary["méthode"].map(top10_rates)
    ranked_summary = rank_methods(dashboard_summary)
    best_method = ranked_summary.iloc[0]

    st.subheader("1. Synthèse rapide")
    render_decision_summary_cards(best_method)

    st.info(build_automatic_conclusion(best_method, family_matrix_df, problematic_queries, max_rank_display))

    st.subheader("2. Comparaison des modèles")
    st.caption(
        "Le tableau compare des indicateurs métier. Les scores internes TF-IDF et BM25 ne sont pas comparés entre eux."
    )
    top_methods_df = ranked_summary[
        [
            "méthode",
            "taux de réussite",
            f"taux top {VISIBILITY_TOP_RANK}",
            "rang moyen",
            "temps moyen (ms)",
            "lecture rapide",
        ]
    ].rename(
        columns={
            "méthode": "Méthode",
            "taux de réussite": "Taux de réussite",
            f"taux top {VISIBILITY_TOP_RANK}": "Taux top 10",
            "rang moyen": "Rang moyen",
            "temps moyen (ms)": "Temps moyen",
            "lecture rapide": "Lecture rapide",
        }
    )
    st.dataframe(top_methods_df, hide_index=True, use_container_width=True)

    render_business_charts(ranked_summary)

    st.subheader("3. Qualité du ranking")
    render_quality_analysis(quality_df)

    st.subheader("4. Analyse par famille de requêtes")
    render_family_analysis(family_matrix_df)

    st.subheader("5. Exemples par famille de requêtes")
    render_family_examples(comparison_df)


def render_comparison_matrix(matrix_df, max_rank_display):
    st.subheader("Matrice requête x méthode")
    st.caption(
        "1 signifie que le brevet cible apparaît dans le top choisi pour cette méthode ; "
        "0 signifie qu'il n'apparaît pas dans ce seuil."
    )
    st.dataframe(
        matrix_df.style.background_gradient(axis=None, cmap="Greens", vmin=0, vmax=1),
        use_container_width=True,
    )

    st.caption(f"Seuil actif : top {max_rank_display}.")


def render_comparison_details(comparison_df, summary_df):
    st.subheader("Synthèse brute")
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    st.subheader("Détail par requête et méthode")
    st.caption(
        "La colonne score sert à classer les résultats à l'intérieur d'une même méthode. "
        "Un score TF-IDF et un score BM25 ne doivent pas être comparés directement."
    )
    detail_df = comparison_df.copy()
    detail_df["score"] = detail_df["score"].round(6)
    detail_df["temps (ms)"] = detail_df["temps (ms)"].round(2)
    st.dataframe(detail_df, hide_index=True, use_container_width=True)


def get_search_methods():
    return [
        ("TF-IDF pur", lambda moteur, query: moteur.chercher_tfidf(query)),
        ("BM25 pur", lambda moteur, query: moteur.chercher_bm25(query)),
        ("TF-IDF + synonymes", lambda moteur, query: moteur.chercher_tfidf_synonymes(query)),
        ("BM25 + synonymes", lambda moteur, query: moteur.chercher_bm25_synonymes(query)),
        ("TF-IDF + lemmatisation", lambda moteur, query: moteur.chercher_tfidf_lemmatise(query)),
        ("BM25 + lemmatisation", lambda moteur, query: moteur.chercher_bm25_lemmatise(query)),
        ("SPLADE", search_splade),
        ("BM25 synonymes + SPLADE", lambda moteur, query: search_hybrid_bm25_splade(moteur, query)),
        ("TF-IDF synonymes + SPLADE", lambda moteur, query: search_hybrid_tfidf_splade(moteur, query)),
        ("TF-IDF/BM25 synonymes + SPLADE", lambda moteur, query: search_hybrid_tfidf_bm25_splade(moteur, query)),
        ("PatentSBERTa", search_patentsberta),
        ("BM25 synonymes + PatentSBERTa", lambda moteur, query: search_hybrid_bm25_patentsberta(moteur, query)),
        ("TF-IDF synonymes + PatentSBERTa", lambda moteur, query: search_hybrid_tfidf_patentsberta(moteur, query)),
        ("TF-IDF/BM25 synonymes + PatentSBERTa", lambda moteur, query: search_hybrid_tfidf_bm25_patentsberta(moteur, query)),
    ]


def find_patent_doc_id(patents, target_title):
    normalized_target = clean_value(target_title).lower()
    if not normalized_target:
        return None

    for doc_id, patent in patents.items():
        if clean_value(patent["title"]).lower() == normalized_target:
            return doc_id

    for doc_id, patent in patents.items():
        if normalized_target in clean_value(patent["title"]).lower():
            return doc_id

    return None


def build_comparison_dataframe(moteur, target_doc_id, queries, target_title):
    rows = []

    for query in queries:
        query_family = classify_query_family(query, target_title)
        for method_name, search_callback in get_search_methods():
            start_time = time.perf_counter()
            results = search_callback(moteur, query)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            rank, score = find_result_rank_and_score(results, target_doc_id)
            rows.append(
                {
                    "requête": query,
                    "famille": query_family,
                    "méthode": method_name,
                    "trouvé": rank is not None,
                    "rang": rank,
                    "score": score,
                    "temps (ms)": elapsed_ms,
                }
            )

    return pd.DataFrame(rows)


def build_comparison_summary(comparison_df, max_rank_display):
    summary_rows = []
    query_count = comparison_df["requête"].nunique()

    for method_name, method_df in comparison_df.groupby("méthode", sort=False):
        found_df = method_df[method_df["trouvé"]]
        top_rank_df = found_df[found_df["rang"] <= max_rank_display]
        best_rank = int(top_rank_df["rang"].min()) if not top_rank_df.empty else None
        average_rank = round(top_rank_df["rang"].mean(), 2) if not top_rank_df.empty else None
        found_count = int(found_df.shape[0])
        top_count = int(top_rank_df.shape[0])
        average_time = round(method_df["temps (ms)"].mean(), 2)
        summary_rows.append(
            {
                "méthode": method_name,
                "requêtes trouvées": top_count,
                "taux de réussite": round((top_count / query_count) * 100, 1),
                f"top {max_rank_display}": top_count,
                f"taux top {max_rank_display}": round((top_count / query_count) * 100, 1),
                "meilleur rang": best_rank,
                "rang moyen": average_rank,
                "temps moyen (ms)": average_time,
            }
        )

    return pd.DataFrame(summary_rows)


def enrich_comparison_summary(summary_df, comparison_df, max_rank_display):
    enriched_df = summary_df.copy()
    enriched_df["pénalité bruit"] = enriched_df["méthode"].apply(
        lambda method_name: compute_noise_penalty(comparison_df, method_name, max_rank_display)
    )
    enriched_df["point de vigilance"] = enriched_df.apply(build_method_warning, axis=1)
    return enriched_df


def compute_noise_penalty(comparison_df, method_name, max_rank_display):
    noisy_families = ["Requête large", "Requête bruitée ou discutable"]
    method_df = comparison_df[
        (comparison_df["méthode"] == method_name)
        & (comparison_df["famille"].isin(noisy_families))
    ]
    if method_df.empty:
        return 0.0

    noisy_success_rate = (
        method_df["trouvé"]
        & method_df["rang"].notna()
        & (method_df["rang"] <= max_rank_display)
    ).mean() * 100
    method_factor = 1.0 if "synonymes" in method_name else 0.6
    return round(min(10, noisy_success_rate * 0.1 * method_factor), 1)


def build_method_decision(row):
    method_name = row["méthode"]
    success_rate = row["taux de réussite"]
    top10_rate = row[f"taux top {VISIBILITY_TOP_RANK}"]
    rank = row["rang moyen"]
    penalty = row["pénalité bruit"]

    if row["rang dashboard"] == 1:
        return "Meilleur compromis"
    if "synonymes" in method_name and penalty >= 4:
        return "Très bon rappel mais attention au bruit"
    if success_rate >= 70 and top10_rate >= 60 and pd.notna(rank) and rank <= 20:
        return "Stable mais moins couvrant"
    if row["temps moyen (ms)"] <= 1.2 and success_rate < 60:
        return "Rapide mais moins efficace"
    if "synonymes" in method_name or "lemmatisation" in method_name:
        return "Coûteux pour un gain limité"
    return "Baseline simple"


def build_method_warning(row):
    if row["pénalité bruit"] >= 4:
        return "Surveillez les requêtes larges ou artificielles"
    if pd.isna(row["rang moyen"]):
        return "Ne retrouve pas le brevet cible"
    if row["rang moyen"] > 50:
        return "Brevet retrouvé mais souvent loin dans les résultats"
    if row["taux de réussite"] < 50:
        return "Couverture limitée sur les variantes"
    if row["temps moyen (ms)"] > 0 and row["temps moyen (ms)"] == row["temps moyen (ms)"]:
        return "A valider sur un corpus plus grand"
    return "Pas de vigilance majeure sur ce test"


def render_decision_summary_cards(best_method):
    columns = st.columns(5)
    columns[0].metric(
        "Modèle recommandé",
        best_method["méthode"],
        help="Modèle qui offre le meilleur compromis entre taux de réussite, présence dans le top 10, rang moyen et temps moyen.",
    )
    columns[1].metric(
        "Taux de réussite",
        f"{best_method['taux de réussite']:.1f}%",
        help="Nombre de formulations où le brevet cible apparaît dans le rang maximal considéré / nombre total de formulations testées × 100.",
    )
    columns[2].metric(
        "Taux top 10",
        f"{best_method[f'taux top {VISIBILITY_TOP_RANK}']:.1f}%",
        help="Nombre de formulations où le brevet cible apparaît dans les 10 premiers résultats / nombre total de formulations testées × 100.",
    )
    rank_value = "n/a" if pd.isna(best_method["rang moyen"]) else f"{best_method['rang moyen']:.2f}"
    columns[3].metric(
        "Rang moyen",
        rank_value,
        help="Moyenne des positions du brevet cible lorsqu'il est retrouvé. Plus le rang est faible, meilleur est le modèle.",
    )
    columns[4].metric(
        "Temps moyen",
        f"{best_method['temps moyen (ms)']:.2f} ms",
        help="Temps moyen nécessaire pour traiter une requête avec ce modèle. Plus le temps est faible, meilleur est le modèle.",
    )
    st.caption(f"Point de vigilance principal : {best_method['point de vigilance']}.")


def render_business_charts(ranked_summary):
    chart_df = ranked_summary.set_index("méthode")

    left_col, right_col = st.columns(2)
    with left_col:
        st.write("**Taux de réussite par méthode**")
        st.caption(
            "Règle de calcul : nombre de formulations où le brevet cible apparaît dans le rang maximal considéré / "
            "nombre total de formulations testées × 100."
        )
        st.bar_chart(chart_df["taux de réussite"], use_container_width=True)

        st.write("**Taux top 10 par méthode**")
        st.caption(
            "Règle de calcul : nombre de formulations où le brevet cible apparaît dans les 10 premiers résultats / "
            "nombre total de formulations testées × 100."
        )
        st.bar_chart(chart_df[f"taux top {VISIBILITY_TOP_RANK}"], use_container_width=True)

    with right_col:
        st.write("**Rang moyen par méthode - plus bas = meilleur**")
        st.caption(
            "Règle de calcul : moyenne des positions du brevet cible lorsqu'il est retrouvé. "
            "Plus la barre est basse, meilleur est le modèle."
        )
        average_rank_series = chart_df["rang moyen"].fillna(0).sort_values(ascending=True)
        st.bar_chart(average_rank_series, use_container_width=True)

        st.write("**Temps moyen par méthode - plus bas = meilleur**")
        st.caption(
            "Règle de calcul : moyenne des temps nécessaires pour traiter une formulation. "
            "Plus la barre est basse, plus le modèle est rapide."
        )
        st.bar_chart(chart_df["temps moyen (ms)"].sort_values(ascending=True), use_container_width=True)


def render_quality_analysis(quality_df):
    st.caption(
        "Cette section mesure la qualité du classement. Comme le test suit un seul brevet cible pertinent "
        "par requête, les métriques les plus utiles sont Recall@10, MRR@10 et NDCG@10."
    )
    with st.expander("Comment lire ces métriques de qualité", expanded=False):
        st.write(
            "**Recall global** : part des requêtes où le brevet cible est retrouvé quelque part dans les résultats."
        )
        st.write(
            "**Recall@10** : part des requêtes où le brevet cible apparaît dans les 10 premiers résultats. "
            "C'est l'indicateur le plus proche de la visibilité utilisateur."
        )
        st.write(
            "**Precision@10** : part des 10 premières places occupée par le brevet cible. "
            "Comme on ne suit qu'un seul brevet pertinent, cette métrique est surtout indicative et plafonne à 10%."
        )
        st.write(
            "**MRR@10** : moyenne de `1 / rang` quand le brevet est dans le top 10. "
            "Un modèle qui place souvent le brevet en rang 1 obtient un meilleur MRR."
        )
        st.write(
            "**NDCG@10** : qualité du classement dans le top 10. "
            "Plus le brevet cible est haut dans les résultats, plus la valeur se rapproche de 100%."
        )

    st.dataframe(quality_df, hide_index=True, use_container_width=True)


def build_quality_metrics(comparison_df, top_k):
    rows = []

    for method_name, method_df in comparison_df.groupby("méthode", sort=False):
        query_count = method_df["requête"].nunique()
        if query_count == 0:
            continue

        found_anywhere = method_df["trouvé"].fillna(False)
        ranks = method_df["rang"]
        in_top_k = found_anywhere & ranks.notna() & (ranks <= top_k)

        recall_global = found_anywhere.mean() * 100
        recall_at_k = in_top_k.mean() * 100
        precision_at_k = (in_top_k.sum() / (query_count * top_k)) * 100
        reciprocal_ranks = ranks.where(in_top_k, other=0).apply(
            lambda rank: 1 / rank if rank else 0
        )
        ndcg_values = ranks.where(in_top_k, other=0).apply(
            lambda rank: 1 / math.log2(rank + 1) if rank else 0
        )

        rows.append(
            {
                "Méthode": method_name,
                "Recall global": round(recall_global, 1),
                f"Recall@{top_k}": round(recall_at_k, 1),
                f"Precision@{top_k}": round(precision_at_k, 1),
                f"MRR@{top_k}": round(reciprocal_ranks.mean(), 3),
                f"NDCG@{top_k}": round(ndcg_values.mean() * 100, 1),
                "Lecture qualité": build_quality_reading(
                    recall_at_k,
                    reciprocal_ranks.mean(),
                    ndcg_values.mean() * 100,
                ),
            }
        )

    quality_df = pd.DataFrame(rows)
    if quality_df.empty:
        return quality_df

    return quality_df.sort_values(
        [f"NDCG@{top_k}", f"MRR@{top_k}", f"Recall@{top_k}"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_quality_reading(recall_at_k, mrr_at_k, ndcg_at_k):
    if recall_at_k >= 80 and mrr_at_k >= 0.5:
        return "Très bonne qualité : le brevet est souvent visible et bien classé"
    if recall_at_k >= 70:
        return "Bonne visibilité, mais le rang peut encore varier"
    if ndcg_at_k >= 40:
        return "Qualité correcte, avec des résultats parfois trop bas"
    if recall_at_k > 0:
        return "Qualité limitée : le brevet est retrouvé mais rarement bien placé"
    return "Qualité faible : le brevet cible n'est pas visible dans le top 10"


def build_family_matrix(comparison_df, max_rank_display):
    family_df = comparison_df.copy()
    family_df["réussite top n"] = (
        family_df["trouvé"]
        & family_df["rang"].notna()
        & (family_df["rang"] <= max_rank_display)
    )
    matrix = family_df.pivot_table(
        index="famille",
        columns="méthode",
        values="réussite top n",
        aggfunc=lambda values: round(values.mean() * 100, 1),
        fill_value=0,
        sort=False,
    )
    return matrix


def render_family_analysis(family_matrix_df):
    st.caption(
        "Cette section explique pourquoi un modèle fonctionne ou échoue selon le type de formulation utilisé. "
        "Les pourcentages indiquent la part des requêtes de la famille où le brevet cible apparaît dans le top 10. "
        "Les familles sont classées avec des règles heuristiques définies pour cette analyse pédagogique."
    )
    render_family_definition_help()

    st.write("**Taux top 10 par famille et par méthode**")
    st.dataframe(
        family_matrix_df.style.format("{:.1f}%").background_gradient(axis=None, cmap="Greens", vmin=0, vmax=100),
        use_container_width=True,
    )


def render_family_definition_help():
    with st.expander("Comment les familles de requêtes sont définies", expanded=False):
        st.write(
            "Avant de classer une requête, le texte est normalisé : passage en minuscules, suppression de la ponctuation, "
            "puis découpage en mots. La requête est ensuite comparée au titre du brevet cible normalisé."
        )
        st.warning(
            "Important : cette classification n'est pas une compréhension sémantique générale. "
            "Elle repose sur des règles simples et quelques listes de mots choisies manuellement pour ce brevet cible "
            "et pour ce jeu de requêtes."
        )
        st.write(
            "**Requête exacte** : la requête reprend le titre complet ou une partie très proche du titre, "
            "avec seulement des différences de casse ou de ponctuation. Règle heuristique basée sur la similarité "
            "avec le titre cible."
        )
        st.write(
            "**Ordre des mots** : la requête contient les mêmes mots importants que le titre, ou un sous-ensemble de ces mots, "
            "mais dans un ordre différent. Règle heuristique basée sur la comparaison des ensembles de mots."
        )
        st.write(
            "**Pluriel / forme grammaticale** : la requête contient des variantes comme singulier/pluriel ou formes en `-ing`, "
            "par exemple `robot`/`robots` ou `cleaner`/`cleaning`. Règle heuristique basée sur des suffixes simples."
        )
        st.write(
            "**Reformulation sémantique** : la requête utilise d'autres mots pour exprimer une idée proche, "
            "par exemple `smart`, `autonomous`, `vacuum`, `apparatus` ou `automated`. Cette famille est en partie "
            "manuelle : ces mots ont été choisis pour cette analyse spécifique."
        )
        st.write(
            "**Requête large** : la requête contient seulement un ou deux mots très génériques, "
            "comme `system`, `robot` ou `cleaning`. Règle heuristique basée sur une liste courte de mots génériques."
        )
        st.write(
            "**Requête bruitée ou discutable** : la requête contient des termes volontairement éloignés ou peu naturels, "
            "par exemple `automaton` ou `cleanser`. Cette famille est manuelle et sert à repérer les cas qui peuvent "
            "introduire du bruit."
        )
        st.write(
            "**Recherche brevet / prior art** : la requête ressemble à une recherche d'ingénieur brevet ou de juriste, "
            "avec des termes comme `patent`, `prior art`, `control`, `sensor`, `navigation` ou `obstacle detection`."
        )
        st.write(
            "**Recherche utilisateur naturelle** : la requête décrit le besoin avec des mots plus courants, "
            "par exemple une machine qui nettoie seule, apprend le logement ou décide où nettoyer."
        )


def render_family_examples(comparison_df):
    examples_df = build_family_examples(comparison_df)
    if examples_df.empty:
        st.info("Aucun exemple de famille disponible pour les requêtes testées.")
        return

    st.caption(
        "Ce tableau donne un exemple concret pour chaque famille détectée dans les variantes testées."
    )
    st.dataframe(examples_df, hide_index=True, use_container_width=True)


def build_family_examples(comparison_df):
    query_family_df = comparison_df[["requête", "famille"]].drop_duplicates()
    if query_family_df.empty:
        return pd.DataFrame()

    family_rows = []
    family_order = get_query_family_order()
    family_descriptions = get_query_family_descriptions()

    for family in family_order:
        family_queries = query_family_df[query_family_df["famille"] == family]
        if family_queries.empty:
            continue

        family_rows.append(
            {
                "Famille": family,
                "Exemple de requête": family_queries.iloc[0]["requête"],
                "Nombre de variantes": int(family_queries.shape[0]),
                "Lecture": family_descriptions.get(family, ""),
            }
        )

    extra_families = [
        family
        for family in query_family_df["famille"].dropna().unique()
        if family not in family_order
    ]
    for family in extra_families:
        family_queries = query_family_df[query_family_df["famille"] == family]
        family_rows.append(
            {
                "Famille": family,
                "Exemple de requête": family_queries.iloc[0]["requête"],
                "Nombre de variantes": int(family_queries.shape[0]),
                "Lecture": family_descriptions.get(family, ""),
            }
        )

    return pd.DataFrame(family_rows)


def get_query_family_order():
    return [
        "Requête exacte",
        "Ordre des mots",
        "Pluriel / forme grammaticale",
        "Reformulation sémantique",
        "Requête large",
        "Requête bruitée ou discutable",
        "Recherche brevet / prior art",
        "Recherche utilisateur naturelle",
    ]


def get_query_family_descriptions():
    return {
        "Requête exacte": "Reprend le titre cible ou une variante très proche.",
        "Ordre des mots": "Utilise les mots importants du titre, mais dans un autre ordre ou en sous-ensemble.",
        "Pluriel / forme grammaticale": "Change surtout les formes des mots, par exemple singulier, pluriel ou forme en -ing.",
        "Reformulation sémantique": "Exprime l'idée avec d'autres mots proches du besoin initial.",
        "Requête large": "Reste très générale, avec un ou deux mots peu spécifiques.",
        "Requête bruitée ou discutable": "Ajoute des termes éloignés, artificiels ou peu naturels pour le cas étudié.",
        "Recherche brevet / prior art": "Ressemble à une recherche technique ou juridique autour du brevet.",
        "Recherche utilisateur naturelle": "Formule le besoin comme un utilisateur non spécialiste.",
    }


def build_problematic_queries(comparison_df):
    rows = []
    synonym_methods = comparison_df["méthode"].str.contains("synonymes", regex=False)

    for query, query_df in comparison_df.groupby("requête", sort=False):
        found_methods = query_df[query_df["trouvé"]]["méthode"].tolist()
        synonym_found = query_df[synonym_methods.loc[query_df.index] & query_df["trouvé"]]
        non_synonym_found = query_df[~synonym_methods.loc[query_df.index] & query_df["trouvé"]]
        family = query_df["famille"].iloc[0]

        if not found_methods:
            rows.append(
                {
                    "type": "Échec pour tous les modèles",
                    "requête": query,
                    "famille": family,
                    "lecture": "Aucune méthode ne retrouve le brevet cible.",
                }
            )
        elif not synonym_found.empty and non_synonym_found.empty:
            rows.append(
                {
                    "type": "Retrouvée uniquement par synonymes",
                    "requête": query,
                    "famille": family,
                    "lecture": "Les synonymes apportent un gain réel sur cette formulation.",
                }
            )
        elif family in ["Requête large", "Requête bruitée ou discutable"]:
            rows.append(
                {
                    "type": "Potentiellement bruitée",
                    "requête": query,
                    "famille": family,
                    "lecture": "Cette formulation peut favoriser des résultats peu spécifiques.",
                }
            )

    return pd.DataFrame(rows)


def render_problematic_queries(problematic_queries):
    st.write("**Requêtes à surveiller**")
    if problematic_queries.empty:
        st.success("Aucune requête problématique détectée sur ce jeu de test.")
        return

    st.dataframe(problematic_queries, hide_index=True, use_container_width=True)


def build_automatic_conclusion(best_method, family_matrix_df, problematic_queries, max_rank_display):
    method_name = best_method["méthode"]
    method_family_scores = family_matrix_df[method_name].sort_values(ascending=False)
    robust_families = method_family_scores[method_family_scores >= 80].index.tolist()[:3]
    weak_families = method_family_scores[method_family_scores < 50].index.tolist()[:3]

    advantage = build_main_advantage(robust_families)
    warning = build_main_limit(problematic_queries, weak_families, best_method)
    rank_text = "n/a" if pd.isna(best_method["rang moyen"]) else f"{best_method['rang moyen']:.2f}"

    return (
        f"Sur ce jeu de test, le meilleur compromis est {method_name}. "
        f"Il obtient {best_method['taux de réussite']:.1f}% de réussite et "
        f"{best_method[f'taux top {VISIBILITY_TOP_RANK}']:.1f}% d'apparition dans le top 10 "
        f"avec un rang moyen de {rank_text}. "
        f"Son principal avantage est {advantage}. "
        f"Son principal point de vigilance est {warning}."
    )


def build_main_advantage(robust_families):
    if robust_families:
        return f"sa robustesse sur {', '.join(robust_families[:2])}"
    return "sa capacité à retrouver le brevet cible sur plusieurs formulations"


def build_main_limit(problematic_queries, weak_families, best_method):
    if not problematic_queries.empty and "Échec pour tous les modèles" in problematic_queries["type"].values:
        return "certaines requêtes restent introuvables pour tous les modèles"
    if best_method["pénalité bruit"] >= 4:
        return "les requêtes larges ou discutables peuvent créer du bruit"
    if weak_families:
        return f"des résultats plus faibles sur {', '.join(weak_families[:2])}"
    return "ces résultats doivent être confirmés sur davantage de brevets cibles"


def classify_query_family(query, target_title):
    normalized_query = normalize_for_family(query)
    normalized_target = normalize_for_family(target_title)
    query_tokens = normalized_query.split()
    target_tokens = normalized_target.split()

    if normalized_query == normalized_target:
        return "Requête exacte"
    if query.strip().lower() == target_title.strip().lower() and query.strip() != target_title.strip():
        return "Requête exacte"
    if normalized_query in normalized_target:
        if has_case_or_punctuation_variation(query, normalized_query):
            return "Requête exacte"
    if any(token in {"golem", "cleanser", "automaton"} for token in query_tokens):
        return "Requête bruitée ou discutable"
    if is_broad_query(query_tokens, normalized_query):
        return "Requête large"
    if set(query_tokens) == set(target_tokens) and query_tokens != target_tokens:
        return "Ordre des mots"
    if set(query_tokens).issubset(set(target_tokens)) and query_tokens != target_tokens:
        return "Ordre des mots"
    if is_patent_search_query(query_tokens):
        return "Recherche brevet / prior art"
    if is_natural_user_query(query_tokens):
        return "Recherche utilisateur naturelle"
    if has_plural_or_grammar_variation(query_tokens, target_tokens):
        return "Pluriel / forme grammaticale"
    if any(token in {"intelligent", "smart", "vacuum", "automated", "autonomous", "apparatus"} for token in query_tokens):
        return "Reformulation sémantique"
    if len(set(query_tokens) & set(target_tokens)) < max(1, len(query_tokens) // 3):
        return "Reformulation sémantique"
    if any(token.endswith("ing") or token.endswith("s") for token in query_tokens):
        return "Pluriel / forme grammaticale"
    return "Reformulation sémantique"


def normalize_for_family(value):
    return " ".join(re.findall(r"[a-z0-9]+", clean_value(value).lower()))


def remove_punctuation(value):
    return re.sub(r"[^A-Za-z0-9\s]", " ", value)


def has_case_or_punctuation_variation(query, normalized_query):
    compact_original = re.sub(r"\s+", " ", remove_punctuation(query).lower()).strip()
    return compact_original == normalized_query and query != compact_original


def has_plural_or_grammar_variation(query_tokens, target_tokens):
    target_roots = {token.rstrip("s") for token in target_tokens}
    query_roots = {token.rstrip("s") for token in query_tokens}
    return bool(query_roots & target_roots) and query_roots != set(query_tokens)


def is_broad_query(query_tokens, normalized_query):
    broad_terms = {"system", "robot", "cleaning", "artificial", "intelligence"}
    if len(query_tokens) <= 1:
        return True
    if len(query_tokens) <= 2 and set(query_tokens).issubset(broad_terms):
        return True
    return normalized_query in broad_terms


def is_patent_search_query(query_tokens):
    patent_markers = {
        "patent", "prior", "art", "invention", "control", "sensor", "sensors",
        "navigation", "mapping", "obstacle", "detection", "path", "route",
        "planning", "search", "system",
    }
    return len(set(query_tokens) & patent_markers) >= 2


def is_natural_user_query(query_tokens):
    natural_markers = {
        "machine", "floors", "itself", "room", "home", "learns", "decides",
        "where", "clean", "uses", "automatically",
    }
    return len(set(query_tokens) & natural_markers) >= 2


def build_presence_matrix(comparison_df, max_rank_display):
    matrix_source = comparison_df.copy()
    matrix_source["présent"] = (
        matrix_source["trouvé"]
        & matrix_source["rang"].notna()
        & (matrix_source["rang"] <= max_rank_display)
    ).astype(int)
    matrix = matrix_source.pivot(
        index="requête",
        columns="méthode",
        values="présent",
    )
    return matrix.fillna(0).astype(int)


def rank_methods(summary_df):
    ranked_summary = summary_df.copy()
    ranked_summary["_rang_moyen_tri"] = ranked_summary["rang moyen"].fillna(float("inf"))
    ranked_summary = ranked_summary.sort_values(
        ["taux de réussite", f"taux top {VISIBILITY_TOP_RANK}", "_rang_moyen_tri", "temps moyen (ms)"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    ranked_summary["rang dashboard"] = ranked_summary.index + 1
    ranked_summary["lecture rapide"] = ranked_summary.apply(build_method_decision, axis=1)
    return ranked_summary.drop(columns=["_rang_moyen_tri"])


def extract_top_column(summary_df):
    for column in summary_df.columns:
        if column.startswith("taux top "):
            return column.replace("taux top ", "")
    return "50"


def find_result_rank_and_score(results, target_doc_id):
    for rank, (doc_id, score) in enumerate(results, start=1):
        if doc_id == target_doc_id:
            return rank, score
    return None, 0.0


@st.cache_resource(show_spinner="Chargement du modèle SPLADE...")
def get_splade_encoder():
    import torch
    from pinecone_text.sparse import SpladeEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SpladeEncoder(device=device)


@st.cache_resource(show_spinner="Chargement du modèle PatentSBERTa...")
def get_patentsberta_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(PATENTSBERTA_MODEL_NAME)


def search_splade(moteur, query):
    """
    Recherche SPLADE locale.

    Les documents et la requête sont encodés en vecteurs creux, puis comparés
    par produit scalaire. Plus le score est élevé, plus le document est proche
    de la requête selon SPLADE.
    """
    if not query.strip():
        return []

    if not moteur.corpus_vectors:
        moteur.indexer_corpus_splade(moteur.documents, encoder=get_splade_encoder())

    try:
        query_vector = get_splade_encoder().encode_queries(query)
    except Exception:
        return []

    query_sparse = {
        int(index): float(value)
        for index, value in zip(
            query_vector.get("indices", []),
            query_vector.get("values", []),
        )
        if value != 0
    }

    scores = {}
    for doc_id, doc_vector in moteur.corpus_vectors.items():
        if doc_id not in moteur.documents:
            continue

        score = 0.0
        for index, weight in query_sparse.items():
            score += weight * doc_vector.get(index, 0.0)
        if score > 0:
            scores[doc_id] = score

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def search_hybrid_bm25_splade(moteur, query, alpha=0.5, rrf_k=60):
    if not query.strip():
        return []

    alpha = min(1.0, max(0.0, alpha))
    bm25_results = moteur.chercher_bm25_synonymes(query)
    splade_results = search_splade(moteur, query)
    hybrid_scores = {}

    add_rrf_scores(hybrid_scores, bm25_results, weight=1 - alpha, rrf_k=rrf_k)
    add_rrf_scores(hybrid_scores, splade_results, weight=alpha, rrf_k=rrf_k)

    return sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)


def search_hybrid_tfidf_bm25_splade(moteur, query, rrf_k=60):
    if not query.strip():
        return []

    tfidf_results = moteur.chercher_tfidf_synonymes(query)
    bm25_results = moteur.chercher_bm25_synonymes(query)
    splade_results = search_splade(moteur, query)
    hybrid_scores = {}
    weight = 1 / 3

    add_rrf_scores(hybrid_scores, tfidf_results, weight=weight, rrf_k=rrf_k)
    add_rrf_scores(hybrid_scores, bm25_results, weight=weight, rrf_k=rrf_k)
    add_rrf_scores(hybrid_scores, splade_results, weight=weight, rrf_k=rrf_k)

    return sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)


def search_hybrid_tfidf_splade(moteur, query, rrf_k=60):
    if not query.strip():
        return []

    tfidf_results = moteur.chercher_tfidf_synonymes(query)
    splade_results = search_splade(moteur, query)
    hybrid_scores = {}

    add_rrf_scores(hybrid_scores, tfidf_results, weight=0.5, rrf_k=rrf_k)
    add_rrf_scores(hybrid_scores, splade_results, weight=0.5, rrf_k=rrf_k)

    return sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)


def search_patentsberta(moteur, query):
    if not query.strip():
        return []

    try:
        vectors = get_patentsberta_vectors(moteur)
    except Exception:
        return []

    if not vectors:
        return []

    try:
        query_vector = encode_patentsberta_texts([query])[0]
    except Exception:
        return []

    return score_dense_vectors(query_vector, vectors, moteur.documents)


def search_hybrid_bm25_patentsberta(moteur, query, alpha=0.5, rrf_k=60):
    if not query.strip():
        return []

    alpha = min(1.0, max(0.0, alpha))
    bm25_results = moteur.chercher_bm25_synonymes(query)
    patentsberta_results = search_patentsberta(moteur, query)
    hybrid_scores = {}

    add_rrf_scores(hybrid_scores, bm25_results, weight=1 - alpha, rrf_k=rrf_k)
    add_rrf_scores(hybrid_scores, patentsberta_results, weight=alpha, rrf_k=rrf_k)

    return sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)


def search_hybrid_tfidf_patentsberta(moteur, query, rrf_k=60):
    if not query.strip():
        return []

    tfidf_results = moteur.chercher_tfidf_synonymes(query)
    patentsberta_results = search_patentsberta(moteur, query)
    hybrid_scores = {}

    add_rrf_scores(hybrid_scores, tfidf_results, weight=0.5, rrf_k=rrf_k)
    add_rrf_scores(hybrid_scores, patentsberta_results, weight=0.5, rrf_k=rrf_k)

    return sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)


def search_hybrid_tfidf_bm25_patentsberta(moteur, query, rrf_k=60):
    if not query.strip():
        return []

    tfidf_results = moteur.chercher_tfidf_synonymes(query)
    bm25_results = moteur.chercher_bm25_synonymes(query)
    patentsberta_results = search_patentsberta(moteur, query)
    hybrid_scores = {}
    weight = 1 / 3

    add_rrf_scores(hybrid_scores, tfidf_results, weight=weight, rrf_k=rrf_k)
    add_rrf_scores(hybrid_scores, bm25_results, weight=weight, rrf_k=rrf_k)
    add_rrf_scores(hybrid_scores, patentsberta_results, weight=weight, rrf_k=rrf_k)

    return sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)


def add_rrf_scores(hybrid_scores, results, weight, rrf_k):
    if weight == 0:
        return

    for rank, (doc_id, _) in enumerate(results, start=1):
        hybrid_scores[doc_id] = hybrid_scores.get(doc_id, 0.0) + weight / (rrf_k + rank)


def get_patentsberta_vectors(moteur):
    cached_vectors = getattr(moteur, "patentsberta_vectors", None)
    if dense_index_matches_corpus(cached_vectors, moteur.documents):
        return cached_vectors

    cached_vectors = load_vector_index(PATENTSBERTA_INDEX_PATH)
    if dense_index_matches_corpus(cached_vectors, moteur.documents):
        moteur.patentsberta_vectors = trim_vector_index(cached_vectors, moteur.documents)
        return moteur.patentsberta_vectors

    print(f"Index PatentSBERTa absent ou incomplet, création du cache dans {PATENTSBERTA_INDEX_PATH}...")
    moteur.patentsberta_vectors = index_dense_corpus(moteur.documents, label="PatentSBERTa", batch_size=32)
    save_vector_index(PATENTSBERTA_INDEX_PATH, moteur.patentsberta_vectors)
    print(f"Index PatentSBERTa sauvegardé dans {PATENTSBERTA_INDEX_PATH}.")
    return moteur.patentsberta_vectors


def index_dense_corpus(corpus, label, batch_size=32):
    vectors = {}
    ordered_doc_ids = sorted(corpus.keys())
    total_docs = len(ordered_doc_ids)
    start_time = time.perf_counter()

    for start in range(0, total_docs, batch_size):
        batch_doc_ids = ordered_doc_ids[start:start + batch_size]
        batch_texts = [corpus[doc_id] for doc_id in batch_doc_ids]
        batch_vectors = encode_patentsberta_texts(batch_texts)

        for doc_id, vector in zip(batch_doc_ids, batch_vectors):
            vectors[doc_id] = vector

        processed_docs = min(start + batch_size, total_docs)
        elapsed = time.perf_counter() - start_time
        progress = processed_docs / total_docs if total_docs else 1.0
        if processed_docs % 100 == 0 or processed_docs == total_docs:
            speed = processed_docs / elapsed if elapsed > 0 else 0.0
            print(
                f"Indexation {label} en cours... {processed_docs}/{total_docs} "
                f"documents traités ({progress * 100:.1f}%) - "
                f"{speed:.2f} doc/s - {elapsed:.1f}s"
            )

    return vectors


def encode_patentsberta_texts(texts):
    embeddings = get_patentsberta_model().encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [
        embedding.astype(float).tolist() if hasattr(embedding, "astype") else [float(value) for value in embedding]
        for embedding in embeddings
    ]


def score_dense_vectors(query_vector, corpus_vectors, documents):
    scores = {}
    for doc_id, doc_vector in corpus_vectors.items():
        if doc_id not in documents:
            continue

        score = sum(query_value * doc_value for query_value, doc_value in zip(query_vector, doc_vector))
        scores[doc_id] = score

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def load_vector_index(path):
    try:
        with path.open("rb") as index_file:
            return pickle.load(index_file)
    except (FileNotFoundError, EOFError, pickle.PickleError):
        return {}


def save_vector_index(path, vectors):
    with path.open("wb") as index_file:
        pickle.dump(vectors, index_file)


def dense_index_matches_corpus(vectors, corpus):
    if not vectors:
        return False

    return set(corpus.keys()).issubset(set(vectors.keys()))


def trim_vector_index(vectors, corpus):
    return {
        doc_id: vectors[doc_id]
        for doc_id in sorted(corpus.keys())
    }


@st.cache_data(show_spinner="Comptage des documents...")
def count_searchable_patents():
    total = 0
    ensure_dataset_available()
    increase_csv_field_limit()

    with DATA_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            title = clean_value(row.get("Title"))
            abstract = clean_value(row.get("Abstract"))
            if title or abstract:
                total += 1

    return total


@st.cache_data(show_spinner="Chargement du corpus...")
def load_patents(max_documents):
    patents = {}
    ensure_dataset_available()
    increase_csv_field_limit()

    with DATA_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            title = clean_value(row.get("Title"))
            abstract = clean_value(row.get("Abstract"))
            if not title and not abstract:
                continue

            doc_id = len(patents)
            patents[doc_id] = {
                "patent_id": clean_value(row.get("Patent ID")),
                "title": title,
                "assignee": clean_value(row.get("Assignee")),
                "publication_date": clean_value(row.get("Publication Date")),
                "abstract": abstract,
                "result_link": clean_value(row.get("Result Link")),
                "search_text": build_search_text(row),
            }

            if len(patents) >= max_documents:
                break

    return patents


@st.cache_resource(show_spinner="Indexation du corpus...")
def build_search_engine(patents):
    corpus = {
        doc_id: patent["search_text"]
        for doc_id, patent in patents.items()
        if patent["search_text"].strip()
    }
    moteur = MoteurRechercheTextuel()
    moteur.indexer_corpus(corpus)

    index_path = Path("index_splade.pkl")
    if index_path.exists():
        print(f"Chargement de l'index SPLADE depuis {index_path}...")
        moteur.charger_index(index_path)

    if not splade_index_matches_corpus(moteur.corpus_vectors, corpus):
        print(f"Index SPLADE absent ou incomplet, création du cache dans {index_path}...")
        moteur.indexer_corpus_splade(corpus, encoder=get_splade_encoder())
        moteur.sauvegarder_index(index_path)
        print(f"Index SPLADE sauvegardé dans {index_path}.")
    else:
        moteur.corpus_vectors = {
            doc_id: moteur.corpus_vectors[doc_id]
            for doc_id in sorted(corpus.keys())
        }

    return moteur


def splade_index_matches_corpus(corpus_vectors, corpus):
    if not corpus_vectors:
        return False

    corpus_doc_ids = set(corpus.keys())
    indexed_doc_ids = set(corpus_vectors.keys())
    return corpus_doc_ids.issubset(indexed_doc_ids)


def build_search_text(row):
    return " ".join(clean_value(row.get(column)) for column in TEXT_COLUMNS).strip()


def clean_value(value):
    return str(value or "").strip()


def ensure_dataset_available():
    if is_valid_dataset_csv(DATA_PATH):
        return str(DATA_PATH)

    try:
        import kagglehub
    except ImportError as import_error:
        raise ImportError(
            "La dépendance 'kagglehub' est requise pour télécharger le dataset automatiquement. "
            "Installez-la avec : pip install -r requirements.txt"
        ) from import_error

    downloaded_path = download_kaggle_dataset(kagglehub)
    csv_files = sorted(
        downloaded_path.rglob("*.csv"),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if not csv_files:
        raise FileNotFoundError(
            f"Aucun fichier CSV trouvé dans le dataset téléchargé : {downloaded_path}"
        )

    source_csv = find_dataset_csv(csv_files)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = DATA_PATH.with_suffix(".csv.tmp")
    shutil.copy(source_csv, temporary_path)
    temporary_path.replace(DATA_PATH)

    if not is_valid_dataset_csv(DATA_PATH):
        raise ValueError(
            f"Le fichier téléchargé ne contient pas les colonnes attendues : {DATA_PATH}"
        )

    return str(DATA_PATH)


def download_kaggle_dataset(kagglehub):
    try:
        return Path(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE))
    except Exception as error:
        raise RuntimeError(
            "Le téléchargement automatique du dataset Kaggle a échoué. "
            "Vérifiez la connexion Internet et, si Kaggle le demande, la configuration des identifiants Kaggle. "
            f"Dataset attendu : {KAGGLE_DATASET_HANDLE}"
        ) from error


def find_dataset_csv(csv_files):
    for csv_file in csv_files:
        if is_valid_dataset_csv(csv_file):
            return csv_file
    return csv_files[0]


def is_valid_dataset_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        increase_csv_field_limit()
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = set(reader.fieldnames or [])
    except Exception:
        return False

    return DATASET_REQUIRED_COLUMNS.issubset(fieldnames)


def increase_csv_field_limit():
    field_size_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_size_limit)
            break
        except OverflowError:
            field_size_limit = int(field_size_limit / 10)


if __name__ == "__main__":
    main()
