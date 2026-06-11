import csv
import sys
from pathlib import Path

import streamlit as st

from src.lib.MoteurRechercheTextuel import MoteurRechercheTextuel


DATA_PATH = Path("data/patent_analysis_data.csv")
TEXT_COLUMNS = ("Title", "Abstract")


def main():
    st.set_page_config(
        page_title="Moteur de recherche textuel",
        layout="wide",
    )

    st.title("Moteur de recherche textuel")

    tab_tfidf, tab_bm25, tab_next = st.tabs(["TF-IDF pur", "BM25 pur", "Prochaines méthodes"])

    with tab_tfidf:
        render_tfidf_tab()

    with tab_bm25:
        render_bm25_tab()

    with tab_next:
        render_next_methods_tab()


def render_tfidf_tab():
    render_search_tab(
        method_key="tfidf",
        score_name="TF-IDF",
        search_callback=lambda moteur, query: moteur.chercher_tfidf(query),
        score_description=(
            "Score : score TF-IDF calculé pour la requête ; plus il est élevé, "
            "plus le document contient fortement les termes recherchés."
        ),
    )


def render_bm25_tab():
    render_search_tab(
        method_key="bm25",
        score_name="BM25",
        search_callback=lambda moteur, query: moteur.chercher_bm25(query),
        score_description=(
            "Score : score BM25 calculé pour la requête ; plus il est élevé, "
            "plus le document est pertinent en tenant compte de la fréquence des termes et de la longueur du document."
        ),
    )


def render_search_tab(method_key, score_name, search_callback, score_description):
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

    patents = load_patents(max_documents)
    moteur = build_search_engine(patents)

    with controls_col:
        st.metric("Documents", len(patents))
        st.metric("Documents disponibles", total_documents)
        st.metric("Termes indexés", len(moteur.index))

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

        results = search_callback(moteur, query)
        st.caption(f"{len(results)} résultat(s) trouvé(s)")

        if not results:
            st.warning("Aucun document ne contient tous les mots de la requête.")
            return

        st.info(
            "Rang : position du résultat après tri par pertinence. "
            f"{score_description}"
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


def render_next_methods_tab():
    st.subheader("Extensions prévues")
    st.write(
        "Cet espace est réservé aux prochaines variantes du moteur, par exemple "
        "une recherche avec dictionnaire de synonymes ou une méthode de ranking différente."
    )


@st.cache_data(show_spinner="Comptage des documents...")
def count_searchable_patents():
    total = 0
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
    return moteur


def build_search_text(row):
    return " ".join(clean_value(row.get(column)) for column in TEXT_COLUMNS).strip()


def clean_value(value):
    return str(value or "").strip()


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
