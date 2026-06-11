import csv
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from src.lib.MoteurRechercheTextuel import MoteurRechercheTextuel


DATA_PATH = Path("data/patent_analysis_data.csv")
TEXT_COLUMNS = ("Title", "Abstract")
TARGET_PATENT_TITLE = "Artificial intelligence robot cleaner and robot cleaning system"
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
    "robots cleaning system",
    "intelligent robot cleaner",
    "AI robot vacuum",
    "automated robot cleaner",
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
]


def main():
    st.set_page_config(
        page_title="Moteur de recherche textuel",
        layout="wide",
    )

    st.title("Moteur de recherche textuel")

    (
        tab_tfidf,
        tab_bm25,
        tab_tfidf_synonyms,
        tab_bm25_synonyms,
        tab_tfidf_lemma,
        tab_bm25_lemma,
        tab_comparison,
        tab_next,
    ) = st.tabs(
        [
            "TF-IDF pur",
            "BM25 pur",
            "TF-IDF + synonymes",
            "BM25 + synonymes",
            "TF-IDF + lemmatisation",
            "BM25 + lemmatisation",
            "Comparaison",
            "Prochaines méthodes",
        ]
    )

    with tab_tfidf:
        render_tfidf_tab()

    with tab_bm25:
        render_bm25_tab()

    with tab_tfidf_synonyms:
        render_tfidf_synonyms_tab()

    with tab_bm25_synonyms:
        render_bm25_synonyms_tab()

    with tab_tfidf_lemma:
        render_tfidf_lemmatization_tab()

    with tab_bm25_lemma:
        render_bm25_lemmatization_tab()

    with tab_comparison:
        render_comparison_tab()

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
    )


def render_bm25_synonyms_tab():
    render_search_tab(
        method_key="bm25_synonyms",
        score_name="BM25 + synonymes",
        search_callback=lambda moteur, query: moteur.chercher_bm25_synonymes(query),
        score_description=(
            "Score : score BM25 calculé après enrichissement de la requête avec WordNet ; "
            "chaque mot peut être retrouvé via lui-même, son lemme ou un synonyme."
        ),
        requires_wordnet=True,
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
    )


def render_bm25_lemmatization_tab():
    render_search_tab(
        method_key="bm25_lemmatization",
        score_name="BM25 + lemmatisation",
        search_callback=lambda moteur, query: moteur.chercher_bm25_lemmatise(query),
        score_description=(
            "Score : score BM25 calculé après normalisation des formes fléchies ; "
            "il tient aussi compte de la fréquence des termes et de la longueur du document."
        ),
        requires_wordnet=True,
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


def render_comparison_tab():
    st.subheader("Comparaison des méthodes")
    st.write(
        "Ce tableau de bord teste si un brevet cible reste retrouvable quand la requête change "
        "de casse, d'ordre, de forme grammaticale ou de vocabulaire."
    )

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
            min_value=10,
            max_value=500,
            value=50,
            step=10,
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

    comparison_df = build_comparison_dataframe(moteur, target_doc_id, queries)
    if comparison_df.empty:
        st.warning("Aucune variante de requête à comparer.")
        return

    summary_df = build_comparison_summary(comparison_df, max_rank_display)
    query_summary_df = build_query_summary(comparison_df, max_rank_display)
    matrix_df = build_presence_matrix(comparison_df, max_rank_display)

    dashboard_tab, matrix_tab, details_tab = st.tabs(
        ["Dashboard", "Matrice de robustesse", "Détails"]
    )

    with dashboard_tab:
        render_comparison_dashboard(summary_df, query_summary_df, max_rank_display)

    with matrix_tab:
        render_comparison_matrix(matrix_df, max_rank_display)

    with details_tab:
        render_comparison_details(comparison_df, summary_df)

    st.subheader("Lecture rapide")
    st.write(
        "Une méthode est plus robuste si elle retrouve le brevet cible sur davantage de variantes "
        "et avec un rang faible. Les variantes en majuscules testent la casse ; les pluriels et formes "
        "comme cleaning testent la lemmatisation ; les mots proches comme intelligent ou vacuum testent "
        "l'apport des synonymes."
    )


def render_search_tab(
    method_key,
    score_name,
    search_callback,
    score_description,
    requires_wordnet=False,
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

        results = search_callback(moteur, query)
        st.caption(f"{len(results)} résultat(s) trouvé(s)")

        if not results:
            st.warning("Aucun résultat trouvé pour cette requête.")
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


def render_comparison_dashboard(summary_df, query_summary_df, max_rank_display):
    ranked_summary = rank_methods(summary_df)

    st.subheader("Top des méthodes")
    podium_cols = st.columns(3)
    for index, (_, method_row) in enumerate(ranked_summary.head(3).iterrows()):
        podium_cols[index].metric(
            f"Top {index + 1}",
            method_row["méthode"],
            delta=f"{method_row['taux de réussite']}% de réussite",
            delta_color="off",
        )

    chart_col_1, chart_col_2 = st.columns(2, gap="large")
    with chart_col_1:
        st.caption("Nombre de variantes qui retrouvent le brevet cible")
        st.bar_chart(
            ranked_summary.set_index("méthode")["requêtes trouvées"],
            use_container_width=True,
        )
    with chart_col_2:
        st.caption("Nombre de méthodes qui retrouvent chaque variante")
        st.bar_chart(
            query_summary_df.set_index("requête")["méthodes qui trouvent"],
            use_container_width=True,
        )

    st.subheader("Rang moyen")
    st.write(
        "Le rang est la position du brevet cible dans la liste des résultats : "
        "rang 1 veut dire premier résultat, rang 10 veut dire dixième résultat. "
        "Le rang moyen est cette position moyenne sur les requêtes où le brevet a été retrouvé. "
        "Plus il est bas, mieux c'est."
    )
    average_rank_series = ranked_summary.set_index("méthode")["rang moyen"].fillna(0)
    st.bar_chart(average_rank_series, use_container_width=True)


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
    detail_df = comparison_df.copy()
    detail_df["score"] = detail_df["score"].round(6)
    st.dataframe(detail_df, hide_index=True, use_container_width=True)


def get_search_methods():
    return [
        ("TF-IDF pur", lambda moteur, query: moteur.chercher_tfidf(query)),
        ("BM25 pur", lambda moteur, query: moteur.chercher_bm25(query)),
        ("TF-IDF + synonymes", lambda moteur, query: moteur.chercher_tfidf_synonymes(query)),
        ("BM25 + synonymes", lambda moteur, query: moteur.chercher_bm25_synonymes(query)),
        ("TF-IDF + lemmatisation", lambda moteur, query: moteur.chercher_tfidf_lemmatise(query)),
        ("BM25 + lemmatisation", lambda moteur, query: moteur.chercher_bm25_lemmatise(query)),
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


def build_comparison_dataframe(moteur, target_doc_id, queries):
    rows = []

    for query in queries:
        for method_name, search_callback in get_search_methods():
            results = search_callback(moteur, query)
            rank, score = find_result_rank_and_score(results, target_doc_id)
            rows.append(
                {
                    "requête": query,
                    "méthode": method_name,
                    "trouvé": rank is not None,
                    "rang": rank,
                    "score": score,
                }
            )

    return pd.DataFrame(rows)


def build_comparison_summary(comparison_df, max_rank_display):
    summary_rows = []
    query_count = comparison_df["requête"].nunique()

    for method_name, method_df in comparison_df.groupby("méthode", sort=False):
        found_df = method_df[method_df["trouvé"]]
        top_rank_df = found_df[found_df["rang"] <= max_rank_display]
        best_rank = int(found_df["rang"].min()) if not found_df.empty else None
        average_rank = round(found_df["rang"].mean(), 2) if not found_df.empty else None
        found_count = int(found_df.shape[0])
        top_count = int(top_rank_df.shape[0])
        summary_rows.append(
            {
                "méthode": method_name,
                "requêtes trouvées": found_count,
                "taux de réussite": round((found_count / query_count) * 100, 1),
                f"top {max_rank_display}": top_count,
                f"taux top {max_rank_display}": round((top_count / query_count) * 100, 1),
                "meilleur rang": best_rank,
                "rang moyen": average_rank,
                "meilleur score": round(found_df["score"].max(), 6) if not found_df.empty else 0.0,
            }
        )

    return pd.DataFrame(summary_rows)


def build_query_summary(comparison_df, max_rank_display):
    summary_rows = []

    for query, query_df in comparison_df.groupby("requête", sort=False):
        found_df = query_df[query_df["trouvé"]]
        top_rank_df = found_df[found_df["rang"] <= max_rank_display]
        best_rank = int(found_df["rang"].min()) if not found_df.empty else None
        best_methods = ", ".join(
            found_df[found_df["rang"] == found_df["rang"].min()]["méthode"].tolist()
        ) if not found_df.empty else ""
        summary_rows.append(
            {
                "requête": query,
                "méthodes qui trouvent": int(found_df.shape[0]),
                f"méthodes top {max_rank_display}": int(top_rank_df.shape[0]),
                "meilleur rang": best_rank,
                "meilleure(s) méthode(s)": best_methods,
            }
        )

    return pd.DataFrame(summary_rows)


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
        ["taux de réussite", f"taux top {extract_top_column(summary_df)}", "_rang_moyen_tri"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranked_summary["rang dashboard"] = ranked_summary.index + 1
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
