import csv
import re
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.lib.MoteurRechercheTextuel import MoteurRechercheTextuel


DATA_PATH = Path("data/patent_analysis_data.csv")
TEXT_COLUMNS = ("Title", "Abstract")
TARGET_PATENT_TITLE = "Artificial intelligence robot cleaner and robot cleaning system"
VISIBILITY_TOP_RANK = 10
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
    ) = st.tabs(
        [
            "TF-IDF pur",
            "BM25 pur",
            "TF-IDF + synonymes",
            "BM25 + synonymes",
            "TF-IDF + lemmatisation",
            "BM25 + lemmatisation",
            "Comparaison",
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
    problematic_queries = build_problematic_queries(comparison_df)

    render_comparison_dashboard(
        top10_summary_df,
        summary_df,
        comparison_df,
        family_matrix_df,
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
            "dans les résultats de la méthode."
        )
        st.write(
            "**Taux de réussite** : `requêtes trouvées / nombre total de variantes testées * 100`."
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
            "**Rang moyen** : position moyenne du brevet cible dans les résultats quand il est retrouvé. "
            "Rang 1 signifie premier résultat ; plus le rang moyen est bas, meilleure est la méthode."
        )
        st.write(
            "**Matrice de robustesse** : `1` signifie que le brevet cible est retrouvé dans le top N "
            "pour une requête et une méthode ; `0` signifie qu'il ne l'est pas."
        )


def render_comparison_dashboard(
    top10_summary_df,
    summary_df,
    comparison_df,
    family_matrix_df,
    problematic_queries,
    matrix_df,
    max_rank_display,
):
    ranked_summary = rank_methods(top10_summary_df)
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

    st.subheader("3. Analyse par famille de requêtes")
    render_family_analysis(family_matrix_df)

    st.subheader("4. Détails")
    render_problematic_queries(problematic_queries)
    render_comparison_matrix(matrix_df, max_rank_display)
    render_comparison_details(comparison_df, summary_df)


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
        best_rank = int(found_df["rang"].min()) if not found_df.empty else None
        average_rank = round(found_df["rang"].mean(), 2) if not found_df.empty else None
        found_count = int(found_df.shape[0])
        top_count = int(top_rank_df.shape[0])
        average_time = round(method_df["temps (ms)"].mean(), 2)
        summary_rows.append(
            {
                "méthode": method_name,
                "requêtes trouvées": found_count,
                "taux de réussite": round((found_count / query_count) * 100, 1),
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
        help="Nombre de formulations où le brevet cible est retrouvé / nombre total de formulations testées × 100.",
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
            "Règle de calcul : nombre de formulations où le brevet cible est retrouvé / "
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
    if set(query_tokens) == set(target_tokens) and query_tokens != target_tokens:
        return "Ordre des mots"
    if set(query_tokens).issubset(set(target_tokens)) and query_tokens != target_tokens:
        return "Ordre des mots"
    if has_plural_or_grammar_variation(query_tokens, target_tokens):
        return "Pluriel / forme grammaticale"
    if len(query_tokens) <= 1 or normalized_query in {"system", "robot", "cleaning", "artificial", "intelligence"}:
        return "Requête large"
    if any(token in {"golem", "cleanser", "automaton"} for token in query_tokens):
        return "Requête bruitée ou discutable"
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
