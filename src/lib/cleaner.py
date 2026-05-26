from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import csv
import sys

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0
except ImportError as import_error:
    detect_langs = None
    LangDetectException = Exception
    _LANGDETECT_IMPORT_ERROR = import_error
else:
    _LANGDETECT_IMPORT_ERROR = None


DEFAULT_MISSING_MARKERS = {
    "",
    "abstract not found",
    "not found",
}


@dataclass
class BatchReport:
    batch_number: int
    start_row: int
    end_row: int
    total_rows: int
    kept_rows: int
    empty_abstracts: int
    non_english_abstracts: int


@dataclass
class CleaningReport:
    total_rows: int = 0
    kept_rows: int = 0
    empty_abstracts: int = 0
    non_english_abstracts: int = 0
    batches: list[BatchReport] = field(default_factory=list)

    @property
    def rejected_rows(self):
        return self.total_rows - self.kept_rows

    def summary(self):
        return {
            "total_rows": self.total_rows,
            "kept_rows": self.kept_rows,
            "rejected_rows": self.rejected_rows,
            "empty_abstracts": self.empty_abstracts,
            "non_english_abstracts": self.non_english_abstracts,
            "batch_count": len(self.batches),
        }


def clean_patent_csv(
    input_path,
    output_path=None,
    *,
    abstract_column="Abstract",
    batch_size=1000,
    remove_empty_abstracts=True,
    keep_only_english_abstracts=True,
    language_text_limit=4000,
    missing_markers=None,
    return_rows=False,
    progress_callback=None,
    encoding="utf-8",
):
    """
    Nettoie un CSV de brevets en lecture par lots.

    Par défaut, la fonction supprime les brevets dont l'abstract est vide
    ou vaut "Abstract not found", puis garde seulement les abstracts détectés
    comme anglais.

    Args:
        input_path: chemin du CSV source.
        output_path: chemin du CSV nettoyé. Si None, aucun fichier n'est écrit.
        abstract_column: nom de la colonne contenant les abstracts.
        batch_size: nombre de lignes traitées avant de produire un rapport de lot.
        remove_empty_abstracts: active la suppression des abstracts vides.
        keep_only_english_abstracts: active le filtrage anglais.
        language_text_limit: nombre maximal de caractères envoyés à langdetect.
        missing_markers: valeurs considérées comme manquantes.
        return_rows: si True, retourne aussi les lignes conservées en mémoire.
        progress_callback: fonction appelée avec chaque BatchReport.
        encoding: encodage du CSV.

    Returns:
        Si return_rows vaut False: CleaningReport.
        Si return_rows vaut True: tuple (CleaningReport, rows).
    """
    _increase_csv_field_limit()
    _validate_options(batch_size, keep_only_english_abstracts)

    input_path = Path(input_path)
    markers = _normalize_missing_markers(missing_markers)
    report = CleaningReport()
    current_batch = Counter()
    kept_rows = [] if return_rows else None

    output_file = None
    writer = None

    try:
        with input_path.open("r", encoding=encoding, newline="") as input_file:
            reader = csv.DictReader(input_file)
            if not reader.fieldnames:
                raise ValueError("Le CSV ne contient pas d'en-tête.")

            _validate_columns(reader.fieldnames, abstract_column)

            if output_path is not None:
                output_file = Path(output_path).open("w", encoding=encoding, newline="")
                writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
                writer.writeheader()

            for row in reader:
                keep_row = _process_row(
                    row=row,
                    report=report,
                    current_batch=current_batch,
                    abstract_column=abstract_column,
                    remove_empty_abstracts=remove_empty_abstracts,
                    keep_only_english_abstracts=keep_only_english_abstracts,
                    language_text_limit=language_text_limit,
                    missing_markers=markers,
                )

                if keep_row:
                    if writer is not None:
                        writer.writerow(row)
                    if kept_rows is not None:
                        kept_rows.append(row)

                if current_batch["total_rows"] >= batch_size:
                    _close_batch(report, current_batch, progress_callback)

            _close_batch(report, current_batch, progress_callback)
    finally:
        if output_file is not None:
            output_file.close()

    if return_rows:
        return report, kept_rows
    return report


def _process_row(
    row,
    report,
    current_batch,
    abstract_column,
    remove_empty_abstracts,
    keep_only_english_abstracts,
    language_text_limit,
    missing_markers,
):
    report.total_rows += 1
    current_batch["total_rows"] += 1

    keep_row = _passes_filters(
        row=row,
        abstract_column=abstract_column,
        remove_empty_abstracts=remove_empty_abstracts,
        keep_only_english_abstracts=keep_only_english_abstracts,
        language_text_limit=language_text_limit,
        missing_markers=missing_markers,
    )

    if keep_row:
        report.kept_rows += 1
        current_batch["kept_rows"] += 1
    elif _is_empty_abstract(row.get(abstract_column, ""), missing_markers):
        report.empty_abstracts += 1
        current_batch["empty_abstracts"] += 1
    else:
        report.non_english_abstracts += 1
        current_batch["non_english_abstracts"] += 1

    return keep_row


def _passes_filters(
    row,
    abstract_column,
    remove_empty_abstracts,
    keep_only_english_abstracts,
    language_text_limit,
    missing_markers,
):
    abstract = row.get(abstract_column, "")

    if remove_empty_abstracts and _is_empty_abstract(abstract, missing_markers):
        return False

    if keep_only_english_abstracts and not _is_english(abstract, language_text_limit):
        return False

    return True


def _is_empty_abstract(value, missing_markers):
    normalized_value = str(value or "").strip().lower()
    return normalized_value in missing_markers


def _is_english(value, language_text_limit):
    text = str(value or "").strip()
    if len(text) < 20:
        return False

    try:
        detections = detect_langs(text[:language_text_limit])
    except LangDetectException:
        return False

    return bool(detections and detections[0].lang == "en")


def _close_batch(report, current_batch, progress_callback):
    if current_batch["total_rows"] == 0:
        return

    batch_number = len(report.batches) + 1
    end_row = report.total_rows
    start_row = end_row - current_batch["total_rows"] + 1

    batch_report = BatchReport(
        batch_number=batch_number,
        start_row=start_row,
        end_row=end_row,
        total_rows=current_batch["total_rows"],
        kept_rows=current_batch["kept_rows"],
        empty_abstracts=current_batch["empty_abstracts"],
        non_english_abstracts=current_batch["non_english_abstracts"],
    )
    report.batches.append(batch_report)

    if progress_callback is not None:
        progress_callback(batch_report)

    current_batch.clear()


def _normalize_missing_markers(missing_markers):
    markers = DEFAULT_MISSING_MARKERS if missing_markers is None else missing_markers
    return {str(marker or "").strip().lower() for marker in markers}


def _validate_options(batch_size, keep_only_english_abstracts):
    if batch_size <= 0:
        raise ValueError("batch_size doit être supérieur à 0.")

    if keep_only_english_abstracts and detect_langs is None:
        raise ImportError(
            "La dépendance 'langdetect' est requise pour filtrer les abstracts en anglais. "
            "Installez-la avec: pip install -r requirements.txt"
        ) from _LANGDETECT_IMPORT_ERROR


def _validate_columns(fieldnames, abstract_column):
    if abstract_column not in fieldnames:
        raise ValueError(
            f"La colonne '{abstract_column}' est introuvable. "
            f"Colonnes disponibles: {', '.join(fieldnames)}"
        )


def _increase_csv_field_limit():
    field_size_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_size_limit)
            break
        except OverflowError:
            field_size_limit = field_size_limit // 10
