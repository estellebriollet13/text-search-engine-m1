print("Importing modules...\n")

#add imports here
from src.lib.cleaner import clean_patent_csv

print("Modules imported successfully.\n")


print("Nettoyage du fichier CSV des brevets...")
#-----------------------------------------------------------------------------
# Nettoyage du fichier CSV des brevets
#------------------------------------------------------------------------------
try:
    report = clean_patent_csv(
        "data/patent_analysis_data.csv",
        output_path=None,
        batch_size=1000,
        remove_empty_abstracts=True,
        keep_only_english_abstracts=True,
    )
    print("CSV cleaning completed successfully.\n")
    print(f"Report:\n {report.summary()}\n")
except Exception as e:
    print(f"An error occurred during CSV cleaning: {e}")
    
