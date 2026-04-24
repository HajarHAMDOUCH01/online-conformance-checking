import pandas as pd
import re

CHEMIN_XLSX   = "/content/drive/MyDrive/PFE_data_CRAN/DataExtracted.xlsx"
CHEMIN_LOG    = "/content/drive/MyDrive/PFE_data_CRAN/event_log_normalise.csv"
CHEMIN_SORTIE = "/content/drive/MyDrive/PFE_data_CRAN/event_log_with_pathways.csv"

print("Loading files...")
df_diag = pd.read_excel(CHEMIN_XLSX, sheet_name="Sheet1", dtype={"Identifiant passage unique": str})
df_log  = pd.read_csv(CHEMIN_LOG, dtype={"case_id": str})

df_diag["case_id"] = df_diag["Identifiant passage unique"].astype(str).str.strip()
df_log["case_id"]  = df_log["case_id"].astype(str).str.strip()


COLS_KEEP = [
    "case_id",
    "CIM10 Diag Principal",
    "Motif recours",
    "Secteur",
    "Age(mois)",
    "Sexe",
    "CCMU",
    "GEMSA",
    "Durée séjour(h)",
    "Hospitalisé",
    "UHCD",
]
df_meta = df_diag[COLS_KEEP].copy()
df_meta = df_meta.rename(columns={
    "CIM10 Diag Principal": "cim10_principal",
    "Motif recours":        "motif_recours",
    "Secteur":              "secteur",
    "Age(mois)":            "age_mois",
    "Sexe":                 "sexe",
    "Durée séjour(h)":      "duree_sejour_h",
    "Hospitalisé":          "hospitalise",
    "UHCD":                 "uhcd",
})

df_meta = df_meta.drop_duplicates(subset="case_id", keep="first")

df_joined = df_log.merge(df_meta, on="case_id", how="left")

print(f"Event log rows           : {len(df_log)}")
print(f"After join               : {len(df_joined)}")
print(f"Cases with CIM10         : {df_joined['cim10_principal'].notna().sum()}")
print(f"Cases WITHOUT CIM10      : {df_joined['cim10_principal'].isna().sum()}")

def extraire_code_cim10(texte: str) -> str | None:
    """Extract the bare ICD-10 code from strings like '(J21.9)Bronchiolite...'"""
    if not isinstance(texte, str):
        return None
    m = re.search(r"\(([A-Z]\d{2}(?:\.\d+)?)\)", texte)
    if m:
        return m.group(1)
    m2 = re.search(r"([A-Z]\d{2}(?:\.\d+)?)", texte)
    return m2.group(1) if m2 else None

def extraire_chapitre_cim10(code: str | None) -> str | None:
    """Return ICD-10 chapter letter(s) from a code like 'J21.9' → 'J'"""
    if not isinstance(code, str):
        return None
    return code[0].upper()

def extraire_bloc_cim10(code: str | None) -> int | None:
    """Return numeric block (first 2 digits) from 'J21.9' → 21"""
    if not isinstance(code, str):
        return None
    m = re.search(r"[A-Z](\d{2})", code)
    return int(m.group(1)) if m else None


df_joined["cim10_code"]     = df_joined["cim10_principal"].apply(extraire_code_cim10)
df_joined["cim10_chapitre"] = df_joined["cim10_code"].apply(extraire_chapitre_cim10)
df_joined["cim10_bloc"]     = df_joined["cim10_code"].apply(extraire_bloc_cim10)

print("\nTop 30 ICD-10 codes in joined log:")
print(df_joined["cim10_code"].value_counts().head(30).to_string())
print("\nICD-10 chapters distribution:")
print(df_joined["cim10_chapitre"].value_counts().to_string())

CODE_OVERRIDES: dict[str, str] = {
    # Respiratory infections
    "J00":   "VOIES_RESPIRATOIRES_HAUTES",   # Rhinopharyngitis
    "J02.0": "VOIES_RESPIRATOIRES_HAUTES",   # Strep pharyngitis
    "J02.9": "VOIES_RESPIRATOIRES_HAUTES",   # Pharyngitis NOS
    "J04.0": "VOIES_RESPIRATOIRES_HAUTES",   # Laryngitis
    "J06.9": "VOIES_RESPIRATOIRES_HAUTES",   # Upper resp infection NOS
    "J09":   "VOIES_RESPIRATOIRES_BASSES",   # Pandemic flu
    "J10.8": "VOIES_RESPIRATOIRES_BASSES",   # Seasonal flu
    "J11.1": "VOIES_RESPIRATOIRES_BASSES",   # Flu NOS
    "J18.9": "VOIES_RESPIRATOIRES_BASSES",   # Pneumonia NOS
    "J21.0": "VOIES_RESPIRATOIRES_BASSES",   # RSV bronchiolitis
    "J21.9": "VOIES_RESPIRATOIRES_BASSES",   # Bronchiolitis NOS
    "J45.9": "ASTHME",
    "J45.0": "ASTHME",
    "J45.1": "ASTHME",
    # Gastro / digestive
    "A09.0": "GASTRO_DIGESTIF",
    "A09.9": "GASTRO_DIGESTIF",
    "K52.9": "GASTRO_DIGESTIF",
    "K59.0": "GASTRO_DIGESTIF",              # Constipation
    "R10.4": "GASTRO_DIGESTIF",              # Abdominal pain
    "R11":   "GASTRO_DIGESTIF",              # Nausea/vomiting
    "N10":   "GASTRO_DIGESTIF",              # UTI (handled under urologic below, kept here as fallback)
    # ENT
    "H65.0": "ORL",
    "H65.9": "ORL",
    "H66.0": "ORL",
    "H66.9": "ORL",
    # Trauma head / neurology
    "S06.00":"TRAUMATISME_CRANIEN",
    "S09.9": "TRAUMATISME_CRANIEN",
    # Febrile convulsions / neurologic
    "R56.0": "NEUROLOGIQUE",
    "G02.0": "NEUROLOGIQUE",
    # Fever / general symptoms
    "R50.9": "FIEVRE_SYMPTOMES_GENERAUX",
    "R68.1": "FIEVRE_SYMPTOMES_GENERAUX",
    "R53.+1":"FIEVRE_SYMPTOMES_GENERAUX",
    # Skin / wounds
    "S01.8": "PLAIE_SUTURE",
    "S01.5": "PLAIE_SUTURE",
    "S01.9": "PLAIE_SUTURE",
    "T14.1": "PLAIE_SUTURE",
    # Orthopedic / fractures
    "S93.4": "TRAUMATISME_ORTHOPEDIQUE",
    "S60.2": "TRAUMATISME_ORTHOPEDIQUE",
    # Urologic
    "N10":   "UROLOGIQUE",
    "N39.0": "UROLOGIQUE",
    # Admin / no act
    "Z53.9": "ADMINISTRATIF",
    "Z53.8": "ADMINISTRATIF",
    "Z71.1": "ADMINISTRATIF",
    "R05":   "FIEVRE_SYMPTOMES_GENERAUX",   # Cough only
}


def assigner_pathway_par_chapitre(chapitre: str | None, bloc: int | None,
                                   secteur: str | None) -> str:
    if chapitre is None:
        return "INCONNU"

    # Infectious / parasitic (A, B)
    if chapitre in ("A", "B"):
        return "GASTRO_DIGESTIF"  

    # Respiratory (J)
    if chapitre == "J":
        if bloc is not None:
            if bloc <= 6:  return "VOIES_RESPIRATOIRES_HAUTES"
            if bloc == 45: return "ASTHME"
            if 9 <= bloc <= 22: return "VOIES_RESPIRATOIRES_BASSES"
        return "VOIES_RESPIRATOIRES_BASSES"

    # Trauma / musculoskeletal (S, T, M)
    if chapitre == "S":
        if bloc is not None:
            if 0 <= bloc <= 9:   return "TRAUMATISME_CRANIEN"
            if 10 <= bloc <= 19: return "TRAUMATISME_CRANIEN"  # neck
            if 20 <= bloc <= 29: return "TRAUMATISME_THORACIQUE"
            if 30 <= bloc <= 39: return "TRAUMATISME_ABDOMINAL"
            if 40 <= bloc <= 49: return "TRAUMATISME_ORTHOPEDIQUE"   # shoulder/arm
            if 50 <= bloc <= 69: return "TRAUMATISME_ORTHOPEDIQUE"   # forearm/wrist/hand
            if 70 <= bloc <= 79: return "TRAUMATISME_ORTHOPEDIQUE"   # hip/thigh
            if 80 <= bloc <= 89: return "TRAUMATISME_ORTHOPEDIQUE"   # knee/leg
            if 90 <= bloc <= 99: return "TRAUMATISME_ORTHOPEDIQUE"   # ankle/foot
        # Use secteur if available
        return "TRAUMATISME_ORTHOPEDIQUE" if secteur == "Chirurgie" else "TRAUMATISME_CRANIEN"

    if chapitre == "T":
        return "PLAIE_SUTURE"   # Burns, injuries, foreign bodies

    if chapitre == "M":
        return "TRAUMATISME_ORTHOPEDIQUE"

    # Digestive (K)
    if chapitre == "K":
        return "GASTRO_DIGESTIF"

    # ENT / Eye (H)
    if chapitre == "H":
        if bloc is not None:
            if 0 <= bloc <= 59: return "OPHTALMOLOGIQUE"
            if 60 <= bloc <= 95: return "ORL"
        return "ORL"

    # Symptoms / general (R)
    if chapitre == "R":
        if bloc is not None:
            if 50 <= bloc <= 69: return "FIEVRE_SYMPTOMES_GENERAUX"
            if 10 <= bloc <= 19: return "GASTRO_DIGESTIF"   # abdominal pain/nausea
            if 0 <= bloc <= 9:   return "CARDIOVASCULAIRE"
        return "FIEVRE_SYMPTOMES_GENERAUX"

    # Genitourinary (N)
    if chapitre == "N":
        return "UROLOGIQUE"

    # Nervous system (G)
    if chapitre == "G":
        return "NEUROLOGIQUE"

    # Congenital / perinatal (P, Q)
    if chapitre in ("P", "Q"):
        return "NEONATOLOGIE"

    # Neoplasms (C, D)
    if chapitre in ("C", "D"):
        return "ONCOLOGIQUE"

    # Endocrine / metabolic (E)
    if chapitre == "E":
        return "METABOLIQUE"

    # Skin (L)
    if chapitre == "L":
        return "DERMATOLOGIQUE"

    # Cardiovascular (I)
    if chapitre == "I":
        return "CARDIOVASCULAIRE"

    # Admin / factors (Z, U)
    if chapitre in ("Z", "U"):
        return "ADMINISTRATIF"

    return "AUTRE"


def assigner_pathway(row) -> str:
    code      = row.get("cim10_code")
    chapitre  = row.get("cim10_chapitre")
    bloc      = row.get("cim10_bloc")
    secteur   = row.get("secteur")

    if isinstance(code, str):
        if code in CODE_OVERRIDES:
            return CODE_OVERRIDES[code]
        for prefix, pathway in CODE_OVERRIDES.items():
            if code.startswith(prefix):
                return pathway

    pathway = assigner_pathway_par_chapitre(chapitre, bloc, secteur)
    if pathway != "INCONNU":
        return pathway

    if isinstance(secteur, str):
        if "Chirurgie" in secteur:
            return "TRAUMATISME_ORTHOPEDIQUE"
        if "Médecine" in secteur:
            return "FIEVRE_SYMPTOMES_GENERAUX"

    return "INCONNU"


print("\nAssigning pathways...")
df_joined["pathway"] = df_joined.apply(assigner_pathway, axis=1)

print("\n" + "=" * 65)
print("PATHWAY DISTRIBUTION (events)")
print("=" * 65)
print(df_joined["pathway"].value_counts().to_string())

df_cases = df_joined.drop_duplicates(subset="case_id")[["case_id", "pathway", "secteur", "cim10_code", "cim10_principal", "motif_recours", "age_mois", "sexe", "CCMU", "GEMSA", "duree_sejour_h"]].copy()

print("\n" + "=" * 65)
print("PATHWAY DISTRIBUTION (cases)")
print("=" * 65)
print(df_cases["pathway"].value_counts().to_string())

print(f"\nTotal cases                : {len(df_cases)}")
print(f"Cases with pathway         : {(df_cases['pathway'] != 'INCONNU').sum()}")
print(f"Cases INCONNU              : {(df_cases['pathway'] == 'INCONNU').sum()}")

print("\n" + "=" * 65)
print("PATHWAY vs SECTEUR cross-tab (cases)")
print("=" * 65)
print(pd.crosstab(df_cases["pathway"], df_cases["secteur"].fillna("NaN")).to_string())

print("\n" + "=" * 65)
print("TOP 5 CIM10 codes per pathway")
print("=" * 65)
for pathway in sorted(df_cases["pathway"].unique()):
    sub = df_cases[df_cases["pathway"] == pathway]
    top = sub["cim10_code"].value_counts().head(5)
    print(f"\n  {pathway} ({len(sub)} cases):")
    print(top.to_string())

print("\n" + "=" * 65)
print("TRACE LENGTH STATS per pathway (events per case)")
print("=" * 65)
trace_len = df_joined.groupby("case_id").size().reset_index(name="trace_len")
df_cases2 = df_cases.merge(trace_len, on="case_id", how="left")
print(df_cases2.groupby("pathway")["trace_len"].describe().round(1).to_string())

df_joined.to_csv(CHEMIN_SORTIE, index=False)
print(f"\n Saved to {CHEMIN_SORTIE}")
print(f"  Shape: {df_joined.shape}")