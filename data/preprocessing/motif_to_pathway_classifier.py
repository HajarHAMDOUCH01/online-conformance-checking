import re
import unicodedata
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    train_test_split, cross_val_score
)
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight



CSV_PATH       = "/content/drive/MyDrive/PFE_data_CRAN/event_log_with_pathways.csv"
MODEL_OUT      = "/content/drive/MyDrive/PFE_data_CRAN/motif_pathway_model.pkl"
REPORT_OUT     = "/content/drive/MyDrive/PFE_data_CRAN/classification_report.txt"
CM_OUT         = "/content/drive/MyDrive/PFE_data_CRAN/confusion_matrix.png"

COL_TEXT       = "motif_recours"   
COL_TARGET     = "pathway"          

LABELS_TO_DROP = {"INCONNU", "ADMINISTRATIF"}

TEST_SIZE      = 0.20
RANDOM_STATE   = 42

print("=" * 65)
print("STEP 1 — Loading data")
print("=" * 65)

df_raw = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Raw shape: {df_raw.shape}")
print(f"Columns  : {list(df_raw.columns)}")

# one row per case_id
if "case_id" in df_raw.columns:
    df = df_raw.drop_duplicates(subset="case_id")[[COL_TEXT, COL_TARGET]].copy()
else:
    df = df_raw[[COL_TEXT, COL_TARGET]].drop_duplicates().copy()

print(f"After (per case): {len(df)} rows")

print("\n" + "=" * 65)
print("STEP 2 — Cleaning")
print("=" * 65)

def normaliser_texte(texte: str) -> str:
    """Lowercase, strip accents, punctuation removal, spaces removal"""
    if not isinstance(texte, str) or texte.strip() == "":
        return ""
    s = texte.lower().strip()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df[COL_TEXT] = df[COL_TEXT].apply(normaliser_texte)

df = df[df[COL_TEXT].str.len() > 2]
df = df[~df[COL_TARGET].isin(LABELS_TO_DROP)]
df = df.dropna(subset=[COL_TEXT, COL_TARGET])

print(f"After cleaning: {len(df)} rows")
print(f"\nPathway distribution:")
print(df[COL_TARGET].value_counts().to_string())

print("\n" + "=" * 65)
print("STEP 3 — Train/Test split")
print("=" * 65)

# Remove classes with fewer than 5 samples
counts = df[COL_TARGET].value_counts()
valid_classes = counts[counts >= 5].index
df = df[df[COL_TARGET].isin(valid_classes)]

X = df[COL_TEXT].values
y = df[COL_TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

STOPWORDS_FR_MED = [
    "le","la","les","de","du","des","un","une","en","au","aux",
    "et","ou","est","a","pour","sur","avec","par","dans","ce",
    "se","ne","pas","plus","qui","que","je","tu","il","elle",
    "nous","vous","ils","elles","on","mon","ma","mes","ton","ta",
    "tes","son","sa","ses","notre","votre","leur","leurs",
    
    "motif","consultation","urgence","urgences","pediatrique",
    "enfant","enfants","suite","lors","depuis","apres","avant",
    "sans","chez","vers","droit","gauche","droite","gauche",
]

tfidf_params = dict(
    analyzer="char_wb",   
    ngram_range=(2, 10),
    max_features=300_000,
    sublinear_tf=True,
    min_df=2,
    stop_words=STOPWORDS_FR_MED,
)

candidates = {
    "LinearSVC (calibrated)": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   CalibratedClassifierCV(
                      LinearSVC(class_weight="balanced", max_iter=2000, C=1.0),
                      cv=3
                  )),
    ]),
}


print("\n" + "=" * 65)
print("STEP 6 — Final training")
print("=" * 65)

best_pipeline = candidates["LinearSVC (calibrated)"]
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
print(f"  Test accuracy : {acc:.3f}")
print(f"  Test macro-F1 : {f1:.3f}")

report = classification_report(y_test, y_pred, zero_division=0)
print("\n" + report)

with open(REPORT_OUT, "w") as fh:
    fh.write(f"Model: LinearSVC (calibrated)\n")
    fh.write(f"Test accuracy : {acc:.3f}\n")
    fh.write(f"Test macro-F1 : {f1:.3f}\n\n")
    fh.write(report)
print(f"  Saved to {REPORT_OUT}")

labels = sorted(set(y_test) | set(y_pred))
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(max(10, len(labels)), max(8, len(labels) - 2)))
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot(ax=ax, colorbar=True, xticks_rotation=45, values_format="d")
ax.set_title(f"Confusion Matrix — LinearSVC (calibrated)\n"
             f"macro-F1={f1:.3f}  acc={acc:.3f}", fontsize=11)
plt.tight_layout()
plt.savefig(CM_OUT, dpi=150)
plt.close()
print(f"  Saved to {CM_OUT}")

print("\n" + "=" * 65)
print("STEP 8 — Top features per pathway")
print("=" * 65)

try:
    vec   = best_pipeline.named_steps["tfidf"]
    inner = best_pipeline.named_steps["clf"]
    real_clf = getattr(inner, "estimator", inner)
    if hasattr(real_clf, "coef_"):
        classes   = real_clf.classes_
        feat_names = vec.get_feature_names_out()
        for i, cls in enumerate(classes):
            coef = real_clf.coef_[i]
            top  = np.argsort(coef)[-8:][::-1]
            print(f"  {cls:35s}: {', '.join(feat_names[top])}")
except Exception as e:
    print(f"  (feature inspection skipped: {e})")

print("\n" + "=" * 65)
print("STEP 9 — Saving model")
print("=" * 65)

with open(MODEL_OUT, "wb") as fh:
    pickle.dump(best_pipeline, fh)
print(f"  Saved → {MODEL_OUT}")

print("\n" + "=" * 65)
print("USAGE EXAMPLE")
print("=" * 65)

def predict_pathway(motif: str, model_path: str = MODEL_OUT) -> dict:
    """
    Predict the clinical pathway from a raw motif de recours string.

    Returns
    -------
    dict with keys:
        pathway     – predicted class label
        confidence  – probability of the predicted class (if available)
        proba_all   – dict of all class probabilities
    """
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    motif_clean = normaliser_texte(motif)
    pathway_pred = model.predict([motif_clean])[0]

    result = {"pathway": pathway_pred, "confidence": None, "proba_all": {}}

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([motif_clean])[0]
        classes = model.classes_
        result["confidence"] = float(proba.max())
        result["proba_all"] = {
            cls: round(float(p), 3)
            for cls, p in sorted(zip(classes, proba),
                                  key=lambda x: -x[1])
        }

    return result

test_cases = [
    "fievre depuis 3 jours",
    "plaie au genou apres chute",
    "douleur abdominale vomissements",
    "toux sifflante dyspnee asthme",
    "traumatisme cheville entorse",
]

for motif in test_cases:
    result = predict_pathway(motif)
    conf_str = (f"  conf={result['confidence']:.2f}"
                if result["confidence"] else "")
    print(f"  '{motif}'")
    print(f"     {result['pathway']}{conf_str}")
    print()