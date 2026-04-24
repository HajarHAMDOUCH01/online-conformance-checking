import re
import pandas as pd
import numpy as np
from typing import Optional, Tuple

CHEMIN_ENTREE  = "/content/drive/MyDrive/PFE_data_CRAN/event_log_classified.csv"
CHEMIN_SORTIE  = "/content/drive/MyDrive/PFE_data_CRAN/event_log_normalise.csv"
COLONNE_activity = "activity"


def pretraiter(etiquette: str) -> str:
    if not isinstance(etiquette, str):
        return ""
    s = etiquette.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" -")
    return s

MOTIF_CRP = re.compile(r"crp\s+capillaire\s+resultat", re.IGNORECASE)

_CRP_VALEUR_RE = re.compile(
    r"""
    (?:inf(?:erieure?|érieure?)?[\sà\sa]+)?
    (?:sup(?:erieure?|érieure?)?[\sà\sa]+)?
    (\d[\d,.]*)
    \s*(?:mg[l/]?|g[l/]?|mgl)?
    """,
    re.VERBOSE | re.IGNORECASE
)

_CRP_INF_RE = re.compile(
    r"inf(?:erieure?|érieure?|érieur)?[\sà\sa]+|<", re.IGNORECASE
)
_CRP_SUP_RE = re.compile(
    r"sup(?:erieure?|érieure?|érieur)?[\sà\sa]+|>", re.IGNORECASE
)


def extraire_valeur_crp(brut: str) -> Tuple[Optional[float], Optional[str]]:
    """Retourne (valeur_numerique, qualificatif) depuis une étiquette CRP.
    qualificatif ∈ {None, 'inferieur', 'superieur'}
    """
    qualificatif = None
    if _CRP_INF_RE.search(brut):
        qualificatif = "inferieur"
    elif _CRP_SUP_RE.search(brut):
        qualificatif = "superieur"

    m = _CRP_VALEUR_RE.search(brut)
    if m:
        try:
            valeur = float(m.group(1).replace(",", "."))
            return valeur, qualificatif
        except ValueError:
            pass
    return None, qualificatif


# -----------------------------------------------------------------------
# SECTION 3 — EXTRACTION RÉSULTAT TEST RAPIDE
# -----------------------------------------------------------------------

def extraire_resultat_test(brut: str) -> Optional[str]:
    """Retourne 'positif', 'negatif', ou None."""
    positif_kw = re.compile(
        r"pos(?:it(?:if|ive?|igs?|iig?)?)?$|"
        r"\bpositif\b|positig|posit$",
        re.IGNORECASE
    )
    negatif_kw = re.compile(
        r"neg(?:at(?:if|ive?|aif|tif)?)?$|"
        r"\bnegatif\b|\bnégatif\b|nég$|neg$|negtaif|négaitf|"
        r"\bneg\b|\bn2gatif\b|\bnrg\b",
        re.IGNORECASE
    )
    suffixe = re.split(r"r[eé]sultat\s*", brut, maxsplit=1)[-1].strip(" -")

    if positif_kw.search(suffixe):
        return "positif"
    if negatif_kw.search(suffixe):
        return "negatif"
    if suffixe in ("", "-", "0"):
        return "negatif"
    return None



def mc(*mots) -> re.Pattern:
    """Construit un motif regex insensible à la casse depuis une liste de mots-clés."""
    return re.compile("|".join(mots), re.IGNORECASE)


CORRECTIFS = [

    (
        re.compile(r"suture\s+d.{0,10}plaie\s+cutan[eé]e", re.IGNORECASE),
        "REALISER_SUTURE_SUPERFICIELLE",
        {"action": "realiser", "target": "suture_plaie", "condition": "superficielle"}
    ),
    (
        re.compile(r"surveillance\s+m[eé]dicalis[eé]e?\s+du\s+transport", re.IGNORECASE),
        "SURVEILLER_TRANSPORT_INTRAHOSPITALIER",
        {"action": "surveiller", "target": "transport"}
    ),
    (
        re.compile(r"pose\s+intraoss[ée]u[sx]?e?", re.IGNORECASE),
        "POSER_VOIE_INTRAOSSEUSE",
        {"action": "poser", "target": "voie_intraosseuse"}
    ),
    (
        re.compile(r"ex[eé]r[eè]se\s+(?:partielle|totale)", re.IGNORECASE),
        "ABLATION_TABLETTE_ONGLE",
        {"action": "ablation", "target": "tablette_ongle", "body_part": "ongle"}
    ),
    (
        re.compile(r"r[eé]duction\s+de\s+plusieurs\s+luxations?", re.IGNORECASE),
        "REDUIRE_LUXATION",
        {"action": "reduire", "target": "luxation", "condition": "multiple"}
    ),
    (
        re.compile(r"suture\s+de\s+plaies?\s+muqueuses?", re.IGNORECASE),
        "REALISER_SUTURE_SUPERFICIELLE",
        {"action": "realiser", "target": "suture_plaie",
         "body_part": "bouche", "condition": "superficielle"}
    ),
    (
        re.compile(
            r"(?:[a-z]{2,5}\d{3,6}\s+)?confection\s+d.{0,10}appareil\s+rigide",
            re.IGNORECASE
        ),
        "POSER_PLATRE_AVEC_FRACTURE",
        {"action": "poser", "target": "platre",
         "body_part": "main_poignet", "condition": "avec_fracture"}
    ),
]


REGLES: list[Tuple[re.Pattern, str, dict]] = [


    (
        re.compile(r"^crp\s+capillaire", re.IGNORECASE),
        "ENREGISTRER_RESULTAT_CRP",
        {"action": "enregistrer", "target": "CRP", "method": "capillaire"}
    ),


    (
        re.compile(r"r[ée]sultat\s+test\s+de\s+grippe", re.IGNORECASE),
        "ENREGISTRER_RESULTAT_GRIPPE",
        {"action": "enregistrer", "target": "test_grippe"}
    ),
    (
        re.compile(r"r[ée]sultat\s+test\s+de\s+vrs", re.IGNORECASE),
        "ENREGISTRER_RESULTAT_VRS",
        {"action": "enregistrer", "target": "test_VRS"}
    ),
    (
        re.compile(r"streptotest\s+r[ée]sultat", re.IGNORECASE),
        "ENREGISTRER_RESULTAT_STREP",
        {"action": "enregistrer", "target": "test_strep"}
    ),
    (
        re.compile(r"quicktest\s+r[ée]sultat", re.IGNORECASE),
        "ENREGISTRER_RESULTAT_QUICKTEST",
        {"action": "enregistrer", "target": "quicktest"}
    ),


    (
        re.compile(r"streptotest\s+prescription", re.IGNORECASE),
        "PRESCRIRE_TEST_STREP",
        {"action": "prescrire", "target": "test_strep"}
    ),
    (
        re.compile(r"quicktest\s+prescription", re.IGNORECASE),
        "PRESCRIRE_QUICKTEST",
        {"action": "prescrire", "target": "quicktest"}
    ),
    (
        re.compile(r"diagnostic\s+rapide\s+de\s+grippe", re.IGNORECASE),
        "REALISER_TEST_GRIPPE",
        {"action": "realiser", "target": "test_grippe"}
    ),
    (
        re.compile(r"diagnostic\s+rapide\s+de\s+vrs", re.IGNORECASE),
        "REALISER_TEST_VRS",
        {"action": "realiser", "target": "test_VRS"}
    ),
    (
        re.compile(r"r[ée]sultats?\s+des?\s+tdr", re.IGNORECASE),
        "ENREGISTRER_RESULTAT_TEST_RAPIDE",
        {"action": "enregistrer", "target": "test_diagnostic_rapide"}
    ),


    (
        re.compile(r"prise\s+de\s+la\s+temp[eé]rature|temp[eé]rature", re.IGNORECASE),
        "MESURER_TEMPERATURE",
        {"action": "mesurer", "target": "temperature", "ressource": "thermometre"}
    ),
    (
        re.compile(r"mesure\s+ponctuelle\s+spo2|spo2\s+seule|surveillance\s+spo2", re.IGNORECASE),
        "MESURER_SPO2",
        {"action": "mesurer", "target": "SpO2", "ressource": "oxymetre_pouls"}
    ),
    (
        re.compile(r"surveillance\s+scope.*spo2", re.IGNORECASE),
        "SURVEILLER_SCOPE_SPO2",
        {"action": "surveiller", "target": "scope_et_SpO2"}
    ),
    (
        re.compile(r"surveillance\s+scope", re.IGNORECASE),
        "SURVEILLER_SCOPE_CARDIAQUE",
        {"action": "surveiller", "target": "rythme_cardiaque", "ressource": "scope"}
    ),
    (
        re.compile(r"surveillance\s+spo2\s+seule", re.IGNORECASE),
        "MESURER_SPO2",
        {"action": "mesurer", "target": "SpO2", "ressource": "oxymetre_pouls"}
    ),
    (
        re.compile(r"surveillance\s+neurologique", re.IGNORECASE),
        "SURVEILLER_NEUROLOGIQUE",
        {"action": "surveiller", "target": "etat_neurologique"}
    ),
    (
        re.compile(r"glyc[ée]mie\s+capillaire", re.IGNORECASE),
        "MESURER_GLYCEMIE",
        {"action": "mesurer", "target": "glycemie_sanguine", "method": "capillaire"}
    ),
    (
        re.compile(r"ecg", re.IGNORECASE),
        "REALISER_ECG",
        {"action": "realiser", "target": "ECG", "ressource": "appareil_ECG"}
    ),
    (
        re.compile(r"test\s+hypotension\s+orthostatique", re.IGNORECASE),
        "MESURER_PRESSION_ORTHOSTATIQUE",
        {"action": "mesurer", "target": "pression_arterielle_orthostatique"}
    ),


    (
        re.compile(r"pr[eé]l[eè]vement\s+veineux", re.IGNORECASE),
        "PRELEVER_SANG",
        {"action": "prelever", "target": "sang", "method": "veineux"}
    ),
    (
        re.compile(r"pr[eé]l[eè]vement\s+art[eé]riel", re.IGNORECASE),
        "PRELEVER_SANG",
        {"action": "prelever", "target": "sang", "method": "arteriel"}
    ),
    (
        re.compile(r"pr[eé]l[eè]vement\s+urinaire|bandelette\s+urinaire", re.IGNORECASE),
        "PRELEVER_URINE",
        {"action": "prelever", "target": "urine"}
    ),
    (
        re.compile(r"pr[eé]l[eè]vement\s+de\s+selles", re.IGNORECASE),
        "PRELEVER_SELLES",
        {"action": "prelever", "target": "selles"}
    ),
    (
        re.compile(r"pr[eé]l[eè]vement\s+s[eé]cr[eé]tions|pr[eé]l[eè]vement\s+bact[eé]rio|pr[eé]l[eè]vement\s+sur\s+[eé]couvillon", re.IGNORECASE),
        "PRELEVER_SECRETIONS",
        {"action": "prelever", "target": "secretions_ou_bacteriologie"}
    ),
    (
        re.compile(r"aspiration\s+rhinopharyng[eé]|d[eé]sobstruction\s+rhinopharyng[eé]e?\s+drp", re.IGNORECASE),
        "REALISER_DESENCOMBREMENT_RHINOPHARYNGE",
        {"action": "realiser", "target": "desencombrement_rhinopharynge", "body_part": "rhinopharynx"}
    ),


    (
        re.compile(r"dispensation\s+de\s+m[eé]dicament\s+po|r[eé]hydratation\s+po", re.IGNORECASE),
        "ADMINISTRER_MEDICAMENT_ORAL",
        {"action": "administrer", "target": "medicament", "method": "orale"}
    ),
    (
        re.compile(r"dispensation\s+de\s+m[eé]dicament\s+intranasal", re.IGNORECASE),
        "ADMINISTRER_MEDICAMENT_INTRANASAL",
        {"action": "administrer", "target": "medicament", "method": "intranasale"}
    ),
    (
        re.compile(r"injection\s+iv\s+sur\s+chambre\s+implantable", re.IGNORECASE),
        "ADMINISTRER_INJECTION_IV_PORT",
        {"action": "administrer", "target": "medicament", "method": "IV_chambre_implantable"}
    ),
    (
        re.compile(r"injection\s+iv|perfusion", re.IGNORECASE),
        "ADMINISTRER_INJECTION_IV",
        {"action": "administrer", "target": "medicament", "method": "IV"}
    ),
    (
        re.compile(r"injection\s+d.{0,10}m[eé]dicament\s+intrarectal", re.IGNORECASE),
        "ADMINISTRER_MEDICAMENT_RECTAL",
        {"action": "administrer", "target": "medicament", "method": "rectale"}
    ),
    (
        re.compile(r"injection\s+(?:im|sc|id)", re.IGNORECASE),
        "ADMINISTRER_INJECTION_AUTRE",
        {"action": "administrer", "target": "medicament", "method": "IM_SC_ID"}
    ),
    (
        re.compile(r"r[eé]alisation\s+d.{0,5}a[eé]rosol|traitement\s+par\s+voie\s+inhal[eé]e", re.IGNORECASE),
        "ADMINISTRER_AEROSOL",
        {"action": "administrer", "target": "medicament", "method": "inhalation"}
    ),
    (
        re.compile(r"inhalation\s+de\s+meopa", re.IGNORECASE),
        "ADMINISTRER_MEOPA",
        {"action": "administrer", "target": "MEOPA", "method": "inhalation",
         "condition": "analgesique"}
    ),
    (
        re.compile(r"oxyg[eé]noth[eé]rapie", re.IGNORECASE),
        "ADMINISTRER_OXYGENE",
        {"action": "administrer", "target": "oxygene", "method": "inhalation"}
    ),
    (
        re.compile(r"mise\s+sous\s+airvo", re.IGNORECASE),
        "POSER_AIRVO",
        {"action": "poser", "target": "oxygene_haut_debit", "ressource": "AIRVO"}
    ),
    (
        re.compile(r"ventilation\s+ambu", re.IGNORECASE),
        "REALISER_VENTILATION_AMBU",
        {"action": "realiser", "target": "ventilation", "ressource": "masque_AMBU"}
    ),
    (
        re.compile(r"branchement.{0,15}alimentation\s+ent[eé]rale|rinçure\s+alimentation\s+ent[eé]rale", re.IGNORECASE),
        "GERER_NUTRITION_ENTERALE",
        {"action": "gerer", "target": "nutrition_enterale"}
    ),
    (
        re.compile(r"changement\s+de\s+poche\s+de\s+perfusion", re.IGNORECASE),
        "CHANGER_POCHE_PERFUSION",
        {"action": "changer", "target": "poche_perfusion"}
    ),
    (
        re.compile(r"goutte\s+[àa]\s+goutte\s+rectal|lavement\s+[eé]vacuateur", re.IGNORECASE),
        "ADMINISTRER_TRAITEMENT_RECTAL",
        {"action": "administrer", "target": "traitement_rectal", "method": "rectale"}
    ),


    (
        re.compile(r"pose\s+de\s+vvp", re.IGNORECASE),
        "POSER_VVP",
        {"action": "poser", "target": "voie_veineuse_peripherique"}
    ),
    (
        re.compile(r"ablation\s+de\s+vvp", re.IGNORECASE),
        "RETIRER_VVP",
        {"action": "retirer", "target": "voie_veineuse_peripherique"}
    ),
    (
        re.compile(r"pose\s+intraosseux?e", re.IGNORECASE),
        "POSER_VOIE_INTRAOSSEUSE",
        {"action": "poser", "target": "voie_intraosseuse"}
    ),
    (
        re.compile(r"pose\s+sng", re.IGNORECASE),
        "POSER_SONDE_NASOGASTRIQUE",
        {"action": "poser", "target": "sonde_nasogastrique"}
    ),
    (
        re.compile(r"pose\s+de\s+collecteur\s+urinaire", re.IGNORECASE),
        "POSER_COLLECTEUR_URINAIRE",
        {"action": "poser", "target": "collecteur_urinaire"}
    ),
    (
        re.compile(r"sondage\s+urinaire", re.IGNORECASE),
        "POSER_SONDE_URINAIRE",
        {"action": "poser", "target": "sonde_urinaire"}
    ),
    (
        re.compile(r"changement\s+gastrostomie|dilatation\s+trajet\s+gastrostomie|bouton\s+de\s+gastrostomie", re.IGNORECASE),
        "GERER_GASTROSTOMIE",
        {"action": "gerer", "target": "gastrostomie"}
    ),


    (
        re.compile(r"pose\s+xylocaine|pose\s+de\s+patch\s+emla", re.IGNORECASE),
        "POSER_ANESTHESIE_LOCALE",
        {"action": "poser", "target": "anesthesie_locale"}
    ),
    (
        re.compile(r"s[eé]ance\s+d.{0,5}hypnose", re.IGNORECASE),
        "REALISER_HYPNOSE_ANTALGIQUE",
        {"action": "realiser", "target": "hypnose", "condition": "antalgique"}
    ),
    (
        re.compile(r"cryoth[eé]rapie|application\s+de\s+froid|refroidissement\s+externe", re.IGNORECASE),
        "APPLIQUER_FROID",
        {"action": "appliquer", "target": "froid_therapeutique"}
    ),


    (
        re.compile(r"suture\s+d.{0,10}plaie\s+profonde", re.IGNORECASE),
        "REALISER_SUTURE_PROFONDE",
        {"action": "realiser", "target": "suture_plaie", "condition": "profonde"}
    ),
    (
        re.compile(r"suture\s+d.{0,10}plaie\s+superficielle|suture\s+d.{0,10}plaie\s+(?:du|de|muqueuse)", re.IGNORECASE),
        "REALISER_SUTURE_SUPERFICIELLE",
        {"action": "realiser", "target": "suture_plaie", "condition": "superficielle"}
    ),
    (
        re.compile(r"suture\s+d.{0,10}plaie\s+(?:transfixiante|pulpo)", re.IGNORECASE),
        "REALISER_SUTURE_SPECIALE",
        {"action": "realiser", "target": "suture_plaie", "condition": "speciale"}
    ),
    (
        re.compile(r"aide\s+[àa]\s+la\s+suture", re.IGNORECASE),
        "ASSISTER_SUTURE",
        {"action": "assister", "target": "suture_plaie"}
    ),
    (
        re.compile(r"suture\s+de\s+plusieurs\s+plaies|sutures?\s+multiples?", re.IGNORECASE),
        "REALISER_SUTURES_MULTIPLES",
        {"action": "realiser", "target": "sutures_multiples"}
    ),


    (
        re.compile(r"d[eé]sinfection\s+plaie", re.IGNORECASE),
        "DESINFECTER_PLAIE",
        {"action": "desinfecter", "target": "plaie"}
    ),
    (
        re.compile(r"pansement\s+initial\s+de\s+br[uû]lure", re.IGNORECASE),
        "PANSEMENT_BRULURE_INITIAL",
        {"action": "panser", "target": "plaie_brulure", "condition": "initial"}
    ),
    (
        re.compile(r"pansement\s+secondaire\s+de\s+br[uû]lure", re.IGNORECASE),
        "PANSEMENT_BRULURE_SECONDAIRE",
        {"action": "panser", "target": "plaie_brulure", "condition": "secondaire"}
    ),
    (
        re.compile(r"pansement\s+(?:petit|moyen|grand|initial|secondaire)", re.IGNORECASE),
        "REALISER_PANSEMENT",
        {"action": "panser", "target": "plaie"}
    ),
    (
        re.compile(r"soins\s+oculaires", re.IGNORECASE),
        "REALISER_SOINS_OCULAIRES",
        {"action": "realiser", "target": "soins_oculaires", "body_part": "oeil"}
    ),
    (
        re.compile(r"test\s+[àa]\s+la\s+fluor[eé]sc[eé]ine", re.IGNORECASE),
        "REALISER_TEST_FLUORESCEINE",
        {"action": "realiser", "target": "test_fluoresceine", "body_part": "oeil"}
    ),


    (
        re.compile(r"(?:bab|botte|manchette|r[eé]sine|pl[âa]tre|attelle).{0,50}avec\s+fracture", re.IGNORECASE),
        "POSER_PLATRE_AVEC_FRACTURE",
        {"action": "poser", "target": "platre", "condition": "avec_fracture"}
    ),
    (
        re.compile(r"(?:bab|botte|manchette|r[eé]sine|pl[âa]tre|attelle).{0,50}(?:sans\s+fracture|sans\s+r[eé]duction)", re.IGNORECASE),
        "POSER_PLATRE_SANS_FRACTURE",
        {"action": "poser", "target": "platre", "condition": "sans_fracture"}
    ),
    (
        re.compile(r"(?:bab|botte|manchette|pl[âa]tre|r[eé]sine|cruro|attelle).{0,80}fracture", re.IGNORECASE),
        "POSER_PLATRE_AVEC_FRACTURE",
        {"action": "poser", "target": "platre", "condition": "avec_fracture"}
    ),
    (
        re.compile(r"(?:bab|botte|manchette|pl[âa]tre|r[eé]sine|cruro|attelle)", re.IGNORECASE),
        "POSER_PLATRE",
        {"action": "poser", "target": "platre"}
    ),
    (
        re.compile(r"confection\s+d.{0,10}attelle", re.IGNORECASE),
        "POSER_ATTELLE",
        {"action": "poser", "target": "attelle"}
    ),
    (
        re.compile(r"bandage\s+clavicula|bandage\s+en\s+8", re.IGNORECASE),
        "POSER_BANDAGE_CLAVICULAIRE",
        {"action": "poser", "target": "bandage", "body_part": "clavicule"}
    ),
    (
        re.compile(r"bandage\s+coude\s+au\s+corps", re.IGNORECASE),
        "POSER_BANDAGE_COUDE",
        {"action": "poser", "target": "bandage", "body_part": "coude"}
    ),
    (
        re.compile(r"syndactylie\s+souple\s+pour\s+fracture", re.IGNORECASE),
        "POSER_SYNDACTYLIE_FRACTURE",
        {"action": "poser", "target": "syndactylie", "condition": "avec_fracture"}
    ),
    (
        re.compile(r"syndactylie\s+souple", re.IGNORECASE),
        "POSER_SYNDACTYLIE",
        {"action": "poser", "target": "syndactylie"}
    ),
    (
        re.compile(r"ablation\s+pl[âa]tre", re.IGNORECASE),
        "RETIRER_PLATRE",
        {"action": "retirer", "target": "platre"}
    ),


    (
        re.compile(r"r[eé]duction\s+de\s+(?:la\s+)?fracture|r[eé]duction\s+de\s+fracture|r[eé]duction\s+diaphysaire", re.IGNORECASE),
        "REDUIRE_FRACTURE",
        {"action": "reduire", "target": "fracture"}
    ),
    (
        re.compile(r"r[eé]duction\s+d.{0,10}fracture", re.IGNORECASE),
        "REDUIRE_FRACTURE",
        {"action": "reduire", "target": "fracture"}
    ),
    (
        re.compile(r"r[eé]duction\s+de\s+pronation\s+douloureuse", re.IGNORECASE),
        "REDUIRE_PRONATION_DOULOUREUSE",
        {"action": "reduire", "target": "pronation_douloureuse", "body_part": "coude"}
    ),
    (
        re.compile(r"r[eé]duction\s+de\s+(?:la\s+)?luxation|r[eé]duction\s+d.{0,5}une\s+luxation", re.IGNORECASE),
        "REDUIRE_LUXATION",
        {"action": "reduire", "target": "luxation"}
    ),


    (
        re.compile(r"[eé]vacuation\s+(?:par\s+incision\s+|hématome|par\s+ponction|de\s+collection)", re.IGNORECASE),
        "EVACUER_COLLECTION",
        {"action": "evacuer", "target": "collection_ou_hematome"}
    ),
    (
        re.compile(r"incision\s+de\s+panaris", re.IGNORECASE),
        "INCISER_PANARIS",
        {"action": "inciser", "target": "panaris"}
    ),
    (
        re.compile(r"ablation\s+(?:secondaire\s+)?de\s+(?:plusieurs\s+)?ce\s+(?:de\s+la\s+cavit[eé]\s+nasale|conduit\s+auditif|superficiel|profond)", re.IGNORECASE),
        "RETIRER_CORPS_ETRANGER",
        {"action": "retirer", "target": "corps_etranger"}
    ),
    (
        re.compile(r"ablation\s+de\s+(?:ce\s+de\s+la\s+cavit[eé]\s+nasale|ce\s+conduit|bouchon\s+de\s+c[eé]rumen)", re.IGNORECASE),
        "RETIRER_CORPS_ETRANGER",
        {"action": "retirer", "target": "corps_etranger"}
    ),
    (
        re.compile(r"ablation\s+secondaire\s+de\s+(?:plusieurs\s+)?ce", re.IGNORECASE),
        "RETIRER_CORPS_ETRANGER",
        {"action": "retirer", "target": "corps_etranger"}
    ),
    (
        re.compile(r"reposition\s+(?:faux\s+)?(?:de\s+l.)?ongle|suture\s+d.{0,10}plaie\s+pulpo", re.IGNORECASE),
        "GERER_ONGLE",
        {"action": "gerer", "target": "ongle"}
    ),
    (
        re.compile(r"ex[eé]r[eè]se\s+(?:partielle|totale)\s+de\s+la\s+tablette", re.IGNORECASE),
        "ABLATION_TABLETTE_ONGLE",
        {"action": "ablation", "target": "tablette_ongle"}
    ),
    (
        re.compile(r"r[eé]duction\s+manuelle\s+d.{0,5}un\s+paraphimosis", re.IGNORECASE),
        "REDUIRE_PARAPHIMOSIS",
        {"action": "reduire", "target": "paraphimosis", "body_part": "penis"}
    ),
    (
        re.compile(r"lib[eé]ration\s+d.{0,10}adh[eé]rences\s+du\s+pr[eé]puce", re.IGNORECASE),
        "LIBERER_ADHERENCES_PREPUCE",
        {"action": "liberer", "target": "adherences_prepuce", "body_part": "penis"}
    ),
    (
        re.compile(r"lambeau\s+de\s+recouvrement", re.IGNORECASE),
        "REALISER_LAMBEAU_CUTANE",
        {"action": "realiser", "target": "lambeau_cutane"}
    ),
    (
        re.compile(r"pose\s+d.{0,10}dispositif.{0,20}contention|hbld", re.IGNORECASE),
        "POSER_CONTENTION_DENTAIRE",
        {"action": "poser", "target": "contention_dentaire", "body_part": "dents"}
    ),

    (
        re.compile(r"[eé]chographie\s+cardiaque", re.IGNORECASE),
        "REALISER_ECHO_CARDIAQUE",
        {"action": "realiser", "target": "echocardiographie", "body_part": "coeur"}
    ),
    (
        re.compile(r"[eé]chographie\s+abdominale", re.IGNORECASE),
        "REALISER_ECHO_ABDOMINALE",
        {"action": "realiser", "target": "echographie", "body_part": "abdomen"}
    ),
    (
        re.compile(r"[eé]chographie\s+r[eé]nale", re.IGNORECASE),
        "REALISER_ECHO_RENALE",
        {"action": "realiser", "target": "echographie", "body_part": "rein"}
    ),
    (
        re.compile(r"[eé]chographie\s+(?:orl|du\s+cou)", re.IGNORECASE),
        "REALISER_ECHO_COU",
        {"action": "realiser", "target": "echographie", "body_part": "cou"}
    ),
    (
        re.compile(r"[eé]chographie\s+trans.fontanellaire", re.IGNORECASE),
        "REALISER_ECHO_TRANSFONTANELLAIRE",
        {"action": "realiser", "target": "echographie", "body_part": "fontanelle"}
    ),
    (
        re.compile(r"[eé]chographie\s+(?:pulmonaire|pleurale)", re.IGNORECASE),
        "REALISER_ECHO_PULMONAIRE",
        {"action": "realiser", "target": "echographie", "body_part": "poumon"}
    ),
    (
        re.compile(r"[eé]chographie\s+(?:muscle|tissus\s+mous)", re.IGNORECASE),
        "REALISER_ECHO_TISSUS_MOUS",
        {"action": "realiser", "target": "echographie", "body_part": "tissus_mous"}
    ),
    (
        re.compile(r"[eé]chographie\s+articulaire", re.IGNORECASE),
        "REALISER_ECHO_ARTICULAIRE",
        {"action": "realiser", "target": "echographie", "body_part": "articulation"}
    ),
    (
        re.compile(r"doppler\s+trans.cr[âa]nien", re.IGNORECASE),
        "REALISER_DOPPLER_TRANSCRANIEN",
        {"action": "realiser", "target": "doppler_transcranien", "body_part": "cerveau"}
    ),


    (
        re.compile(r"aide\s+[àa]\s+la\s+ponction\s+lombaire", re.IGNORECASE),
        "ASSISTER_PONCTION_LOMBAIRE",
        {"action": "assister", "target": "ponction_lombaire"}
    ),
    (
        re.compile(r"ponction\s+lombaire", re.IGNORECASE),
        "REALISER_PONCTION_LOMBAIRE",
        {"action": "realiser", "target": "ponction_lombaire"}
    ),


    (
        re.compile(r"[eé]ducation\s+parentale", re.IGNORECASE),
        "EDUCATION_PARENTALE",
        {"action": "eduquer", "target": "parent"}
    ),
    (
        re.compile(r"pr[eé]paration\s+des\s+repas|pr[eé]paration\s+des\s+biberons", re.IGNORECASE),
        "PREPARER_ALIMENTATION",
        {"action": "preparer", "target": "alimentation"}
    ),
    (
        re.compile(r"change\s+d.{0,5}un\s+b[eé]b[eé]", re.IGNORECASE),
        "CHANGER_COUCHE",
        {"action": "changer", "target": "couche"}
    ),
    (
        re.compile(r"toilette|bain", re.IGNORECASE),
        "REALISER_SOINS_HYGIENE",
        {"action": "realiser", "target": "hygiene"}
    ),
    (
        re.compile(r"effleurage", re.IGNORECASE),
        "REALISER_MASSAGE",
        {"action": "realiser", "target": "massage"}
    ),
    (
        re.compile(r"tire\s+lait", re.IGNORECASE),
        "UTILISER_TIRE_LAIT",
        {"action": "utiliser", "target": "tire_lait"}
    ),
]

REGLES = CORRECTIFS + REGLES

REGISTRE_META: dict[str, dict] = {}


def appliquer_regles(brut: str) -> Tuple[str, dict]:
    """Retourne (activity_processus, meta_dict). Fallback = activity_INCONNUE."""
    etiquette = pretraiter(brut)

    for motif, process_activity, meta in REGLES:
        if motif.search(etiquette):
            REGISTRE_META[process_activity] = meta
            return process_activity, meta

    return "activity_INCONNUE", {"action": "inconnu", "target": "inconnu"}

def extraire_body_part(brut: str) -> Optional[str]:
    etiquette = pretraiter(brut)
    correspondances = {
        "visage|face":                  "visage",
        "cuir chevelu":                 "cuir_chevelu",
        "main|mains":                   "main",
        "doigt|doigts|orteil":          "doigt_orteil",
        "paupi[eè]re":                  "paupiere",
        "l[eè]vre":                     "levre",
        "langue":                       "langue",
        "nez|nasale|nasal|rhinopharyng":"nez_rhinopharynx",
        "oreille|auricule|auditif":     "oreille",
        "sourcil":                      "sourcil",
        "poignet":                      "poignet",
        "coude":                        "coude",
        "avant.bras":                   "avant_bras",
        "hum[eé]rus":                   "humerus",
        "tibia|fibula":                 "jambe",
        "cheville|malleol":             "cheville",
        "pied|pédieux|avant.pied":      "pied",
        "rotule":                       "rotule",
        "[eé]paule":                    "epaule",
        "pouce":                        "pouce",
        "lombaire":                     "lombaire",
        "cr[âa]nien|fontanelle":        "tete",
        "pulmonaire|pleurale":          "poumon",
        "abdominale?":                  "abdomen",
        "cardiaque|coeur":              "coeur",
        "r[eé]nale?":                   "rein",
        "pr[eé]puce|p[eé]nis|phimosis|paraphimosis": "penis",
        "unguéale?|ongle":              "ongle",
    }
    for motif, partie in correspondances.items():
        if re.search(motif, etiquette, re.IGNORECASE):
            return partie
    return None


def extraire_condition(brut: str) -> Optional[str]:
    etiquette = pretraiter(brut)
    if re.search(r"superficielle?", etiquette, re.IGNORECASE):
        return "superficielle"
    if re.search(r"profonde?", etiquette, re.IGNORECASE):
        return "profonde"
    if re.search(r"avec\s+fracture|pour\s+fracture", etiquette, re.IGNORECASE):
        return "avec_fracture"
    if re.search(r"sans\s+fracture", etiquette, re.IGNORECASE):
        return "sans_fracture"
    if re.search(r"sans\s+r[eé]duction", etiquette, re.IGNORECASE):
        return "sans_reduction"
    if re.search(r"avec.{0,20}fracture\s+associ[eé]e", etiquette, re.IGNORECASE):
        return "avec_fracture_associee"
    if re.search(r"initial", etiquette, re.IGNORECASE):
        return "initial"
    if re.search(r"secondaire", etiquette, re.IGNORECASE):
        return "secondaire"
    if re.search(r"petit", etiquette, re.IGNORECASE):
        return "petit"
    if re.search(r"moyen", etiquette, re.IGNORECASE):
        return "moyen"
    if re.search(r"grand", etiquette, re.IGNORECASE):
        return "grand"
    return None


def extraire_taille_cm(brut: str) -> Optional[float]:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*cm", brut, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", "."))
    m2 = re.search(r"(\d+)\s+[àa]\s+(\d+)", brut)
    if m2:
        return (float(m2.group(1)) + float(m2.group(2))) / 2
    return None

CARTE_SEMANTIQUE: dict[str, tuple[str | None, str | None]] = {

    # Signes vitaux & surveillance
    "MESURER_TEMPERATURE":                  ("peau",            "thermometre"),
    "MESURER_SPO2":                         ("sang",            "oxymetre_pouls"),
    "MESURER_GLYCEMIE":                     ("sang",            "glucometre"),
    "MESURER_PRESSION_ORTHOSTATIQUE":       ("vaisseau_sanguin","sphygmomanometre"),
    "REALISER_ECG":                         ("coeur",           "appareil_ECG"),
    "SURVEILLER_SCOPE_CARDIAQUE":           ("coeur",           "scope_cardiaque"),
    "SURVEILLER_NEUROLOGIQUE":              ("cerveau",         None),
    "SURVEILLER_TRANSPORT_INTRAHOSPITALIER":(None,              "moniteur_transport"),
    "SURVEILLER_SCOPE_SPO2":                ("coeur",           "scope_cardiaque"),

    # Prélèvements
    "PRELEVER_SANG":                        ("sang",            "aiguille_seringue"),
    "ENREGISTRER_RESULTAT_CRP":             ("sang",            "analyseur_capillaire"),
    "PRELEVER_URINE":                       ("vessie",          "bandelette"),
    "POSER_COLLECTEUR_URINAIRE":            ("vessie",          "poche_urinaire"),
    "POSER_SONDE_URINAIRE":                 ("vessie",          "sonde_urinaire"),
    "PRELEVER_SELLES":                      ("colon",           "pot_selles"),
    "ADMINISTRER_TRAITEMENT_RECTAL":        ("colon",           None),
    "POSER_SONDE_NASOGASTRIQUE":            ("estomac",         "sonde_nasogastrique"),
    "GERER_GASTROSTOMIE":                   ("estomac",         "bouton_gastrostomie"),
    "GERER_NUTRITION_ENTERALE":             ("estomac",         "pompe_nutrition"),
    "CHANGER_POCHE_PERFUSION":              ("veine",           "ligne_IV"),
    "PRELEVER_SECRETIONS":                  ("rhinopharynx",    "ecouvillonnage"),

    # Voies respiratoires
    "REALISER_DESENCOMBREMENT_RHINOPHARYNGE":("rhinopharynx",   "dispositif_aspiration"),
    "ADMINISTRER_AEROSOL":                  ("poumon",          "nebuliseur"),
    "ADMINISTRER_MEOPA":                    ("poumon",          "masque_MEOPA"),
    "ADMINISTRER_OXYGENE":                  ("poumon",          "masque_oxygene"),
    "POSER_AIRVO":                          ("poumon",          "AIRVO"),
    "REALISER_VENTILATION_AMBU":            ("poumon",          "masque_AMBU"),

    # Accès vasculaire
    "POSER_VVP":                            ("veine",           "catheter_veineux"),
    "RETIRER_VVP":                          ("veine",           "catheter_veineux"),
    "POSER_VOIE_INTRAOSSEUSE":              ("moelle_osseuse",  "aiguille_IO"),
    "ADMINISTRER_INJECTION_IV":             ("veine",           "ligne_IV"),
    "ADMINISTRER_INJECTION_IV_PORT":        ("veine",           "chambre_implantable"),
    "ADMINISTRER_INJECTION_AUTRE":          (None,              "seringue"),
    "ADMINISTRER_MEDICAMENT_ORAL":          ("bouche",          None),
    "ADMINISTRER_MEDICAMENT_INTRANASAL":    ("nez_rhinopharynx","spray_nasal"),
    "ADMINISTRER_MEDICAMENT_RECTAL":        ("colon",           "applicateur_rectal"),

    # Tests rapides
    "PRESCRIRE_TEST_STREP":                 ("gorge",           "kit_test_rapide"),
    "ENREGISTRER_RESULTAT_STREP":           ("gorge",           "kit_test_rapide"),
    "REALISER_TEST_GRIPPE":                 ("rhinopharynx",    "kit_test_rapide"),
    "ENREGISTRER_RESULTAT_GRIPPE":          ("rhinopharynx",    "kit_test_rapide"),
    "REALISER_TEST_VRS":                    ("rhinopharynx",    "kit_test_rapide"),
    "ENREGISTRER_RESULTAT_VRS":             ("rhinopharynx",    "kit_test_rapide"),
    "ENREGISTRER_RESULTAT_QUICKTEST":       (None,              "kit_test_rapide"),
    "PRESCRIRE_QUICKTEST":                  (None,              "kit_test_rapide"),
    "ENREGISTRER_RESULTAT_TEST_RAPIDE":     (None,              "kit_test_rapide"),

    # Anesthésie / douleur
    "POSER_ANESTHESIE_LOCALE":              ("peau",            "aiguille_ou_patch"),
    "REALISER_HYPNOSE_ANTALGIQUE":          (None,              None),
    "APPLIQUER_FROID":                      ("peau",            "pack_froid"),

    # Soins plaies / sutures
    "REALISER_SUTURE_SUPERFICIELLE":        ("peau",            "kit_suture"),
    "REALISER_SUTURE_PROFONDE":             ("peau",            "kit_suture"),
    "REALISER_SUTURE_SPECIALE":             ("peau",            "kit_suture"),
    "REALISER_SUTURES_MULTIPLES":           ("peau",            "kit_suture"),
    "ASSISTER_SUTURE":                      ("peau",            "kit_suture"),
    "DESINFECTER_PLAIE":                    ("peau",            "antiseptique"),
    "REALISER_PANSEMENT":                   ("peau",            "kit_pansement"),
    "PANSEMENT_BRULURE_INITIAL":            ("peau",            "kit_pansement_brulure"),
    "PANSEMENT_BRULURE_SECONDAIRE":         ("peau",            "kit_pansement_brulure"),

    # Orthopédie
    "POSER_PLATRE_AVEC_FRACTURE":           ("os",              "materiau_platre"),
    "POSER_PLATRE_SANS_FRACTURE":           ("os",              "materiau_platre"),
    "POSER_PLATRE":                         ("os",              "materiau_platre"),
    "POSER_ATTELLE":                        ("os",              "attelle"),
    "RETIRER_PLATRE":                       ("os",              "scie_platre"),
    "POSER_SYNDACTYLIE":                    ("doigt_orteil",    "sparadrap"),
    "POSER_SYNDACTYLIE_FRACTURE":           ("doigt_orteil",    "sparadrap"),
    "POSER_BANDAGE_CLAVICULAIRE":           ("clavicule",       "bandage"),
    "POSER_BANDAGE_COUDE":                  ("coude",           "bandage"),
    "REDUIRE_FRACTURE":                     ("os",              None),
    "REDUIRE_LUXATION":                     ("articulation",    None),
    "REDUIRE_PRONATION_DOULOUREUSE":        ("coude",           None),

    # Imagerie
    "REALISER_ECHO_CARDIAQUE":              ("coeur",           "echographe"),
    "REALISER_ECHO_ABDOMINALE":             ("abdomen",         "echographe"),
    "REALISER_ECHO_RENALE":                 ("rein",            "echographe"),
    "REALISER_ECHO_COU":                    ("cou",             "echographe"),
    "REALISER_ECHO_TRANSFONTANELLAIRE":     ("cerveau",         "echographe"),
    "REALISER_ECHO_PULMONAIRE":             ("poumon",          "echographe"),
    "REALISER_ECHO_TISSUS_MOUS":            ("tissus_mous",     "echographe"),
    "REALISER_ECHO_ARTICULAIRE":            ("articulation",    "echographe"),
    "REALISER_DOPPLER_TRANSCRANIEN":        ("cerveau",         "echographe_doppler"),

    # Gestes chirurgicaux mineurs
    "REALISER_PONCTION_LOMBAIRE":           ("moelle_epiniere", "aiguille_lombaire"),
    "ASSISTER_PONCTION_LOMBAIRE":           ("moelle_epiniere", "aiguille_lombaire"),
    "EVACUER_COLLECTION":                   ("peau",            "bistouri"),
    "INCISER_PANARIS":                      ("ongle",           "bistouri"),
    "RETIRER_CORPS_ETRANGER":              (None,              "pince"),
    "GERER_ONGLE":                          ("ongle",           "kit_ongle"),
    "ABLATION_TABLETTE_ONGLE":              ("ongle",           "kit_ongle"),
    "REALISER_SOINS_OCULAIRES":             ("oeil",            "kit_soins_oculaires"),
    "REALISER_TEST_FLUORESCEINE":           ("oeil",            "lampe_fente"),
    "REDUIRE_PARAPHIMOSIS":                 ("penis",           None),
    "LIBERER_ADHERENCES_PREPUCE":           ("penis",           None),
    "REALISER_LAMBEAU_CUTANE":              ("peau",            "kit_chirurgical"),
    "POSER_CONTENTION_DENTAIRE":            ("dents",           "attelle_dentaire"),

    # Soins infirmiers
    "EDUCATION_PARENTALE":                  (None,              None),
    "PREPARER_ALIMENTATION":                ("estomac",         None),
    "CHANGER_COUCHE":                       (None,              None),
    "REALISER_SOINS_HYGIENE":              ("peau",            None),
    "REALISER_MASSAGE":                     ("peau",            None),
    "UTILISER_TIRE_LAIT":                   ("sein",            "tire_lait"),
}


def enrichir_ligne(brut: str) -> dict:
    etiquette = pretraiter(brut)
    process_activity, meta = appliquer_regles(brut)

    body_part = meta.get("body_part") or extraire_body_part(brut)
    ressource    = meta.get("ressource")
    condition    = meta.get("condition") or extraire_condition(brut)
    method      = meta.get("method")

    sem_corps, sem_ressource = CARTE_SEMANTIQUE.get(process_activity, (None, None))
    if body_part is None and sem_corps is not None:
        body_part = sem_corps
    if ressource is None and sem_ressource is not None:
        ressource = sem_ressource

    resultat = {
        "activity":             brut,
        "process_activity":   process_activity,
        "action":               meta.get("action"),
        "target":                meta.get("target"),
        "body_part":         body_part,
        "condition":            condition,
        "method":              method,
        "ressource":            ressource,
        "taille_cm":            extraire_taille_cm(brut),
        "valeur_crp":           None,
        "qualificatif_crp":     None,
        "resultat_test":        None,
    }

    if process_activity == "ENREGISTRER_RESULTAT_CRP":
        val, qual = extraire_valeur_crp(etiquette)
        resultat["valeur_crp"]       = val
        resultat["qualificatif_crp"] = qual

    if process_activity in {
        "ENREGISTRER_RESULTAT_STREP",
        "ENREGISTRER_RESULTAT_GRIPPE",
        "ENREGISTRER_RESULTAT_VRS",
        "ENREGISTRER_RESULTAT_QUICKTEST",
        "ENREGISTRER_RESULTAT_TEST_RAPIDE",
    }:
        resultat["resultat_test"] = extraire_resultat_test(etiquette)

    return resultat


def normaliser_journal_evenements(df: pd.DataFrame,
                                   colonne_activity: str = COLONNE_activity) -> pd.DataFrame:
    """Pipeline principal. Applique l'enrichissement complet à chaque ligne."""
    print(f"Forme d'entrée : {df.shape}")
    print(f"Activités uniques (brutes) : {df[colonne_activity].nunique()}")

    enrichi = df[colonne_activity].apply(lambda x: enrichir_ligne(str(x)))
    enrichi_df = pd.DataFrame(list(enrichi))

    autres_cols = [c for c in df.columns if c != colonne_activity]
    sortie = pd.concat([df[autres_cols].reset_index(drop=True),
                        enrichi_df.reset_index(drop=True)], axis=1)

    print(f"\nAprès normalisation :")
    print(f"Activités_processus uniques : {sortie['process_activity'].nunique()}")
    inconnus = sortie[sortie["process_activity"] == "activity_INCONNUE"]
    print(f"Lignes INCONNUES            : {len(inconnus)}")
    if len(inconnus):
        print("  Échantillon inconnus :")
        print(inconnus["activity"].value_counts().to_string())

    return sortie



if __name__ == "__main__":

    df = pd.read_csv(CHEMIN_ENTREE)
    df_propre = normaliser_journal_evenements(df, colonne_activity=COLONNE_activity)
    print("Parties du corps :\n")
    print(df_propre["body_part"].value_counts(dropna=False))
    print("\nProcess activities :\n")
    print(df_propre["process_activity"].value_counts(dropna=False))
    print("\nCategories : \n")
    print(df_propre["category"].value_counts(dropna=False))
    df_propre.to_csv(CHEMIN_SORTIE, index=False)
    print(f"\nSauvegardé → {CHEMIN_SORTIE}")

    print("\n" + "=" * 65)
    print("COMPLETE LOG : ")
    print("=" * 65)
    print(df_propre)