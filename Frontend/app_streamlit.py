import os
import base64
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

# =========================================================
# CREDIT RISK DASHBOARD — Colorful FinTech edition
# =========================================================
st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("CREDIT_RISK_API_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_URL = f"{API_URL}/predict"
HEALTH_URL = f"{API_URL}/health"

# Chemin de la photo : racine du projet / graphiques / maphoto.jpg
# Chemin exact vers ta photo
PHOTO_PATH = Path(__file__).resolve().parent.parent / "graphiques" / "maphoto..jpg"

# Lire la photo et la convertir en Base64 pour l afficher dans le HTML du dashboard.
with open(PHOTO_PATH, "rb") as image_file:
    photo_base64 = base64.b64encode(image_file.read()).decode("utf-8")

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ----------------------------- CSS -----------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --navy:#071b43;
    --navy2:#102c61;
    --blue:#246BFD;
    --cyan:#18B6D9;
    --green:#16A36A;
    --orange:#F59E0B;
    --red:#E65353;
    --ink:#17233F;
    --muted:#71809A;
    --line:#E8EDF5;
    --bg:#F4F7FC;
}
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
[data-testid="stAppViewContainer"] { background:var(--bg); }
[data-testid="stHeader"] { background:rgba(244,247,252,.92); }
.block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1600px; }
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#06183D 0%,#0B2B61 58%,#103B77 100%);
}
[data-testid="stSidebar"] * { color:#F7FAFF; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius:10px; padding:.55rem .7rem; margin:.12rem 0;
    transition:all .15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background:rgba(255,255,255,.12);
}
.brand-mark {
    width:48px;height:48px;border-radius:15px;
    display:flex;align-items:center;justify-content:center;
    background:linear-gradient(135deg,#2B78FF,#16B9D6);
    box-shadow:0 8px 22px rgba(23,112,255,.3);
    font-size:25px;margin-bottom:12px;
}
.brand-name {font-size:1.08rem;font-weight:800;letter-spacing:.06em;color:white;}
.brand-caption {font-size:.68rem;letter-spacing:.13em;color:#A9C5EF;margin-top:3px;}
.sidebar-status {
    background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.13);
    border-radius:12px;padding:12px;margin-top:14px;font-size:.82rem;
}
.topline {display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;}
/* Style du titre principal du dashboard */
.dashboard-main-title {
    font-size:1.85rem;
    font-weight:800;
    color:var(--ink);
    letter-spacing:-.045em;
    line-height:1.2;
    margin:0;
    padding:0;
    display:block;
}
.page-subtitle {color:var(--muted);font-size:.91rem;margin-top:5px;}
.user-chip {
    display:inline-flex;align-items:center;gap:9px;background:white;
    border:1px solid var(--line);border-radius:30px;padding:7px 13px;
    box-shadow:0 3px 12px rgba(20,40,80,.04);color:var(--ink);
}
.user-avatar {
    width:34px;height:34px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;background:linear-gradient(135deg,#DCEBFF,#B9D7FF);
    color:#174A9B;font-weight:800;object-fit:cover;
}
.hero {
    border-radius:20px;padding:24px 28px;color:white;
    background:linear-gradient(112deg,#09245A 0%,#1557B8 55%,#12A8C8 100%);
    box-shadow:0 12px 30px rgba(20,83,170,.18);
    position:relative;overflow:hidden;margin-bottom:18px;
}
.hero:after {
    content:"";position:absolute;width:260px;height:260px;border-radius:50%;
    right:-60px;top:-120px;background:rgba(255,255,255,.09);
}
.hero-title {font-size:1.55rem;font-weight:800;position:relative;z-index:1;}
.hero-text {font-size:.9rem;color:#DCEBFF;margin-top:5px;position:relative;z-index:1;}
.hero-tag {
    display:inline-block;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.2);
    padding:5px 10px;border-radius:30px;font-size:.72rem;margin-top:13px;position:relative;z-index:1;
}
.section-heading {
    color:var(--ink);font-size:1.06rem;font-weight:800;
    margin:.65rem 0 .8rem 0;letter-spacing:-.02em;
}
.section-sub {color:var(--muted);font-size:.8rem;margin-top:-7px;margin-bottom:12px;}
.metric-card {
    background:white;border:1px solid var(--line);border-radius:16px;
    padding:16px 17px;min-height:125px;box-shadow:0 5px 18px rgba(23,45,84,.045);
}
.metric-top {display:flex;justify-content:space-between;align-items:center;}
.metric-icon {
    width:39px;height:39px;border-radius:12px;display:flex;align-items:center;
    justify-content:center;font-size:19px;
}
.metric-label {font-size:.76rem;font-weight:700;color:var(--muted);margin-top:10px;}
.metric-value {font-size:1.55rem;font-weight:800;color:var(--ink);letter-spacing:-.04em;margin-top:1px;}
.metric-note {font-size:.72rem;color:var(--muted);margin-top:2px;}
.panel {
    background:white;border:1px solid var(--line);border-radius:17px;
    padding:17px 18px;box-shadow:0 5px 18px rgba(23,45,84,.045);height:100%;
}
.panel-title {font-size:.95rem;font-weight:800;color:var(--ink);margin-bottom:12px;}
.info-row {display:flex;justify-content:space-between;gap:10px;padding:8px 0;border-bottom:1px solid #F0F3F8;font-size:.82rem;}
.info-row:last-child {border-bottom:0;}
.info-label {color:var(--muted);}
.info-value {color:var(--ink);font-weight:700;text-align:right;}
.pill {display:inline-block;padding:5px 10px;border-radius:30px;font-size:.72rem;font-weight:700;}
.pill-green {background:#E6F7EF;color:#148052;}
.pill-orange {background:#FFF3DD;color:#A66500;}
.pill-red {background:#FDEBEC;color:#B8323B;}
.callout {
    border-radius:12px;padding:12px 14px;font-size:.84rem;font-weight:600;
    border:1px solid transparent;
}
.callout-green {background:#EAF8F0;color:#14784A;border-color:#CDEEDB;}
.callout-orange {background:#FFF6E6;color:#986000;border-color:#F8E5BB;}
.callout-red {background:#FFF0F0;color:#B12D38;border-color:#F5D2D4;}
div.stButton > button {
    border:0;border-radius:10px;background:linear-gradient(100deg,#246BFD,#168DDE);
    color:white;font-weight:700;min-height:2.75rem;box-shadow:0 5px 14px rgba(36,107,253,.16);
}
div.stButton > button:hover {color:white;border:0;background:linear-gradient(100deg,#1859DB,#087EBF);}
div[data-testid="stForm"] {
    background:white;border:1px solid var(--line);border-radius:16px;padding:18px;
}
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:12px;overflow:hidden;}
.stProgress > div > div > div > div {background:linear-gradient(90deg,#246BFD,#13B6C9);}
hr {border-color:var(--line);}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ----------------------------- Helpers -------------------
def api_is_online():
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.ok
    except requests.RequestException:
        return False


def predict(payload):
    r = requests.post(PREDICT_URL, json=payload, timeout=20)
    r.raise_for_status()
    result = r.json()
    for key in ("prediction", "probabilite_defaut", "score_risque_pourcentage"):
        if key not in result:
            raise ValueError(f"Champ absent dans la réponse API : {key}")
    return result


def risk_details(prob):
    if prob < 0.20:
        return "Risque faible", "green", "Analyse complémentaire habituelle"
    if prob < 0.50:
        return "Risque modéré", "orange", "Examen complémentaire recommandé"
    return "Risque élevé", "red", "Examen approfondi recommandé"


def metric_card(icon, label, value, note, color, tint):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-top">
        <div class="metric-icon" style="background:{tint};color:{color}">{icon}</div>
        <span style="font-size:.68rem;color:{color};font-weight:800">● LIVE</span>
      </div>
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def risk_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 34, "color": "#17233F"}},
        title={"text": "Probabilité de défaut", "font": {"size": 14, "color": "#71809A"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)"},
            "bar": {"color": "#246BFD", "thickness": .26},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "#DDF5E8"},
                {"range": [20, 50], "color": "#FFF0CC"},
                {"range": [50, 100], "color": "#FCE1E2"},
            ],
            "threshold": {"line": {"color": "#17233F", "width": 3}, "thickness": .8, "value": prob * 100},
        },
    ))
    fig.update_layout(height=235, margin=dict(l=20, r=20, t=45, b=5), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ----------------------------- Sidebar -------------------
with st.sidebar:
    # Afficher le titre principal du dashboard
    st.markdown(
    '<div class="dashboard-main-title">Credit Risk Dashboard</div>',
    unsafe_allow_html=True
)
    st.markdown('<div class="brand-name">CREDIT INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-caption">RISK ANALYTICS PLATFORM</div>', unsafe_allow_html=True)
    st.write("")
    st.divider()

    pages = {
        "⌂  Vue d'ensemble": "overview",
        "◎  Nouvelle prédiction": "prediction",
        "▤  Analyse portefeuille": "portfolio",
        "◷  Historique": "history",
        "⚙  Paramètres": "settings",
    }
    selected = st.radio("ESPACE DE TRAVAIL", list(pages.keys()), label_visibility="visible")
    page = pages[selected]

    st.divider()
    online = api_is_online()
    state_text = "Moteur IA connecté" if online else "Moteur IA indisponible"
    state_color = "#35D49A" if online else "#FFB454"
    st.markdown(f"""
      <div class="sidebar-status">
        <div style="font-weight:700;margin-bottom:5px">STATUT DU SYSTÈME</div>
        <span style="color:{state_color}">●</span>
        <span style="color:white">{state_text}</span>
        <div style="color:#A9C5EF;font-size:.7rem;margin-top:7px">Credit Risk · Version 2.0</div>
      </div>
    """, unsafe_allow_html=True)
    st.caption("Démonstration — ne pas utiliser pour une décision réelle de crédit.")


# ----------------------------- Header --------------------
head1, head2 = st.columns([3, 1])
with head1:
    st.markdown('<div class="page-title">Credit Risk Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Analyse intelligente et suivi du risque de crédit</div>', unsafe_allow_html=True)
with head2:
    st.markdown(f"""
      <div style="display:flex;justify-content:flex-end;padding-top:7px">
        <div class="user-chip">
          <img class="user-avatar" src="data:image/jpeg;base64,{photo_base64}" alt="Photo de profil">
          <div><div style="font-size:.78rem;font-weight:800">Data Scientist</div>
          <div style="font-size:.68rem;color:#71809A">Workspace</div></div>
        </div>
      </div>
    """, unsafe_allow_html=True)

st.write("")


# =========================================================
# OVERVIEW
# =========================================================
if page == "overview":
    st.markdown("""
      <div class="hero">
        <div class="hero-title">Bienvenue dans votre espace Credit Risk</div>
        <div class="hero-text">Pilotez vos analyses, estimez les probabilités de défaut et explorez vos résultats.</div>
        <span class="hero-tag">✦ AI-POWERED CREDIT ANALYTICS</span>
      </div>
    """, unsafe_allow_html=True)

    history = st.session_state.prediction_history
    n_pred = len(history)
    last_pd = f"{history[-1]['pd']:.2f}%" if history else "—"
    n_high = sum(1 for x in history if x["risk"] == "Risque élevé")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("🧠", "MODÈLE ACTIF", "XGBoost", "Classification binaire", "#246BFD", "#EAF1FF")
    with m2:
        metric_card("⚡", "STATUT API", "En ligne" if online else "Hors ligne", "Moteur de prédiction", "#16A36A" if online else "#E49A22", "#E7F7EF" if online else "#FFF3DD")
    with m3:
        metric_card("📊", "PRÉDICTIONS DE CETTE SESSION", str(n_pred), "Résultats calculés depuis cette session", "#0D9EBB", "#E4F8FB")
    with m4:
        metric_card("🎯", "DERNIÈRE PROBABILITÉ", last_pd, "Probabilité de défaut estimée", "#A16AE8", "#F3EBFF")

    st.markdown('<div class="section-heading">Analyse et activité</div>', unsafe_allow_html=True)
    left, center, right = st.columns([1.08, 1.35, 1.08])

    with left:
        st.markdown('<div class="panel"><div class="panel-title">Borrower Risk Snapshot</div>', unsafe_allow_html=True)
        if history:
            last = history[-1]
            st.markdown(f'<div class="metric-label">DERNIER DOSSIER ANALYSÉ</div><div style="font-size:1.25rem;font-weight:800;color:#17233F">{last["risk"]}</div><div style="margin:8px 0"><span class="pill pill-green">Classe {last["prediction"]}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-row"><span class="info-label">Probabilité de défaut</span><span class="info-value">{last["pd"]:.2f}%</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-row"><span class="info-label">Date d’analyse</span><span class="info-value">{last["date"]}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="padding:20px 4px;text-align:center;color:#71809A">🗂️<br><b style="color:#17233F">Aucun dossier analysé</b><br><small>Lancez une prédiction pour alimenter ce tableau.</small></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with center:
        st.markdown('<div class="panel"><div class="panel-title">Credit Risk Distribution</div>', unsafe_allow_html=True)
        if history:
            counts = pd.Series([x["risk"] for x in history]).value_counts()
            colors = {"Risque faible": "#16A36A", "Risque modéré": "#F3A712", "Risque élevé": "#E65353"}
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values, hole=.66,
                marker=dict(colors=[colors.get(x, "#246BFD") for x in counts.index],
                            line=dict(color="white", width=4)),
                textinfo="percent", textfont=dict(size=13, color="white"),
            ))
            fig.update_layout(
                height=225, margin=dict(l=5, r=5, t=5, b=5),
                showlegend=True, legend=dict(orientation="h", y=-.08, x=.05, font=dict(size=10)),
                paper_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(text=f"<b>{n_pred}</b><br><span style='font-size:10px'>analyses</span>",
                                  x=.5, y=.5, showarrow=False, font=dict(size=20, color="#17233F"))],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown('<div style="height:190px;display:flex;align-items:center;justify-content:center;color:#71809A;text-align:center">Le graphique apparaîtra après vos premières prédictions.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><div class="panel-title">Portfolio Alerts</div>', unsafe_allow_html=True)
        if history:
            st.markdown(f'<div class="info-row"><span class="info-label">Risque élevé</span><span class="pill pill-red">{n_high}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-row"><span class="info-label">Dossiers analysés</span><span class="info-value">{n_pred}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-row"><span class="info-label">Dernier résultat</span><span class="info-value">{last_pd}</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="callout callout-orange" style="margin-top:14px">⚠️ Les niveaux de risque sont indicatifs et doivent être validés.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="callout callout-orange">ℹ️ Aucune alerte : aucun dossier n’a encore été évalué.</div>', unsafe_allow_html=True)
            st.markdown('<p style="color:#71809A;font-size:.8rem;margin-top:12px">Les alertes seront générées à partir des prédictions réellement effectuées.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Accès rapide</div>', unsafe_allow_html=True)
    qa, qb = st.columns(2)
    with qa:
        if st.button("＋  Nouvelle évaluation", use_container_width=True):
            st.info("Choisissez « Nouvelle prédiction » dans le menu latéral.")
    with qb:
        if st.button("▤  Consulter l’historique", use_container_width=True):
            st.info("Choisissez « Historique » dans le menu latéral.")


# =========================================================
# PREDICTION
# =========================================================
elif page == "prediction":
    st.markdown("""
      <div class="hero">
        <div class="hero-title">Nouvelle évaluation de crédit</div>
        <div class="hero-text">Saisissez les informations de l’emprunteur. Le modèle XGBoost estimera la probabilité de défaut.</div>
        <span class="hero-tag">✦ PREDICTIVE SCORING</span>
      </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown('<div class="section-heading">👤 Profil de l’emprunteur</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Âge", min_value=18, max_value=100, value=32)
            revenu_annuel = st.number_input("Revenu annuel (€)", min_value=0.0, value=40000.0, step=1000.0)
            statut_logement = st.selectbox("Statut du logement", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            anciennete_emploi = st.number_input("Ancienneté professionnelle (années)", min_value=0.0, value=5.0, step=.5)
        with c2:
            motif_pret = st.selectbox("Motif du prêt", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
            niveau_de_risque = st.selectbox("Grade / niveau du prêt", ["Low", "Medium", "High"])
            montant_pret = st.number_input("Montant du prêt (€)", min_value=0.0, value=12000.0, step=500.0)
            taux_interet = st.number_input("Taux d’intérêt (%)", min_value=0.0, value=9.5, step=.1)
        with c3:
            ratio_pret_revenu = st.number_input("Ratio prêt / revenu", min_value=0.0, value=.30, step=.01)
            defaut_historique_credit = st.selectbox("Défaut antérieur", ["N", "Y"])
            duree_historique_credit = st.number_input("Ancienneté historique crédit (années)", min_value=0, value=4)
            st.markdown('<div class="callout callout-orange">Vérifiez les catégories : elles doivent correspondre aux données utilisées lors de l’entraînement.</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("⚡  Lancer l’évaluation du risque", use_container_width=True)

    if submitted:
        payload = {
            "age": int(age),
            "revenu_annuel": float(revenu_annuel),
            "statut_logement": statut_logement,
            "anciennete_emploi": float(anciennete_emploi),
            "motif_pret": motif_pret,
            "niveau_de_risque": niveau_de_risque,
            "montant_pret": float(montant_pret),
            "taux_interet": float(taux_interet),
            "ratio_pret_revenu": float(ratio_pret_revenu),
            "defaut_historique_credit": defaut_historique_credit,
            "duree_historique_credit": int(duree_historique_credit),
        }
        try:
            with st.spinner("Analyse du dossier par le modèle XGBoost..."):
                result = predict(payload)
            prob = float(result["probabilite_defaut"])
            label, color, advice = risk_details(prob)
            pct = prob * 100
            record = {
                "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "pd": pct,
                "risk": label,
                "prediction": int(result["prediction"]),
            }
            st.session_state.prediction_history.append(record)

            st.markdown('<div class="section-heading">Résultat de l’évaluation</div>', unsafe_allow_html=True)
            r1, r2 = st.columns([1.05, 1.45])
            with r1:
                st.markdown('<div class="panel"><div class="panel-title">Credit Risk Score</div>', unsafe_allow_html=True)
                st.plotly_chart(risk_gauge(prob), use_container_width=True, config={"displayModeBar": False})
                pill_class = {"green": "pill-green", "orange": "pill-orange", "red": "pill-red"}[color]
                st.markdown(f'<div style="text-align:center"><span class="pill {pill_class}">{label}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with r2:
                st.markdown('<div class="panel"><div class="panel-title">Analyse du dossier</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">PROBABILITÉ ESTIMÉE DE DÉFAUT</div><div style="font-size:2.35rem;font-weight:800;color:#17233F">{pct:.2f}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-row"><span class="info-label">Classe prédite</span><span class="info-value">{int(result["prediction"])}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-row"><span class="info-label">Niveau de risque</span><span class="info-value">{label}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-row"><span class="info-label">Interprétation</span><span class="info-value">{advice}</span></div>', unsafe_allow_html=True)
                st.markdown('<div class="callout callout-orange" style="margin-top:16px">Cette estimation est une aide à l’analyse. Elle ne constitue pas une décision automatique d’octroi ou de refus.</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        except requests.RequestException as exc:
            st.error(f"Impossible de contacter l’API : {exc}")
        except (KeyError, ValueError, TypeError) as exc:
            st.error(f"Réponse inattendue du modèle : {exc}")


# =========================================================
# PORTFOLIO
# =========================================================
elif page == "portfolio":
    st.markdown('<div class="page-title">Analyse du portefeuille</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Évaluation de plusieurs dossiers à partir d’un fichier CSV.</div>', unsafe_allow_html=True)
    st.write("")
    uploaded = st.file_uploader("Importer un fichier CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.markdown('<div class="section-heading">Aperçu des données importées</div>', unsafe_allow_html=True)
            st.dataframe(df.head(12), use_container_width=True)
            expected = ["age", "revenu_annuel", "statut_logement", "anciennete_emploi",
                        "motif_pret", "niveau_de_risque", "montant_pret", "taux_interet",
                        "ratio_pret_revenu", "defaut_historique_credit", "duree_historique_credit"]
            missing = [c for c in expected if c not in df.columns]
            if missing:
                st.error("Colonnes manquantes : " + ", ".join(missing))
            elif st.button("⚡  Lancer les prédictions du portefeuille"):
                outputs = []
                progress = st.progress(0)
                for i, (_, row) in enumerate(df.iterrows()):
                    p = {c: row[c] for c in expected}
                    for c in ["age", "duree_historique_credit"]:
                        p[c] = int(p[c])
                    for c in ["revenu_annuel", "anciennete_emploi", "montant_pret", "taux_interet", "ratio_pret_revenu"]:
                        p[c] = float(p[c])
                    try:
                        res = predict(p)
                        outputs.append({
                            "prediction": res["prediction"],
                            "probabilite_defaut_pct": res["score_risque_pourcentage"],
                            "statut": "OK",
                        })
                    except Exception as exc:
                        outputs.append({"prediction": None, "probabilite_defaut_pct": None, "statut": str(exc)})
                    progress.progress((i + 1) / max(len(df), 1))
                final_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(outputs)], axis=1)
                st.success("Analyse terminée.")
                st.dataframe(final_df, use_container_width=True)
                st.download_button("⬇ Télécharger les résultats CSV", final_df.to_csv(index=False).encode("utf-8"),
                                   "credit_risk_results.csv", "text/csv")
        except Exception as exc:
            st.error(f"Erreur de lecture du fichier : {exc}")


# =========================================================
# HISTORY
# =========================================================
elif page == "history":
    st.markdown('<div class="page-title">Historique des prédictions</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Résultats calculés pendant la session en cours.</div>', unsafe_allow_html=True)
    hist = st.session_state.prediction_history
    if hist:
        hist_df = pd.DataFrame(hist)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.download_button("⬇ Exporter l’historique CSV", hist_df.to_csv(index=False).encode("utf-8"),
                           "credit_risk_history.csv", "text/csv")
        if st.button("Effacer l’historique de cette session"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("Aucune prédiction enregistrée dans cette session. Effectuez une prédiction pour commencer.")


# =========================================================
# SETTINGS
# =========================================================
else:
    st.markdown('<div class="page-title">Paramètres</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Informations de configuration de la démonstration.</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Connexion au moteur de prédiction</div>', unsafe_allow_html=True)
    st.code(f"API_URL = {API_URL}")
    st.write("Statut :", "Connecté" if online else "Non connecté")
    st.markdown('</div>', unsafe_allow_html=True)
    st.warning("Les seuils faible/modéré/élevé affichés sont des seuils de présentation provisoires. Ils doivent être calibrés et validés sur un jeu de validation.")
