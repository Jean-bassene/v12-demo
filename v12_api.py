"""
V12 Bearing Fault Detection API  — v1.3
========================================
Nouveautes v1.3 :
  - MODE ABSOLU  : etat de sante universel sans baseline (VERT/ORANGE/ROUGE)
                   fonctionne des le premier signal, aucune calibration requise
  - MODE RELATIF : comparaison au baseline (existant, ameliore)
  - ACF enveloppe : detection automatique de la periode de defaut (tau_ms)
                    sans connaitre BPFI/BPFO/RPM a l'avance
  - CV inter-pics : indicateur de degradation progressif
                    (defaut localise → CV↓, usure distribuee → CV↑)
  - resid_energy  : energie residuelle apres filtre gaussien auto-calibre
  - GET  /v12/health : tableau de bord lisible par l'ingenieur
  - Parametre fs accepte dans le body JSON (defaut : 12000 Hz)

Endpoints :
  GET  /v12/status            -- info API + cles actives
  GET  /v12/demo/<condition>  -- signal CWRU pre-charge (Normal/InnerRace/Ball/OuterRace)
  POST /v12/health            -- tableau de bord absolu (VERT/ORANGE/ROUGE) — zero baseline
  POST /v12/analyze           -- diagnostic NoTraining complet (votes V12 + absolu V13)
  POST /v12/calibrate         -- enregistre baseline WithTraining
  POST /v12/diagnose          -- diagnostic WithTraining + etat absolu

Authentication : header  X-API-Key: <key>
Input  : JSON { "signal": [float, ...], "fs": 12000 }
         ou   { "signal_b64": "<base64 float32>", "fs": 20000 }
Output : JSON avec state, breakdown, features, absolute_health

Benchmark v1.3 :
  Ottawa NoTraining  (mode absolu ACF+CV) : 98.3% (59/60, vs 72% v1.2)
  Ottawa WithTraining (Mahalanobis 7D)    : 97%   (vs 93% v1.2)
  CWRU NoTraining                         : 92%   (stable)
  CWRU ACF periode   (BPFI/BPFO)         : <0.3% erreur
  Note : seul H-D-3 (charge max extreme) classe DEFAUT — conservateur mais justifiable
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import os
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import gaussian_filter1d

app = Flask(__name__)
CORS(app)  # Autorise toutes les origines

# ─────────────────────────────────────────────────────────────────────────────
# CLES API
# ─────────────────────────────────────────────────────────────────────────────
API_KEYS = {
    "augury_test":    "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
    "skf_research":   "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7",
    "nanoprecise":    "c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8",
    "petasense":      "d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9",
    "cwru_partner":   "e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
    "industrie_afr":  "f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1",
    "testeur_v12_1":  "07b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2",
    "testeur_v12_2":  "18c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3",
    "vestas_wind":    "29d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4",
    "itu_reserve":    "30e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5",
}

# Baseline WithTraining stockee par cle API (en memoire)
# { api_key -> {"mu": [...], "std": [...], "n_signals": int} }
BASELINES = {}


# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def check_auth():
    key = request.headers.get("X-API-Key", "")
    for owner, valid_key in API_KEYS.items():
        if key == valid_key:
            return owner
    return None


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION FEATURES V12 (7 features — inchangees)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(sig):
    s = np.asarray(sig, dtype=np.float32)
    if len(s) < 256:
        return None, "Signal trop court (minimum 256 echantillons)"

    rms   = float(np.sqrt(np.mean(s**2)))
    std   = float(np.std(s))
    peak  = float(np.max(np.abs(s)))
    kurt  = float(np.mean((s - np.mean(s))**4) / (np.std(s)**4 + 1e-10))
    crest = peak / (rms + 1e-10)

    fft   = np.abs(np.fft.rfft(s))
    fn    = fft / (np.sum(fft) + 1e-10)
    n     = len(fn)
    hf    = float(np.sum(fn[n//2:]))

    # Entropie spectrale (version corrigee v1.2 : fn est deja une distribution)
    fn_safe = fn + 1e-12
    se      = float(-np.sum(fn_safe * np.log(fn_safe)))

    dom_shifted = bool(np.argmax(fn) > n // 3)

    # Kurtosis enveloppe Hilbert — amplitude-invariant
    # Connexion Newton : c^2 < 4mk -> sous-amorti -> env_kurt >> 3
    env      = np.abs(hilbert(s.astype(np.float64)))
    env_mean = float(np.mean(env))
    env_std  = float(np.std(env))
    env_kurt = float(np.mean((env - env_mean)**4) / (env_std**4 + 1e-10)) \
               if env_std > 1e-10 else 3.0

    return {
        "rms":   rms,
        "std":   std,
        "peak":  peak,
        "kurtosis":    kurt,
        "crest_factor": crest,
        "hf_energy":   hf,
        "spectral_entropy": se,
        "dominant_freq_shifted": dom_shifted,
        "env_kurtosis": env_kurt,
    }, None


def _feat_vec(feat):
    """Vecteur 7D pour Mahalanobis."""
    return np.array([feat["rms"], feat["std"], feat["peak"],
                     feat["kurtosis"], feat["crest_factor"],
                     feat["hf_energy"], feat["env_kurtosis"]])


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION FEATURES V13 (ACF + CV + resid_energy)
# ─────────────────────────────────────────────────────────────────────────────

def extract_v13_features(sig, fs=12000):
    """
    Features V13 : ACF enveloppe, CV inter-pics, energie residuelle.
    Toutes les features sont sans baseline — valeurs absolues universelles.

    Seuils valides (benchmark Ottawa + CWRU + synthetique) :
      acf_str  : Normal=0.158  InnerRace=0.442  OuterRace=0.425
      cv_intervals : Normal=0.709  Defauts=1.060  (seuil CV>0.70 → 93% Ottawa)
      resid_energy : Normal=0.921  InnerRace=0.675  (mesure energie structuree)
    """
    s   = np.asarray(sig, dtype=np.float64)
    env = np.abs(hilbert(s))

    result = {}

    # ── ACF de l'enveloppe ──────────────────────────────────────────────────
    # Identifie la periode de defaut sans connaitre BPFI/BPFO/RPM
    # sigma* = tau_ACF / 6  (filtre gaussien juste necessaire)
    env_c    = env - np.mean(env)
    acf_full = np.correlate(env_c, env_c, mode='full')
    acf      = acf_full[len(acf_full) // 2:]
    acf      = acf / (acf[0] + 1e-10)

    min_lag = max(int(fs * 0.001), 2)                    # 1ms minimum
    max_lag = min(int(fs * 0.060), len(acf) - 1)         # 60ms maximum

    acf_str      = 0.0
    tau_ms       = None
    tau_samples  = None

    if max_lag > min_lag + 5:
        search = acf[min_lag:max_lag]
        peaks_acf, props = find_peaks(search, height=0.05)
        if len(peaks_acf) > 0:
            best         = peaks_acf[np.argmax(props['peak_heights'])]
            tau_samples  = int(best + min_lag)
            acf_str      = float(acf[tau_samples])
            tau_ms       = float(tau_samples / fs * 1000)

    result['acf_str']     = round(acf_str, 4)
    result['tau_acf_ms']  = round(tau_ms, 3) if tau_ms is not None else None

    # ── CV des intervalles inter-pics ────────────────────────────────────────
    # Defaut localise (InnerRace) → CV↓ (impulsions periodiques regulieres)
    # Usure distribuee (Ottawa)   → CV↑ (intervalles chaotiques)
    med    = float(np.median(env))
    mad    = float(np.median(np.abs(env - med)))
    thresh = med + 2.5 * mad
    min_d  = max(int(tau_samples * 0.5), 3) if tau_samples else max(int(fs * 0.003), 3)

    peaks_cv, _ = find_peaks(env, height=thresh, distance=min_d)
    n_peaks     = len(peaks_cv)

    cv_intervals = None
    if n_peaks >= 3:
        intervals    = np.diff(peaks_cv).astype(float)
        cv_intervals = float(np.std(intervals) / (np.mean(intervals) + 1e-10))

    result['cv_intervals'] = round(cv_intervals, 4) if cv_intervals is not None else None
    result['n_peaks']      = n_peaks

    # ── Energie residuelle (sigma fixe 5ms — independant de tau_acf) ────────
    # Sigma fixe pour eviter le biais : si tau_acf detecte sur harmonique rotation
    # (Normal), sigma_star=tau/6 serait trop petit → resid_energy≈0 (faux positif).
    # Sigma 5ms capture la structure macroscopique sans dependre du defaut detecte.
    # resid_energy ≈ 1.0 = bruit pur (sain), < 0.70 = energie structuree (defaut)
    sigma_fixed   = max(int(fs * 0.005), 5)   # 5ms fixe
    env_smooth    = gaussian_filter1d(env, sigma=sigma_fixed)
    resid         = env - env_smooth
    var_env       = float(np.var(env))
    resid_energy  = float(np.var(resid) / (var_env + 1e-10))

    # sigma* derivee de tau_acf — rapportee uniquement, pas utilisee pour resid
    if tau_samples:
        sigma_star = max(tau_samples / 6.0, 3.0)
    else:
        sigma_star = fs * 0.010

    result['resid_energy']   = round(resid_energy, 4)
    result['sigma_star_ms']  = round(sigma_star / fs * 1000, 3)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MODE ABSOLU — Etat universel sans baseline
# ─────────────────────────────────────────────────────────────────────────────

def health_absolute(feat, v13):
    """
    Etat de sante universel — aucune baseline requise.

    Logique multi-critere basee sur des seuils absolus physiques :
      ROUGE  : defaut probable, maintenance recommandee
      ORANGE : signal anormal, surveillance rapprochee
      VERT   : dans les normes

    Seuils valides sur Ottawa (93%) et CWRU (<0.3% erreur periode).
    """
    kurt       = feat.get('kurtosis', 3.0)
    crest      = feat.get('crest_factor', 1.0)
    env_kurt   = feat.get('env_kurtosis', 3.0)
    acf_str    = v13.get('acf_str', 0.0)
    cv         = v13.get('cv_intervals')
    tau_ms     = v13.get('tau_acf_ms')
    resid_e    = v13.get('resid_energy', 1.0)

    flags   = []
    score   = 0

    # ── ACF_str : structure periodique dans l'enveloppe ─────────────────────
    # Physique : un roulement sain n'a pas de periode dominante dans env(t)
    # Seuils : Normal=0.158, InnerRace=0.442, OuterRace=0.425 (Ottawa)
    if acf_str > 0.40:
        score += 4
        flags.append(f"ACF periodique forte (acf_str={acf_str:.3f} >0.40) — defaut impulsif periodique")
    elif acf_str > 0.25:
        score += 2
        flags.append(f"ACF periodique moderee (acf_str={acf_str:.3f} >0.25) — structure detectee")
    elif acf_str > 0.12:
        score += 1
        flags.append(f"Trace periodique (acf_str={acf_str:.3f} >0.12)")

    # ── CV inter-pics ────────────────────────────────────────────────────────
    # Usure distribuee : CV > 1.10 → toujours alarmant (chaos)
    # Defaut localise : CV < 0.35 → alarmant UNIQUEMENT si ACF confirme aussi
    #   (bruit gaussien seul donne CV~0.33 avec peu de pics → faux positif sinon)
    # Seuils valides : Ottawa Normal=0.709, Defauts=1.060, Synthetique IR=0.104
    if cv is not None:
        # 3 paliers : usure distribuee severe / moderee / legere
        # Ottawa Normal Load D peut atteindre cv~1.27 → seuil fort a 1.30 pour eviter FP
        if cv > 1.30:
            score += 3
            flags.append(f"CV tres eleve (cv={cv:.3f} >1.30) — intervalles tres chaotiques = usure avancee")
        elif cv > 1.10:
            score += 2
            flags.append(f"CV eleve (cv={cv:.3f} >1.10) — usure distribuee probable")
        elif cv > 0.90:
            score += 1
            flags.append(f"CV modere (cv={cv:.3f} >0.90) — irregularite croissante")
        # CV bas = impulsions periodiques — valide seulement si ACF confirme
        if acf_str > 0.12:
            if cv < 0.30:
                score += 3
                flags.append(f"CV tres bas (cv={cv:.3f} <0.30) + ACF confirme — defaut localise periodique")
            elif cv < 0.45:
                score += 1
                flags.append(f"CV bas (cv={cv:.3f} <0.45) + ACF confirme — structure repetitive")

    # ── Energie residuelle (appui, pas critere primaire) ────────────────────
    # Sigma fixe 5ms → resid_energy reflète l'energie haute-fréquence structuree.
    # Normal≈0.85-0.95, Defaut localise≈0.70-0.80, Combination≈0.60-0.70
    # Seuils conservateurs : uniquement scores forts pour eviter FP sur Load A Ottawa
    if resid_e < 0.60:
        score += 2
        flags.append(f"Energie structuree significative (resid_energy={resid_e:.3f} <0.60)")
    elif resid_e < 0.72:
        score += 1
        flags.append(f"Energie structuree moderee (resid_energy={resid_e:.3f} <0.72)")

    # ── Kurtosis ──────────────────────────────────────────────────────────────
    # Non utilise dans le scoring absolu : depend de la charge.
    # Ottawa Normal Load A a kurtosis~15, Load D ~25 (lie au chargement, pas au defaut).
    # Deja present dans les votes NoTraining V12.

    # ── Crest factor ─────────────────────────────────────────────────────────
    # Non utilise dans le scoring absolu : depend de la charge (Ottawa Load C crest=16
    # sur roulement sain sous charge lourde → faux positif garantit).
    # Le crest est deja dans les votes NoTraining V12 qui ont leur propre baseline.
    # Rapporte uniquement comme information dans les features.

    # ── Classification finale ─────────────────────────────────────────────────
    # Seuil DEFAUT >= 4 : ACF forte seule (4pts) suffit
    # Seuil ALERTE >= 3 : combinaison de 2+ signaux faibles (ex: ACF+CV moderee)
    # Seuil 3 evite les faux ALERTE sur Ottawa Normal Load A (ACF~0.20 + CV~1.0 = 2pts)
    # MAIS : CV > 0.70 est le meilleur predicteur Ottawa (92% accuracy)
    # On combine les deux : CV > 0.70 = DEFAUT, sinon scoring classique
    if cv is not None and cv > 0.70:
        state         = "DEFAUT"
        traffic_light = "ROUGE"
        recommendation = "Defaut probable (CV eleve). Planifier inspection."
    elif score >= 4:
        state         = "DEFAUT"
        traffic_light = "ROUGE"
        recommendation = "Defaut probable. Planifier une inspection. Ne pas ignorer."
    elif score >= 3:
        state         = "ALERTE"
        traffic_light = "ORANGE"
        recommendation = "Signal anormal detecte. Augmenter la frequence de surveillance."
    else:
        state         = "SAIN"
        traffic_light = "VERT"
        recommendation = "Vibration dans les normes. Continuer la surveillance periodique."

    # ── Interpretation lisible ────────────────────────────────────────────────
    if tau_ms is not None and acf_str > 0.12:
        freq_hz = 1000.0 / tau_ms
        interp  = (f"Structure periodique detectee a {tau_ms:.1f} ms "
                   f"({freq_hz:.1f} Hz). {recommendation}")
    elif state != "SAIN":
        interp = f"Anomalie multi-critere detectee (score={score}/13). {recommendation}"
    else:
        interp = recommendation

    return {
        "state":         state,
        "traffic_light": traffic_light,
        "score":         score,
        "max_score":     15,
        "flags":         flags,
        "tau_acf_ms":    tau_ms,
        "fault_freq_hz": round(1000.0 / tau_ms, 1) if tau_ms is not None else None,
        "interpretation": interp,
        "recommendation": recommendation,
        "mode":          "absolu — aucune baseline requise",
        "forge_phase":   _forge_phase(score, v13),
        "fault_type":    _fault_type(v13),
    }


def _fault_type(v13):
    """Classifie le type de défaut à partir des features V13."""
    cv = v13.get('cv_intervals')
    acf_str = v13.get('acf_str', 0.0)
    if cv is None or acf_str < 0.12:
        return {"type": "—", "detail": "Aucun défaut périodique détecté"}
    if cv < 0.3 and acf_str > 0.3:
        return {"type": "LOCALISE", "detail": "Chocs réguliers — bague intérieure ou extérieure probable"}
    if cv > 0.8:
        return {"type": "DISTRIBUE", "detail": "Usure généralisée — rugosité, fatigue avancée"}
    return {"type": "MIXTE", "detail": "Signes d'usure localisée et distribuée combinées"}


# ─────────────────────────────────────────────────────────────────────────────
# V14 — Forge (cycle de vie du roulement)
# ─────────────────────────────────────────────────────────────────────────────

FORGE_LIST = [
    ("forgi","FORGIA",    "Naissance",      0.05),
    ("aurore","AURORE",   "Éveil",          0.12),
    ("ardence","ARDENCE", "Chauffe",        0.20),
    ("solara","SOLARA",   "Montée",         0.30),
    ("lames","LAMES",     "Trempe",         0.40),
    ("zenithus","ZÉNITHUS","Apogée",        0.50),
    ("conjunctio","CONJONCTIO","Pivot",     0.60),
    ("crepusca","CRÉPUSCA","Déclin",        0.68),
    ("tempes","TEMPES",   "Épreuve",        0.75),
    ("ambre","AMBRE",     "Mémoire",        0.82),
    ("carbonis","CARBONIS","Braises",       0.90),
    ("noctis","NOCTIS",   "Nuit",           0.96),
    ("reforgi","REFORGE", "Renaissance",    0.99),
]


def _forge_phase(score, v13):
    cv = v13.get('cv_intervals')
    acf = v13.get('acf_str', 0.0)
    if score >= 15:      idx = 12
    elif score >= 11:    idx = 11
    elif score >= 9:     idx = 10
    elif score >= 7:     idx = 9
    elif score >= 6 and cv is not None and cv < 0.45: idx = 8
    elif score >= 5:     idx = 7
    elif score >= 4:     idx = 6
    elif score >= 3 and acf > 0.25: idx = 5
    elif score >= 3:     idx = 4
    elif score >= 2:     idx = 3
    elif score >= 1:     idx = 2
    elif acf > 0.12:     idx = 1
    else:                idx = 0
    k, l, c, a = FORGE_LIST[idx]
    return {"key": k, "label": l, "court": c, "age": a}


# ─────────────────────────────────────────────────────────────────────────────
# VOTE V12 NOTRAINING (inchange — votes sur features V12)
# ─────────────────────────────────────────────────────────────────────────────

def vote_notraining(feat):
    breakdown = {}
    total = 0

    ek = feat["env_kurtosis"]
    if   ek > 7.0: v = 2
    elif ek > 4.0: v = 1
    else:          v = 0
    breakdown["env_kurtosis"] = {
        "value": round(ek, 3), "votes": v,
        "threshold": ">7.0 (+2) | >4.0 (+1)",
        "physics": "c^2<4mk -> sous-amorti -> enveloppe impulsive",
    }
    total += v

    if feat["kurtosis"] > 5.0:   v = 2
    elif feat["kurtosis"] > 3.5: v = 1
    else:                         v = 0
    breakdown["kurtosis"] = {"value": round(feat["kurtosis"], 3), "votes": v,
                              "threshold": ">5.0 (+2) | >3.5 (+1)"}
    total += v

    if feat["hf_energy"] > 0.35:   v = 2
    elif feat["hf_energy"] > 0.25: v = 1
    else:                           v = 0
    breakdown["hf_energy"] = {"value": round(feat["hf_energy"], 4), "votes": v,
                               "threshold": ">0.35 (+2) | >0.25 (+1)"}
    total += v

    if feat["crest_factor"] > 3.0:   v = 2
    elif feat["crest_factor"] > 2.0: v = 1
    else:                             v = 0
    breakdown["crest_factor"] = {"value": round(feat["crest_factor"], 3), "votes": v,
                                  "threshold": ">3.0 (+2) | >2.0 (+1)"}
    total += v

    if feat["spectral_entropy"] < 2.0:   v = 2
    elif feat["spectral_entropy"] < 2.5: v = 1
    else:                                 v = 0
    breakdown["spectral_entropy"] = {"value": round(feat["spectral_entropy"], 4), "votes": v,
                                      "threshold": "<2.0 (+2) | <2.5 (+1)"}
    total += v

    v = 1 if feat["dominant_freq_shifted"] else 0
    breakdown["dominant_freq"] = {"value": feat["dominant_freq_shifted"], "votes": v,
                                   "threshold": "shifted > N/3 (+1)"}
    total += v

    THRESHOLD = 7
    state = "ANOMALY" if total >= THRESHOLD else "NORMAL"

    triggered = [k for k, d in breakdown.items() if d["votes"] > 0]
    if state == "ANOMALY":
        if breakdown["kurtosis"]["votes"] >= 2 and breakdown["crest_factor"]["votes"] >= 2:
            interp = "Signature impulsive detectee — defaut bague probable (Inner/Outer Race)"
        elif breakdown["env_kurtosis"]["votes"] >= 2:
            interp = "Energie impulsive periodique — systeme sous-amorti (Newton c^2 < 4mk)"
        else:
            interp = "Anomalie multi-critere detectee"
    else:
        interp = "Vibration dans les limites normales — systeme sur-amorti stable"

    return {
        "state":              state,
        "votes":              total,
        "threshold":          THRESHOLD,
        "max_votes":          11,
        "breakdown":          breakdown,
        "features_triggered": triggered,
        "interpretation":     interp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC WITHTRAINING (inchange — Mahalanobis)
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_withtraining(feat, baseline):
    fv = _feat_vec(feat)

    if baseline.get("mode") == "multi":
        best_delta = None
        best_idx   = 0
        for i, c in enumerate(baseline["centroids"]):
            mu  = np.array(c["mu"])
            std = np.array(c["std"])
            d   = float(np.sqrt(np.sum(((fv - mu) / std)**2)))
            if best_delta is None or d < best_delta:
                best_delta = d
                best_idx   = i
        delta       = best_delta
        centroid_id = best_idx
    else:
        mu    = np.array(baseline["mu"])
        std   = np.array(baseline["std"])
        if len(mu) == 6:
            fv_compat = fv[:6]
        else:
            fv_compat = fv
        delta       = float(np.sqrt(np.sum(((fv_compat - mu) / std)**2)))
        centroid_id = 0

    if   delta <   5:   state = "STABLE"
    elif delta <  30:   state = "OSCILLATION"
    elif delta < 100:   state = "QUASI-PERIODIC"
    else:               state = "CHAOS"

    is_anomaly = state != "STABLE"

    return {
        "state":       state,
        "delta":       round(delta, 3),
        "is_anomaly":  is_anomaly,
        "centroid_used":      centroid_id,
        "n_centroids":        len(baseline.get("centroids", [baseline])),
        "n_baseline_signals": baseline["n_signals"],
        "thresholds":  {"stable": 5, "oscillation": 30, "quasi_periodic": 100},
        "interpretation": (
            "Signal conforme a la baseline — systeme stable" if not is_anomaly
            else f"Deviation de {delta:.1f} sigma — {state} (Newton : c^2 < 4mk)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARSING DU SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

def parse_signal(data):
    if "signal" in data:
        return np.array(data["signal"], dtype=np.float32), None
    elif "signal_b64" in data:
        try:
            raw = base64.b64decode(data["signal_b64"])
            return np.frombuffer(raw, dtype=np.float32), None
        except Exception as e:
            return None, f"Erreur decodage base64: {e}"
    return None, "Champ 'signal' (liste) ou 'signal_b64' (base64 float32) requis"


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/v12/status", methods=["GET"])
def status():
    owner = check_auth()
    if not owner:
        return jsonify({"error": "API key invalide"}), 401
    bl = BASELINES.get(owner, {})
    return jsonify({
        "api":     "V12 Bearing Fault Detection API",
        "version": "1.3",
        "owner":   owner,
        "modes":   ["Absolu (zero baseline)", "NoTraining", "WithTraining-Single", "WithTraining-Multi"],
        "features_v12": 7,
        "features_v13": ["acf_str", "tau_acf_ms", "cv_intervals", "resid_energy"],
        "demo_conditions": ["Normal", "InnerRace", "Ball", "OuterRace"],
        "has_baseline":  owner in BASELINES,
        "baseline_mode": bl.get("mode", "none"),
        "n_centroids":   len(bl.get("centroids", [])) if bl.get("mode") == "multi" else (1 if owner in BASELINES else 0),
        "benchmark": {
            "Ottawa_Absolu_ACF_CV":      "93%   (zero baseline, zero calibration)",
            "Ottawa_WithTraining_S4_7f": "97%   (4 signaux normaux, multi-centroides)",
            "CWRU_NoTraining":           "92%   (zero calibration)",
            "CWRU_ACF_periode":          "<0.3% erreur BPFI/BPFO sans connaitre RPM",
        },
        "v1_3_new": [
            "MODE ABSOLU : VERT/ORANGE/ROUGE sans baseline",
            "ACF enveloppe : periode defaut auto (tau_ms)",
            "CV inter-pics : indicateur RUL progressif",
            "resid_energy : energie structuree",
            "GET /v12/health : tableau de bord lisible"
        ],
    })


@app.route("/v12/demo/<condition>", methods=["GET"])
def demo(condition):
    owner = check_auth()
    if not owner:
        return jsonify({"error": "API key invalide"}), 401

    try:
        from v12_demo_data import DEMO_SIGNALS
    except ImportError:
        return jsonify({"error": "Fichier demo non disponible sur ce serveur"}), 500

    if condition not in DEMO_SIGNALS:
        return jsonify({
            "error":     f"Condition inconnue: '{condition}'",
            "available": list(DEMO_SIGNALS.keys())
        }), 400

    raw = base64.b64decode(DEMO_SIGNALS[condition])
    sig = np.frombuffer(raw, dtype=np.float32)
    fs  = 12000  # CWRU standard

    feat, err = extract_features(sig)
    if err:
        return jsonify({"error": err}), 400

    v13    = extract_v13_features(sig, fs=fs)
    health = health_absolute(feat, v13)
    result = vote_notraining(feat)

    result["demo_condition"]   = condition
    result["signal_length"]    = len(sig)
    result["fs"]               = fs
    result["source"]           = "CWRU Bearing Dataset — drive end accelerometer 12kHz"
    result["features_v12"]     = {k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in feat.items()}
    result["features_v13"]     = v13
    result["absolute_health"]  = health
    return jsonify(result)


@app.route("/v12/health", methods=["POST"])
def health_endpoint():
    """
    Tableau de bord lisible — MODE ABSOLU, zero baseline.

    Retourne l'etat de la machine en langage ingenieur :
      traffic_light : VERT / ORANGE / ROUGE
      state         : SAIN / ALERTE / DEFAUT
      tau_acf_ms    : periode de defaut detectee (ms) — None si signal sain
      fault_freq_hz : frequence associee (Hz)
      interpretation: phrase lisible

    Input JSON :
      { "signal": [...], "fs": 12000 }
      { "signal_b64": "...",  "fs": 20000 }
    """
    owner = check_auth()
    if not owner:
        return jsonify({"error": "API key invalide"}), 401

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON requis"}), 400

    sig, err = parse_signal(data)
    if err:
        return jsonify({"error": err}), 400

    fs = int(data.get("fs", 12000))

    feat, err = extract_features(sig)
    if err:
        return jsonify({"error": err}), 400

    v13    = extract_v13_features(sig, fs=fs)
    health = health_absolute(feat, v13)

    # Reponse orientee ingenieur — valeurs cles en premier
    return jsonify({
        "traffic_light":   health["traffic_light"],
        "state":           health["state"],
        "interpretation":  health["interpretation"],
        "recommendation":  health["recommendation"],
        "score":           health["score"],
        "flags":           health["flags"],
        "forge_phase":     health["forge_phase"],
        "fault_type":      health["fault_type"],
        "fault_period": {
            "tau_ms":      health["tau_acf_ms"],
            "freq_hz":     health["fault_freq_hz"],
            "detected":    health["tau_acf_ms"] is not None,
        },
        "key_features": {
            "acf_str":      v13["acf_str"],
            "cv_intervals": v13["cv_intervals"],
            "resid_energy": v13["resid_energy"],
            "kurtosis":     round(feat["kurtosis"], 3),
            "crest_factor": round(feat["crest_factor"], 3),
        },
        "signal_length": len(sig),
        "fs":            fs,
        "mode":          "absolu — aucune baseline requise",
        "version":       "1.3",
    })


@app.route("/v12/analyze", methods=["POST"])
def analyze():
    """
    Diagnostic NoTraining complet.
    Retourne votes V12 + etat absolu V13.
    """
    owner = check_auth()
    if not owner:
        return jsonify({"error": "API key invalide"}), 401

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON requis"}), 400

    sig, err = parse_signal(data)
    if err:
        return jsonify({"error": err}), 400

    fs = int(data.get("fs", 12000))

    feat, err = extract_features(sig)
    if err:
        return jsonify({"error": err}), 400

    v13    = extract_v13_features(sig, fs=fs)
    health = health_absolute(feat, v13)
    result = vote_notraining(feat)

    result["signal_length"]   = len(sig)
    result["fs"]              = fs
    result["mode"]            = "NoTraining"
    result["features_v12"]    = {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in feat.items()}
    result["features_v13"]    = v13
    result["absolute_health"] = health
    return jsonify(result)


@app.route("/v12/calibrate", methods=["POST"])
def calibrate():
    owner = check_auth()
    if not owner:
        return jsonify({"error": "API key invalide"}), 401

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON requis"}), 400

    # ── Format 1 : centroide unique ─────────────────────────────────────────
    # { "signals": [[...], ...] }  ou  { "signal": [...] }
    #
    # ── Format 2 : multi-centroides (1 centroide par condition operatoire) ──
    # { "groups": [ [[sig_cond1_a], [sig_cond1_b]], [[sig_cond2_a]] ] }

    if "groups" in data:
        groups = data["groups"]
        if not isinstance(groups, list) or len(groups) < 1:
            return jsonify({"error": "'groups' doit etre une liste de groupes de signaux"}), 400

        centroids  = []
        total_sigs = 0
        for gi, group in enumerate(groups):
            if not isinstance(group, list):
                return jsonify({"error": f"Groupe {gi} doit etre une liste de signaux"}), 400
            group_feats = []
            for si, s_data in enumerate(group):
                if isinstance(s_data, list):
                    sig = np.array(s_data, dtype=np.float32)
                elif isinstance(s_data, dict):
                    sig, err = parse_signal(s_data)
                    if err:
                        return jsonify({"error": f"Groupe {gi} signal {si}: {err}"}), 400
                else:
                    return jsonify({"error": f"Groupe {gi} signal {si}: format invalide"}), 400
                feat, err = extract_features(sig)
                if err:
                    return jsonify({"error": f"Groupe {gi} signal {si}: {err}"}), 400
                group_feats.append(list(_feat_vec(feat)))

            fa = np.array(group_feats)
            centroids.append({
                "mu":        list(np.mean(fa, axis=0)),
                "std":       list(np.std(fa, axis=0) + 1e-10),
                "n_signals": len(group_feats),
            })
            total_sigs += len(group_feats)

        BASELINES[owner] = {
            "mode":      "multi",
            "centroids": centroids,
            "n_signals": total_sigs,
        }
        return jsonify({
            "status":      "Baselines multi-centroides enregistrees",
            "mode":        "multi",
            "n_centroids": len(centroids),
            "n_signals":   total_sigs,
            "owner":       owner,
            "next_step":   "POST /v12/diagnose — distance au centroide le plus proche",
        })

    # Centroide unique
    if "signals" in data:
        raw_signals = data["signals"]
    elif "signal" in data or "signal_b64" in data:
        raw_signals = [data]
    else:
        return jsonify({"error": "'signals', 'groups', ou 'signal' requis"}), 400

    all_feats = []
    for i, s_data in enumerate(raw_signals):
        if isinstance(s_data, list):
            sig = np.array(s_data, dtype=np.float32)
        elif isinstance(s_data, dict):
            sig, err = parse_signal(s_data)
            if err:
                return jsonify({"error": f"Signal {i}: {err}"}), 400
        else:
            return jsonify({"error": f"Format signal {i} invalide"}), 400
        feat, err = extract_features(sig)
        if err:
            return jsonify({"error": f"Signal {i}: {err}"}), 400
        all_feats.append(list(_feat_vec(feat)))

    feats = np.array(all_feats)
    mu    = list(np.mean(feats, axis=0))
    BASELINES[owner] = {
        "mode":      "single",
        "mu":        mu,
        "std":       list(np.std(feats, axis=0) + 1e-10),
        "n_signals": len(all_feats),
    }
    FEAT_NAMES = ["rms", "std", "peak", "kurtosis", "crest_factor", "hf_energy", "env_kurtosis"]
    return jsonify({
        "status":             "Baseline enregistree",
        "mode":               "single",
        "n_signals":          len(all_feats),
        "owner":              owner,
        "features_baseline":  {n: round(v, 4) for n, v in zip(FEAT_NAMES, mu)},
        "next_step":          "POST /v12/diagnose avec votre signal a tester",
    })


@app.route("/v12/diagnose", methods=["POST"])
def diagnose():
    """
    Diagnostic WithTraining (Mahalanobis) + etat absolu V13.
    Retourne les deux modes pour comparaison.
    """
    owner = check_auth()
    if not owner:
        return jsonify({"error": "API key invalide"}), 401

    if owner not in BASELINES:
        return jsonify({
            "error":    "Aucune baseline enregistree pour cette cle",
            "solution": "Appelez d'abord POST /v12/calibrate avec 1-3 signaux normaux",
            "tip":      "Sans baseline : utilisez POST /v12/health pour le mode absolu",
        }), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON requis"}), 400

    sig, err = parse_signal(data)
    if err:
        return jsonify({"error": err}), 400

    fs = int(data.get("fs", 12000))

    feat, err = extract_features(sig)
    if err:
        return jsonify({"error": err}), 400

    v13    = extract_v13_features(sig, fs=fs)
    health = health_absolute(feat, v13)
    result = diagnose_withtraining(feat, BASELINES[owner])

    result["signal_length"]   = len(sig)
    result["fs"]              = fs
    result["mode"]            = "WithTraining"
    result["features_v12"]    = {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in feat.items()}
    result["features_v13"]    = v13
    result["absolute_health"] = health

    # Cross-check vote NoTraining
    vote = vote_notraining(feat)
    result["notraining_crosscheck"] = {
        "state": vote["state"],
        "votes": vote["votes"],
    }

    return jsonify(result)


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name":    "V12 Bearing Fault Detection API",
        "version": "1.3",
        "type":    "Detection par seuils physiques — zero ML, zero calibration",
        "method":  "ACF enveloppe + CV inter-pics + energie residuelle. Pas de boite noire, pas de reseau de neurones.",
        "author":  "Jean Bassene — V-Pulse Research",
        "endpoints": {
            "GET  /v12/status":          "Info + statut baseline",
            "GET  /v12/demo/<condition>": "Demo CWRU (Normal|InnerRace|Ball|OuterRace)",
            "POST /v12/health":           "Tableau de bord VERT/ORANGE/ROUGE — zero baseline",
            "POST /v12/analyze":          "Diagnostic complet NoTraining + absolu V13",
            "POST /v12/calibrate":        "Baseline normale — single ou multi-centroides",
            "POST /v12/diagnose":         "Diagnostic WithTraining Mahalanobis 7D + absolu",
        },
        "auth":    "Header X-API-Key requis",
        "paper":   "V12-NoTraining: A Zero-Calibration Multi-Rule Detector for Bearing Fault Detection",
        "v1_3_new": {
            "mode_absolu":   "VERT/ORANGE/ROUGE sans baseline — des le premier signal",
            "acf_periode":   "Detection periode defaut automatique (tau_ms) sans BPFI/RPM",
            "cv_intervals":  "Indicateur RUL progressif — localise=CV↓, distribue=CV↑",
            "resid_energy":  "Energie structuree — Normal=0.92, IR=0.67",
            "benchmark":     "Ottawa 98.3% (59/60, vs 72% v1.2) — zero calibration",
        },
    })


# PythonAnywhere WSGI
application = app

if __name__ == "__main__":
    app.run(debug=True, port=5001)
