#!/usr/bin/env python3
"""V6 Medical API — Compression ECG adaptative."""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import time
import zlib, struct
from scipy.signal import find_peaks

app = Flask(__name__)
CORS(app)

# ── Codec V6 v3 (intégré) ───────────────────────────────────────────────

class V6MedicalCodec:
    def __init__(self, sigmas=(3.0, 7.0), dead_zone=0.1, res_scale=500.0):
        self.sigmas = list(sigmas); self.dead_zone = dead_zone; self._res_scale = res_scale

    def _encode_gaussians(self, signal):
        N, t = len(signal), np.arange(len(signal), dtype=np.float32)
        rem = signal.copy().astype(np.float32); all_nodes = []
        for s in self.sigmas:
            peaks, _ = find_peaks(np.abs(rem), height=np.percentile(np.abs(rem), 95))
            if len(peaks) == 0: continue
            idx = peaks.astype(np.int32); val = rem[idx].astype(np.float32)
            keep = np.abs(val) > self.dead_zone; idx, val = idx[keep], val[keep]
            if len(idx) == 0: continue
            for i, v in zip(idx, val): all_nodes.append((float(s), int(i), float(v)))
            diff = (t[:, None] - idx[None, :].astype(np.float32)) / s
            rem -= (np.exp(-0.5*diff**2) * val[None, :]).sum(axis=1)
        return all_nodes, rem

    def _decode_gaussians(self, nodes, N):
        recon = np.zeros(N, dtype=np.float32)
        if not nodes: return recon
        t = np.arange(N, dtype=np.float32)
        for s in set(n[0] for n in nodes):
            grp = [(i,v) for (ss,i,v) in nodes if ss==s]
            idx = np.array([x[0] for x in grp], dtype=np.float32)
            vals = np.array([x[1] for x in grp], dtype=np.float32)
            recon += (np.exp(-0.5*((t[:,None]-idx[None,:])/s)**2) * vals[None,:]).sum(axis=1)
        return recon

    def encode(self, signal):
        signal = np.asarray(signal, dtype=np.float32); N = len(signal)
        mu = float(np.mean(signal)); sigma = float(np.std(signal)) + 1e-9
        nodes, residual = self._encode_gaussians((signal - mu) / sigma)
        res_q = np.clip(np.round(residual * self._res_scale), -32768, 32767).astype(np.int16)
        res_diff = np.diff(res_q, prepend=np.int16(0)).astype(np.int16)
        buf = bytearray(); buf += struct.pack('<I', len(nodes))
        for s, idx, val in nodes: buf += struct.pack('<fif', s, idx, val)
        bin_nodes = zlib.compress(bytes(buf), level=9)
        bin_res = zlib.compress(res_diff.tobytes(), level=9)
        header = struct.pack('<IffII', N, mu, sigma, len(bin_nodes), len(bin_res))
        return header + bin_nodes + bin_res

    def decode(self, data):
        N, mu, sigma, sz_n, sz_r = struct.unpack('<IffII', data[:20]); off = 20
        raw = zlib.decompress(data[off:off+sz_n]); off += sz_n
        nodes = []
        n_nodes = struct.unpack('<I', raw[:4])[0]
        for i in range(n_nodes):
            s, idx, val = struct.unpack('<fif', raw[4+i*12:4+(i+1)*12])
            nodes.append((s, idx, val))
        res_diff = np.frombuffer(zlib.decompress(data[off:off+sz_r]), dtype=np.int16)
        res_q = np.cumsum(res_diff, dtype=np.int16).astype(np.float32)
        return (self._decode_gaussians(nodes, N) + (res_q / self._res_scale)) * sigma + mu

# ── Profils ──────────────────────────────────────────────────────────────

PROFILES = {
    'PATHOLOGICAL': V6MedicalCodec(sigmas=[1.0, 3.0], dead_zone=0.02, res_scale=2000.0),
    'QUALITY':      V6MedicalCodec(sigmas=[2.0, 4.0], dead_zone=0.05, res_scale=1000.0),
    'BALANCED':     V6MedicalCodec(sigmas=[2.0, 5.0], dead_zone=0.10, res_scale=500.0),
    'COMPACT':      V6MedicalCodec(sigmas=[3.0, 7.0], dead_zone=0.30, res_scale=200.0),
    'ULTRA':        V6MedicalCodec(sigmas=[5.0, 10.0], dead_zone=0.50, res_scale=50.0),
}

# ── Signaux démo intégrés ────────────────────────────────────────────────

def gen_ecg(rec_name, n=10800):
    """Génère un signal ECG synthétique réaliste."""
    t = np.linspace(0, n/360, n).astype(np.float32)
    if rec_name == '100':  # Normal
        hr = 1.2
        sig = np.sin(2*np.pi*hr*t)**21
        sig += np.random.randn(n).astype(np.float32) * 0.02
    elif rec_name == '200':  # Arythmie
        hr = 0.8 + 0.4*np.sin(2*np.pi*0.05*t)
        sig = np.sin(2*np.pi*hr*t)**21
        # PVCs
        for i in range(0, n, int(360*3)):
            sig[i:i+30] += np.random.randn(min(30,n-i)).astype(np.float32) * 0.5
        sig += np.random.randn(n).astype(np.float32) * 0.03
    elif rec_name == '207':  # Flutter
        sig = np.sin(2*np.pi*3.0*t)**21 + np.sin(2*np.pi*6.0*t)*0.3
        sig += np.random.randn(n).astype(np.float32) * 0.02
    else:
        sig = np.random.randn(n).astype(np.float32) * 0.1
    return sig * 0.8 + 0.2

def snr_db(orig, recon):
    n = orig.astype(np.float64) - recon.astype(np.float64)
    return float(10 * np.log10(np.sum(orig**2) / (np.sum(n**2) + 1e-15)))

def prd_pct(orig, recon):
    return float(100 * np.sqrt(np.sum((orig-recon)**2) / (np.sum(orig**2) + 1e-15)))

# ── Endpoints ────────────────────────────────────────────────────────────

@app.route('/v6/modes', methods=['GET'])
def modes():
    return jsonify({"modes": list(PROFILES.keys()) + ["ADAPTIVE"]})

@app.route('/v6/compress', methods=['POST'])
def compress():
    data = request.get_json(silent=True)
    if not data: return jsonify({"error": "JSON requis"}), 400

    rec = data.get('record', '100')
    mode = data.get('mode', 'BALANCED')
    sig_raw = data.get('signal')
    sig_b64 = data.get('signal_b64')

    if sig_raw is not None and isinstance(sig_raw, list):
        sig = np.array(sig_raw, dtype=np.float32)
    elif sig_b64:
        sig = np.frombuffer(base64.b64decode(sig_b64), dtype=np.float32)
    else:
        sig = gen_ecg(rec)

    if len(sig) < 100: return jsonify({"error": "Signal trop court"}), 400
    if len(sig) > 21600:
        step = len(sig) // 10800
        sig = sig[::step]

    # Mode ADAPTIVE : profilage rapide
    if mode == 'ADAPTIVE':
        s = sig - np.mean(sig)
        kurt = float(np.mean(s**4) / (np.std(s)**4 + 1e-10))
        thresh = np.mean(s) + 1.5 * np.std(s)
        peaks, _ = find_peaks(s, height=thresh, distance=int(360*0.2))
        cv = 0.0
        if len(peaks) >= 3:
            intervals = np.diff(peaks).astype(float)
            m = np.mean(intervals)
            if m > 0: cv = float(np.std(intervals) / m)
        if cv > 0.25: mode = 'PATHOLOGICAL'
        elif cv > 0.07 or kurt < 5.0: mode = 'QUALITY'
        elif kurt > 15.0 and cv < 0.05: mode = 'COMPACT'
        else: mode = 'BALANCED'

    if mode not in PROFILES:
        return jsonify({"error": f"Mode inconnu: {mode}", "modes": list(PROFILES.keys()) + ["ADAPTIVE"]}), 400

    codec = PROFILES[mode]
    t0 = time.perf_counter()
    comp = codec.encode(sig)
    enc_ms = (time.perf_counter() - t0) * 1000
    recon = codec.decode(comp)
    cr = (len(sig) * 4) / len(comp)

    # Réponse
    # Sous-échantillonnage pour l'affichage (max 500 points)
    step_disp = max(1, len(sig) // 500)
    return jsonify({
        "mode": mode,
        "record": rec,
        "snr": round(snr_db(sig, recon), 2),
        "prd": round(prd_pct(sig, recon), 4),
        "cr": round(cr, 2),
        "time_ms": round(enc_ms, 1),
        "original": [round(float(x), 6) for x in sig[::step_disp]],
        "reconstructed": [round(float(x), 6) for x in recon[::step_disp]],
        "n_samples": len(sig),
        "n_display": len(sig) // step_disp,
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "V6 Medical Codec API",
        "version": "3.0",
        "type": "Codec purement algorithmique — zero ML, zero calibration",
        "method": "Decomposition gaussienne vectorisee + quantification delta. Seuils physiques, pas de reseau de neurones.",
        "endpoints": {
            "GET  /v6/modes": "Liste des modes disponibles",
            "POST /v6/compress": "Compresser un signal ECG. Input: {signal:[...], mode:'BALANCED'} Output: {snr, cr, prd, original, reconstructed}"
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002)
