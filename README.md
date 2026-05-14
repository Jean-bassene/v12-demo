# V12/V14 — Bearing Fault Detection + V6 Medical Codec

## 🎯 Page de démo
https://jeanbassene.github.io/v12-demo/

Glissez un fichier `.csv` de vibrations → diagnostic instantané (feu + forge + type).

## 🔧 V12/V14 — Détection de défauts roulements

### Benchmarks
| Dataset | Méthode | Précision |
|---|---|---|
| **CWRU** | V12/V14 NoTraining | **92%** |
| Ottawa (Channel 2) | V14 (forge + type + CV) | 73% |
| Ottawa (Channel 1) | CV > 0.70 | **92%** |

### vs État de l'art (CWRU)
| Méthode | Accuracy | Calibration |
|---|---|---|
| CNN ResNet | 99.8% | Supervisé (1000s) |
| SVM | 100% | Supervisé |
| **V12/V14** | **92%** | **Zéro** |

## 💓 V6 Medical Codec v3 — Compression ECG

### Benchmarks (MIT-BIH : 100, 200, 207)
| Mode | SNR | PRD | CR | Temps |
|---|---|---|---|---|
| **PATHOLOGICAL** | **78.4 dB** | 0.01% | 2.86x | 152ms |
| QUALITY | 73.0 dB | 0.02% | 2.79x | 153ms |
| **ADAPTIVE** | **76.9 dB** | **0.01%** | **2.83x** | **153ms** |
| BALANCED | 66.7 dB | 0.05% | 2.81x | 156ms |
| COMPACT | 58.6 dB | 0.12% | 2.92x | 146ms |
| ULTRA | 47.8 dB | 0.41% | 3.39x | 127ms |

### vs État de l'art (compression ECG)
| Méthode | CR | SNR | Calibration | Temps |
|---|---|---|---|---|
| Deep ARIMA | **41.5x** | PRD 0.21% | Supervisé | GPU |
| ECGNet+Brotli | **82.4x** | PRD 2.70% | Supervisé | GPU |
| **V6 v3 ADAPTIVE** | **2.83x** | **76.9 dB** | **Aucune** | **CPU 153ms** |

### Gain V2 → V3
| Métrique | V2 | V3 | Gain |
|---|---|---|---|
| BALANCED SNR | 67.4 dB | 66.7 dB | — |
| BALANCED CR | 2.78x | 2.81x | +1% |
| Temps encodage | 317 ms | **156 ms** | **−51%** |
| ADAPTIVE | — | **76.9 dB** | Nouveau |

## 🚀 Déploiement
- **API** : https://v12-demo.onrender.com
- **Page** : https://jeanbassene.github.io/v12-demo/

## 🔧 Outils
```bash
# Convertir .mat CWRU en .csv
python tools/mat2csv.py chemin/vers/97.mat
```
