# V12/V14 Demo — Bearing Fault Detection

Page web de démonstration pour l'API de détection de défauts de roulements.

## Live demo (GitHub Pages)
https://jeanbassene.github.io/v12-demo/

## Utilisation
1. Ouvrir la page
2. Glisser un fichier .csv de vibrations
3. Diagnostic instantané : feu + forge + type

## Convertir un fichier .mat CWRU
```bash
python tools/mat2csv.py chemin/vers/97.mat
```

## API backend
L'API Flask tourne sur PythonAnywhere ou Render.
URL à configurer dans `index.html` (variable `API_URL`).

## Benchmarks
| Dataset | Précision |
|---|---|
| CWRU | 92% |
| Ottawa (Channel 2) | 73% |
| Ottawa (Channel 1, CV>0.70) | 92% |
