"""Convertit un fichier .mat CWRU en .csv pour la page de demo.
Usage : python mat2csv.py <fichier.mat> [fichier.csv]"""
import sys, os, numpy as np, scipy.io as sio

def convert(path, out=None):
    if not os.path.exists(path):
        print(f"Fichier introuvable: {path}")
        return
    mat = sio.loadmat(path)
    sig = None
    for k in mat:
        if 'DE_time' in k:
            sig = mat[k].flatten().astype(np.float32)
            break
    if sig is None:
        print("Aucun signal DE_time trouve")
        return
    if out is None:
        out = os.path.splitext(os.path.basename(path))[0] + '.csv'
    # Sous-echantillonnage a ~12000 Hz (CWRU est deja a 12000 Hz)
    step = max(1, len(sig) // 12000)
    sig_ds = sig[::step][:12000]
    np.savetxt(out, sig_ds, delimiter=',')
    print(f"Converti: {os.path.basename(path)} -> {out} ({len(sig_ds)} echant.)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python mat2csv.py <fichier.mat> [fichier.csv]")
        print("Exemple: python mat2csv.py 97.mat")
        # Test avec CWRU par defaut
        cwru = r"C:\Users\j1jea\Desktop\V6-Engine\datasets_public\cwru"
        for f in ['97.mat','105.mat','169.mat']:
            p = os.path.join(cwru, f)
            if os.path.exists(p): convert(p)
    else:
        convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
