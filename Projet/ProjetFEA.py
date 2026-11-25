import argparse
import json

import numpy as np

parser = argparse.ArgumentParser(description="Calcule le treillis du mini-projet ou un cas JSON.")
parser.add_argument("--cas", help="fichier JSON du treillis à calculer")
parser.add_argument("--details", action="store_true", help="affiche les étapes du calcul")
args = parser.parse_args()

# 1. Données du treillis.
if args.cas is None:
    P, alpha = 1000., np.deg2rad(30)
    donnees = {
        "coord": [[1., 0.], [0., -1.], [0., 0.], [0., 1.]],
        "connec": [[0, 1], [0, 2], [0, 3]],
        "module": 210e9,
        "section": 1e-4,
        "liaisons": ["libre", "bloque", "bloque", "bloque"],
        "val_f": [[P * np.cos(alpha), P * np.sin(alpha)], [0, 0], [0, 0], [0, 0]],
    }
else:
    try:
        with open(args.cas, encoding="utf-8") as fichier:
            donnees = json.load(fichier)
    except (OSError, ValueError) as erreur:
        parser.error(str(erreur))

# La version courte traite les appuis suivant X ou Y, avec blocage à zéro.
champs = {"coord", "connec", "module", "section", "diametre", "liaisons", "bc_u", "val_f"}
if not isinstance(donnees, dict):
    parser.error("Le JSON doit contenir un objet de données.")
champs_en_plus = donnees.keys() - champs
if champs_en_plus:
    parser.error(f"Utiliser ProjetFEA.py pour ces champs : {', '.join(sorted(champs_en_plus))}.")

liaison_vers_blocage = {
    "libre": [0, 0], "bloque": [1, 1], "libre_x": [0, 1], "libre_y": [1, 0],
}
try:
    coord = np.asarray(donnees["coord"], dtype=float)
    if coord.ndim != 2 or coord.shape[1] != 2 or len(coord) < 2 or not np.isfinite(coord).all():
        raise ValueError("coord doit contenir au moins deux paires (X, Y) finies.")

    barres = np.asarray(donnees["connec"])
    if (barres.ndim != 2 or barres.shape[1] != 2 or len(barres) == 0
            or not np.issubdtype(barres.dtype, np.integer)
            or np.any(barres < 0) or np.any(barres >= len(coord))):
        raise ValueError("connec doit contenir des paires d’indices de nœuds valides, à partir de 0.")

    if ("section" in donnees) == ("diametre" in donnees):
        raise ValueError("Donner soit section en m², soit diametre en m.")
    E = np.broadcast_to(np.asarray(donnees["module"], dtype=float).ravel(), (len(barres),))
    if "diametre" in donnees:
        D = np.asarray(donnees["diametre"], dtype=float)
        if np.any(D <= 0):
            raise ValueError("Les diamètres doivent être strictement positifs.")
        S = np.pi * D**2 / 4
    else:
        S = np.asarray(donnees["section"], dtype=float)
    S = np.broadcast_to(S.ravel(), (len(barres),))
    if np.any(E <= 0) or np.any(S <= 0) or not (np.isfinite(E).all() and np.isfinite(S).all()):
        raise ValueError("Les modules et les sections doivent être finis et strictement positifs.")

    if ("liaisons" in donnees) == ("bc_u" in donnees):
        raise ValueError("Donner soit liaisons, soit bc_u.")
    if "liaisons" in donnees:
        blocages = [liaison_vers_blocage[liaison] for liaison in donnees["liaisons"]]
    else:
        blocages = donnees["bc_u"]
    blocages = np.asarray(blocages)
    if blocages.shape != coord.shape or not np.isin(blocages, [0, 1]).all():
        raise ValueError("Indiquer les deux composantes libres ou bloquées de chaque nœud.")

    forces = np.asarray(donnees.get("val_f", np.zeros_like(coord)), dtype=float)
    if forces.shape != coord.shape or not np.isfinite(forces).all():
        raise ValueError("val_f doit donner une paire de forces (F_X, F_Y) par nœud.")
except KeyError as erreur:
    parser.error(f"Champ manquant ou liaison inconnue : {erreur}.")
except (TypeError, ValueError) as erreur:
    parser.error(str(erreur))

noms_noeuds = list("ABCD") if args.cas is None else [f"N{i}" for i in range(len(coord))]
separateur = "" if args.cas is None else "-"
noms_barres = [f"{noms_noeuds[i]}{separateur}{noms_noeuds[j]}" for i, j in barres]
noms_ddl = [f"U_{nom}{axe}" for nom in noms_noeuds for axe in "XY"]

n = 2 * len(coord)
K = np.zeros((n, n))
longueurs, orientations, elements = [], [], []
rigidites = []

for e, (i, j) in enumerate(barres):
    # 2. Géométrie de la barre i → j.
    dx, dy = coord[j] - coord[i]
    L = np.hypot(dx, dy)
    if L == 0 or not np.isfinite(L):
        parser.error(f"La barre {noms_barres[e]} doit avoir une longueur finie et positive.")
    longueurs.append(L)
    orientations.append(np.arctan2(dy, dx))
    c, s = dx / L, dy / L

    # 3. Rigidité de la barre et assemblage.
    b = np.array([-c, -s, c, s])
    ddl = [2*i, 2*i + 1, 2*j, 2*j + 1]
    k = E[e] * S[e] / L
    K_e = k * np.outer(b, b)
    K[np.ix_(ddl, ddl)] += K_e
    rigidites.append(K_e)
    elements.append((ddl, b, k))

# 4. Forces nodales et sélection des déplacements libres.
F = forces.ravel()
libres = np.flatnonzero(blocages.ravel() == 0)
K_red = K[np.ix_(libres, libres)]
F_red = F[libres]

# 5. Résolution puis calcul des réactions et des efforts.
U = np.zeros(n)
if len(libres):
    if np.linalg.matrix_rank(K_red) < len(libres):
        parser.error("Matrice réduite singulière. Vérifier les liaisons et la stabilité du treillis.")
    U[libres] = np.linalg.solve(K_red, F_red)
R = K @ U - F
N = np.array([k * (b @ U[ddl]) for ddl, b, k in elements])

if args.details:
    with np.printoptions(precision=5, linewidth=max(100, n * 18)):
        print(f"1. Coordonnées (m) dans l'ordre {', '.join(noms_noeuds)}\n", coord)
        print("\n   Connectivité (indices à partir de 0)", barres)
        print("\n   Module E (Pa) par barre", E)
        print("   Section S (m²) par barre", S)

        print("\n2. Longueurs obtenues par hypot(ΔX, ΔY) et angles par atan2(ΔY, ΔX)")
        print(f"   {'Barre':<8}{'Longueur (m)':>14}{'Angle (°)':>12}")
        for nom, longueur, angle in zip(noms_barres, longueurs, np.rad2deg(orientations)):
            print(f"   {nom:<8}{longueur:14.6g}{angle:12.6g}")

        print("\n3. Matrices de rigidité (1 MN/m = 10⁶ N/m)")
        for nom, matrice in zip(noms_barres, rigidites):
            print(f"\n   Barre {nom}\n{matrice / 1e6}")
        print("\n   Matrice globale K\n", K / 1e6)

        print("\n4. Forces nodales F (N)")
        for nom, force in zip(noms_ddl, F):
            print(f"   F_{nom[2:]:<10}{force:12.6g}")
        print("\n   Déplacements libres", ", ".join(noms_ddl[i] for i in libres) or "aucun")
        print("\n   Matrice réduite K_red (MN/m)\n", K_red / 1e6)
        print("\n   Second membre F_red (N)\n", F_red.reshape(-1, 1))

        print("\n5. Solution du système réduit (m)", U[libres])
        print()

if args.cas is None:
    print("Déplacement de A (mm)", 1000 * U[:2])
    print("Réactions aux nœuds (N)\n", R.reshape(-1, 2))
    print("Efforts AB, AC, AD (N)", N)
else:
    print(f"{'Nœud':<10}{'UX (mm)':>12}{'UY (mm)':>12}{'RX (N)':>12}{'RY (N)':>12}")
    for nom, (ux, uy), (rx, ry) in zip(noms_noeuds, U.reshape(-1, 2), R.reshape(-1, 2)):
        print(f"{nom:<10}{1000 * ux:12.6f}{1000 * uy:12.6f}{rx:12.3f}{ry:12.3f}")
    print("\nEfforts dans les barres (N)")
    for nom, effort in zip(noms_barres, N):
        print(f"   {nom:<12}{effort:12.3f}")
