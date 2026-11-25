import argparse

import numpy as np

parser = argparse.ArgumentParser(description="Calcule le treillis du mini-projet.")
parser.add_argument("--details", action="store_true", help="affiche les étapes du calcul")
args = parser.parse_args()

# 1. Données du treillis, dans l'ordre A, B, C, D.
coord = np.array([[1., 0.], [0., -1.], [0., 0.], [0., 1.]])
barres = [(0, 1), (0, 2), (0, 3)]
Nn = len(coord)
Nr = len(barres)
Module = np.full(Nr, 210e9)
Section = np.full(Nr, 1e-4)
P, alpha = 1000., np.deg2rad(30)

# 2. Longueur et orientation de chaque barre.
longueurs = np.zeros(Nr)
orientations = np.zeros(Nr)
for e, (i, j) in enumerate(barres):
    dx, dy = coord[j] - coord[i]
    longueurs[e] = np.hypot(dx, dy)
    orientations[e] = np.arctan2(dy, dx)

# 3. Rigidités élémentaires et assemblage dans K.
n = 2 * Nn
K = np.zeros((n, n))
elements = []
rigidites = []

for e, noeuds in enumerate(barres):
    c = np.cos(orientations[e])
    s = np.sin(orientations[e])
    k = Module[e] * Section[e] / longueurs[e]
    K_e = k * np.array([
        [ c*c,  c*s, -c*c, -c*s],
        [ c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s],
    ])
    rigidites.append(K_e)

    # Pour chaque paire de nœuds, ajouter les contributions XX, YX, XY et YY.
    for a, noeud_a in enumerate(noeuds):
        for b, noeud_b in enumerate(noeuds):
            ligne, colonne = 2 * noeud_a, 2 * noeud_b
            K[ligne, colonne] += K_e[2*a, 2*b]
            K[ligne + 1, colonne] += K_e[2*a + 1, 2*b]
            K[ligne, colonne + 1] += K_e[2*a, 2*b + 1]
            K[ligne + 1, colonne + 1] += K_e[2*a + 1, 2*b + 1]

    i, j = noeuds
    ddl = [2*i, 2*i + 1, 2*j, 2*j + 1]
    projection = np.array([-c, -s, c, s])
    elements.append((ddl, projection, k))

# 4. Conditions aux limites, dans l'ordre A, B, C, D.
BC_Ux = [0, 1, 1, 1]
BC_Uy = [0, 1, 1, 1]
VAL_Ux = [0., 0., 0., 0.]
VAL_Uy = [0., 0., 0., 0.]
BC_Fx = [1, 0, 0, 0]
BC_Fy = [1, 0, 0, 0]
VAL_Fx = [P * np.cos(alpha), 0., 0., 0.]
VAL_Fy = [P * np.sin(alpha), 0., 0., 0.]

N_BC_U = sum(BC_Ux) + sum(BC_Uy)
n_libres = n - N_BC_U
Row_Column_elimin = np.zeros(n, dtype=int)
U = np.zeros(n)
# Deux composantes de force par nœud, donc 2*Nn entrées.
F = np.zeros(n)
for i in range(Nn):
    if BC_Ux[i] == 1:
        Row_Column_elimin[2*i] = 1
        U[2*i] = VAL_Ux[i]
    if BC_Uy[i] == 1:
        Row_Column_elimin[2*i + 1] = 1
        U[2*i + 1] = VAL_Uy[i]
    if BC_Fx[i] == 1:
        F[2*i] = VAL_Fx[i]
    if BC_Fy[i] == 1:
        F[2*i + 1] = VAL_Fy[i]

# Première passe : conserver les lignes libres.
K_red_inter = np.zeros((n_libres, n))
F_red = np.zeros(n_libres)
libres = []
ligne_red = 0
for i in range(n):
    if Row_Column_elimin[i] == 0:
        K_red_inter[ligne_red, :] = K[i, :]
        F_red[ligne_red] = F[i]
        libres.append(i)
        ligne_red += 1

# Seconde passe : conserver les colonnes libres.
K_red = np.zeros((n_libres, n_libres))
colonne_red = 0
for j in range(n):
    if Row_Column_elimin[j] == 0:
        K_red[:, colonne_red] = K_red_inter[:, j]
        colonne_red += 1

# Reporter au second membre la contribution des déplacements imposés.
F_red -= K_red_inter @ U

# 5. Résolution puis calcul des réactions et des efforts.
q_red = np.linalg.solve(K_red, F_red)
U[libres] = q_red
R = K @ U - F
N = np.array([k * (projection @ U[ddl]) for ddl, projection, k in elements])

if args.details:
    noms_barres = ("AB", "AC", "AD")
    with np.printoptions(precision=5, linewidth=100, suppress=True):
        print("1. Coordonnées (m) dans l'ordre A, B, C, D\n", coord)
        print("\n   Connectivité (indices à partir de 0)", barres)
        print("\n   Module E (Pa) par barre", Module)
        print("   Section S (m²) par barre", Section)

        print("\n2. Longueurs obtenues par hypot(ΔX, ΔY) et angles par atan2(ΔY, ΔX)")
        print(f"   {'Barre':<8}{'Longueur (m)':>14}{'Angle (°)':>12}")
        for nom, longueur, angle in zip(noms_barres, longueurs, np.rad2deg(orientations)):
            print(f"   {nom:<8}{longueur:14.6g}{angle:12.6g}")

        print("\n3. Matrices de rigidité (1 MN/m = 10⁶ N/m)")
        for nom, matrice in zip(noms_barres, rigidites):
            print(f"\n   Barre {nom}\n{matrice / 1e6}")
        print("\n   Matrice globale K\n", K / 1e6)

        print("\n4. Forces nodales F (N) dans l'ordre AX, AY, BX, BY, CX, CY, DX, DY")
        print(F.reshape(-1, 1))
        print("\n   Lignes et colonnes à éliminer (1 = imposé)", Row_Column_elimin)
        print("   Nombre de déplacements imposés", N_BC_U)
        print("   Indices des déplacements libres", libres)
        print("\n   Matrice intermédiaire K_red_inter (MN/m)\n", K_red_inter / 1e6)
        print("\n   Matrice réduite K_red (MN/m)\n", K_red / 1e6)
        print("\n   Second membre F_red (N)\n", F_red.reshape(-1, 1))

        print("\n5. Solution du système réduit (m)", np.array2string(q_red, suppress_small=False))
        print()

print("Déplacement de A (mm)", 1000 * U[:2])
print("Réactions aux nœuds (N)\n", np.array2string(R.reshape(-1, 2), precision=6, suppress_small=True))
print("Efforts AB, AC, AD (N)", N)
