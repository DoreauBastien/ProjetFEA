import argparse

import numpy as np

parser = argparse.ArgumentParser(description="Calcule le treillis du mini-projet.")
parser.add_argument("--details", action="store_true", help="affiche les étapes du calcul")
args = parser.parse_args()

# 1. Données du treillis, dans l'ordre A, B, C, D.
coord = np.array([[1., 0.], [0., -1.], [0., 0.], [0., 1.]])
barres = [(0, 1), (0, 2), (0, 3)]
E, S = 210e9, 1e-4
P, alpha = 1000., np.deg2rad(30)

n = 2 * len(coord)
K = np.zeros((n, n))
longueurs, orientations, elements = [], [], []
rigidites = []

for i, j in barres:
    # 2. Géométrie de la barre i → j.
    dx, dy = coord[j] - coord[i]
    L = np.hypot(dx, dy)
    longueurs.append(L)
    orientations.append(np.arctan2(dy, dx))
    c, s = dx / L, dy / L

    # 3. Rigidité de la barre et assemblage.
    b = np.array([-c, -s, c, s])
    ddl = [2*i, 2*i + 1, 2*j, 2*j + 1]
    k = E * S / L
    K_e = k * np.outer(b, b)
    K[np.ix_(ddl, ddl)] += K_e
    rigidites.append(K_e)
    elements.append((ddl, b, k))

# 4. Charge en A et sélection des déplacements libres.
F = np.zeros(n)
F[:2] = P * np.array([np.cos(alpha), np.sin(alpha)])

libres = [0, 1]  # U_AX et U_AY
K_red = K[np.ix_(libres, libres)]
F_red = F[libres]

# 5. Résolution puis calcul des réactions et des efforts.
U = np.zeros(n)
U[libres] = np.linalg.solve(K_red, F_red)
R = K @ U - F
N = np.array([k * (b @ U[ddl]) for ddl, b, k in elements])

if args.details:
    noms_barres = ("AB", "AC", "AD")
    with np.printoptions(precision=5, linewidth=100):
        print("1. Coordonnées (m) dans l'ordre A, B, C, D\n", coord)
        print("\n   Connectivité (indices à partir de 0)", barres)
        print(f"\n   Propriétés communes E = {E:g} Pa, S = {S:g} m²")

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
        print("\n   B, C et D sont bloqués. Les deux déplacements libres sont U_AX et U_AY.")
        print("\n   Matrice réduite K_red (MN/m)\n", K_red / 1e6)
        print("\n   Second membre F_red (N)\n", F_red.reshape(-1, 1))

        print("\n5. Solution du système réduit (m)", U[libres])
        print()

print("Déplacement de A (mm)", 1000 * U[:2])
print("Réactions aux nœuds (N)\n", R.reshape(-1, 2))
print("Efforts AB, AC, AD (N)", N)
