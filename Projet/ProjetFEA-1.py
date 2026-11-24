"""Calcul statique de treillis plans articulés, en élasticité linéaire.

Lancer `python ProjetFEA.py --details` pour parcourir le mini-projet.
Pour un autre treillis, utiliser `--cas exemple_treillis.json` ou appeler
`resoudre_treillis` avec ses propres données.
Les unités sont le mètre, le newton et le pascal. La connectivité commence à 0.
"""

import argparse
import json
from dataclasses import dataclass

import numpy as np


@dataclass
class ResultatTreillis:
    """Résultats intermédiaires et finaux, dans l'ordre du mini-projet.

    Les noms suivent l'énoncé (Length, Orient, N_BC_U, Row_Column_elimin,
    K_red_inter, K_red, F_red, q_red). K et F sont dans le repère global
    (X1, Y1, X2, Y2, …). K_red_inter, K_red, F_red et q_red utilisent les axes
    d'appui si ceux-ci sont inclinés.
    Les déplacements et réactions sont dans le repère global, un nœud par
    ligne. Un effort normal positif correspond à une barre en traction.
    """

    longueurs: np.ndarray
    orientations: np.ndarray
    rigidites_elementaires: np.ndarray
    K: np.ndarray
    F: np.ndarray
    N_BC_U: int
    Row_Column_elimin: np.ndarray
    ddl_libres: np.ndarray
    K_red_inter: np.ndarray
    K_red: np.ndarray
    F_red: np.ndarray
    q_red: np.ndarray
    deplacements: np.ndarray
    reactions: np.ndarray
    deformations: np.ndarray
    contraintes: np.ndarray
    efforts_normaux: np.ndarray
    sections: np.ndarray


def _tableau(valeurs, forme, nom):
    tableau = np.asarray(valeurs, dtype=float)
    if tableau.shape != forme or not np.all(np.isfinite(tableau)):
        raise ValueError(f"{nom} doit être de forme {forme} et contenir des valeurs finies.")
    return tableau


def _propriete_barres(valeurs, nombre_barres, nom):
    tableau = np.asarray(valeurs, dtype=float)
    if tableau.ndim == 0:
        tableau = np.full(nombre_barres, tableau)
    elif tableau.shape == (nombre_barres, 1):
        tableau = tableau[:, 0]
    tableau = _tableau(tableau, (nombre_barres,), nom)
    if np.any(tableau <= 0):
        raise ValueError(f"{nom} doit être strictement positif pour chaque barre.")
    return tableau


def _masque(valeurs, forme, nom):
    tableau = _tableau(valeurs, forme, nom)
    if np.any((tableau != 0) & (tableau != 1)):
        raise ValueError(f"{nom} doit contenir uniquement des 0 ou des 1.")
    return tableau.astype(bool)


def resoudre_treillis(coord, connec, module, section=None, *, diametre=None,
                      bc_u=None, liaisons=None, val_u=None, val_f=None,
                      bc_f=None, angles_appuis=None):
    """Résout le treillis défini par `coord` (Nn × 2) et `connec` (Nr × 2).

    Donner `section` en m² ou `diametre` en m pour des barres circulaires
    pleines. Ces propriétés et `module` acceptent une valeur commune ou une
    valeur par barre, dans l'ordre de `connec`.

    Donner `bc_u` (Nn × 2, 1 pour une composante imposée) ou `liaisons` avec
    une valeur par nœud parmi `libre`, `bloque`, `libre_x` et `libre_y`.
    `libre_x` laisse x libre et impose y, `libre_y` fait l'inverse.
    `val_u` donne les déplacements imposés, nuls par défaut. `val_f` donne
    les forces globales aux nœuds, nulles par défaut. Le masque facultatif
    `bc_f` permet de ne retenir que certaines composantes de `val_f`.

    Par défaut, les deux colonnes des conditions cinématiques suivent les
    axes globaux. `angles_appuis` donne, en radians, l'angle de l'axe local x
    de chaque nœud par rapport à l'axe global X, vers Y. Dans ce cas, les
    conditions de déplacement s'appliquent aux axes locaux (x, y).
    """
    # 1. Données du maillage et propriétés des barres.
    coord = np.asarray(coord, dtype=float)
    if coord.ndim != 2 or coord.shape[1] != 2 or len(coord) < 2:
        raise ValueError("Coord doit contenir au moins deux lignes (X, Y).")
    n_noeuds = len(coord)
    forme_nodale = (n_noeuds, 2)
    coord = _tableau(coord, forme_nodale, "Coord")
    connec = np.asarray(connec)
    if (connec.ndim != 2 or connec.shape[1] != 2 or len(connec) == 0
            or not np.issubdtype(connec.dtype, np.integer)):
        raise ValueError("Connec doit contenir au moins une paire d'indices entiers.")
    if np.any(connec < 0) or np.any(connec >= n_noeuds):
        raise ValueError("Les indices de Connec doivent aller de 0 à Nn - 1.")
    n_barres = len(connec)
    module = _propriete_barres(module, n_barres, "Le module d'Young")
    if (section is None) == (diametre is None):
        raise ValueError("Fournir soit section, soit diametre.")
    if diametre is not None:
        diametres = _propriete_barres(diametre, n_barres, "Le diamètre")
        section = np.pi * diametres**2 / 4
    section = _propriete_barres(section, n_barres, "La section")

    # Notations de l'énoncé. Attention : en Python les nœuds et les barres sont
    # numérotés à partir de 0 (A = 0, B = 1, ...), l'énoncé part de 1.
    Nn = n_noeuds
    Nr = n_barres

    # 2. Length[i] et Orient[i] de chaque barre, à partir des coordonnées.
    #    La barre i va du nœud connec[i][0] vers le nœud connec[i][1].
    Length = np.zeros(Nr)
    Orient = np.zeros(Nr)
    cos_orient = np.zeros(Nr)
    sin_orient = np.zeros(Nr)
    for i in range(Nr):
        n1 = connec[i][0]
        n2 = connec[i][1]
        dX = coord[n2][0] - coord[n1][0]
        dY = coord[n2][1] - coord[n1][1]
        Length[i] = np.sqrt(dX**2 + dY**2)
        if not Length[i] > 0 or not np.isfinite(Length[i]):
            raise ValueError("Chaque barre doit avoir une longueur finie et strictement positive.")
        # arctan2 tient compte des signes de dX et dY, donc donne le bon quadrant.
        Orient[i] = np.arctan2(dY, dX)
        # cos(Orient) = dX/L et sin(Orient) = dY/L. On les garde sous cette forme
        # pour avoir des zéros exacts (cos(π/2) vaut 6e-17 en machine, pas 0).
        cos_orient[i] = dX / Length[i]
        sin_orient[i] = dY / Length[i]

    # 3. Matrices de rigidité élémentaires Ke et assemblage dans K (2Nn × 2Nn).
    #    Numérotation des DDL : le DDL X du nœud n est 2n, le DDL Y est 2n + 1
    #    (dans l'énoncé, avec des indices à partir de 1 : 2n - 1 et 2n).
    K = np.zeros((2 * Nn, 2 * Nn))
    rigidites_elementaires = np.zeros((Nr, 4, 4))
    for i in range(Nr):
        c = cos_orient[i]
        s = sin_orient[i]
        Ke = module[i] * section[i] / Length[i] * np.array([
            [ c * c,  c * s, -c * c, -c * s],
            [ c * s,  s * s, -c * s, -s * s],
            [-c * c, -c * s,  c * c,  c * s],
            [-c * s, -s * s,  c * s,  s * s],
        ])
        rigidites_elementaires[i] = Ke

        # Assemblage : boucle sur les 2 nœuds j et k de la barre i.
        # Énoncé : K[2·connec[i][j]-1][2·connec[i][k]-1] += Ke[2j-1][2k-1], etc.
        # Avec des indices Python (à partir de 0), le "-1" disparaît côté X
        # et devient "+1" côté Y.
        for j in range(2):
            for k in range(2):
                nj = connec[i][j]
                nk = connec[i][k]
                K[2 * nj][2 * nk] += Ke[2 * j][2 * k]                  # X_j, X_k
                K[2 * nj + 1][2 * nk] += Ke[2 * j + 1][2 * k]          # Y_j, X_k
                K[2 * nj][2 * nk + 1] += Ke[2 * j][2 * k + 1]          # X_j, Y_k
                K[2 * nj + 1][2 * nk + 1] += Ke[2 * j + 1][2 * k + 1]  # Y_j, Y_k

    # 4. Conditions aux limites et réduction du système.
    if (bc_u is None) == (liaisons is None):
        raise ValueError("Fournir soit bc_u, soit liaisons.")
    if liaisons is not None:
        types = {"libre": (0, 0), "bloque": (1, 1), "libre_x": (0, 1), "libre_y": (1, 0)}
        liaisons = np.asarray(liaisons, dtype=str)
        if liaisons.shape != (n_noeuds,):
            raise ValueError("Fournir une liaison par nœud, dans l'ordre de Coord.")
        try:
            bc_u = [types[liaison] for liaison in liaisons]
        except KeyError as erreur:
            raise ValueError(f"Liaison inconnue {erreur}. Choisir parmi {', '.join(types)}.") from erreur
    # 4.a Tableaux de l'énoncé, de taille Nn.
    #     BC_Ux[i] = 1 si Ux est imposé au nœud i, 0 sinon (idem Uy, Fx, Fy).
    #     VAL_Ux[i] = valeur de Ux au nœud i si BC_Ux[i] = 1 (idem Uy, Fx, Fy).
    bc_u = _masque(bc_u, forme_nodale, "bc_u").astype(int)
    BC_Ux = bc_u[:, 0]
    BC_Uy = bc_u[:, 1]
    val_u = np.zeros(forme_nodale) if val_u is None else _tableau(val_u, forme_nodale, "val_u")
    VAL_Ux = val_u[:, 0]
    VAL_Uy = val_u[:, 1]
    # Par défaut, toutes les forces données dans val_f sont appliquées.
    bc_f = np.ones(forme_nodale, dtype=int) if bc_f is None else _masque(bc_f, forme_nodale, "bc_f").astype(int)
    BC_Fx = bc_f[:, 0]
    BC_Fy = bc_f[:, 1]
    val_f = np.zeros(forme_nodale) if val_f is None else _tableau(val_f, forme_nodale, "val_f")
    VAL_Fx = val_f[:, 0]
    VAL_Fy = val_f[:, 1]

    # 4.b Nombre de conditions cinématiques.
    N_BC_U = int(np.sum(BC_Ux + BC_Uy))

    # 4.c Lignes et colonnes à éliminer (1 = DDL imposé).
    Row_Column_elimin = np.zeros(2 * Nn, dtype=int)
    for i in range(Nn):
        if BC_Ux[i] == 1:
            Row_Column_elimin[2 * i] = 1
        if BC_Uy[i] == 1:
            Row_Column_elimin[2 * i + 1] = 1

    # 4.d Vecteur des forces nodales F (2Nn) et des déplacements imposés U_imp (2Nn).
    F = np.zeros(2 * Nn)
    U_imp = np.zeros(2 * Nn)
    for i in range(Nn):
        if BC_Fx[i] == 1:
            F[2 * i] = VAL_Fx[i]
        if BC_Fy[i] == 1:
            F[2 * i + 1] = VAL_Fy[i]
        if BC_Ux[i] == 1:
            U_imp[2 * i] = VAL_Ux[i]
        if BC_Uy[i] == 1:
            U_imp[2 * i + 1] = VAL_Uy[i]

    # Extension (hors énoncé) : appuis inclinés, comme l'exercice 2 du TD4.
    # On exprime K et F dans les axes locaux des appuis : K' = Tᵀ K T, F' = Tᵀ F.
    # Sans appui incliné, K_appuis = K et F_appuis = F (cas de l'énoncé).
    K_appuis, F_appuis = K, F
    changement_base = None
    if angles_appuis is not None:
        angles = _tableau(angles_appuis, (Nn,), "angles_appuis")
        changement_base = np.eye(2 * Nn)
        for noeud, angle in enumerate(angles):
            c, s = np.cos(angle), np.sin(angle)
            ddl = slice(2 * noeud, 2 * noeud + 2)
            changement_base[ddl, ddl] = [[c, -s], [s, c]]
        # Les colonnes de T sont les axes locaux exprimés dans le repère global.
        K_appuis = changement_base.T @ K @ changement_base
        F_appuis = changement_base.T @ F

    # 4.e Matrice intermédiaire K_red_inter (2Nn - N_BC_U) × 2Nn :
    #     on garde seulement les lignes de K des DDL non imposés.
    K_red_inter = np.zeros((2 * Nn - N_BC_U, 2 * Nn))
    k = 0
    for i in range(2 * Nn):
        if Row_Column_elimin[i] == 0:
            for j in range(2 * Nn):
                K_red_inter[k][j] = K_appuis[i][j]
            k = k + 1

    # 4.f Matrice réduite K_red (2Nn - N_BC_U) × (2Nn - N_BC_U) :
    #     on garde seulement les colonnes de K_red_inter des DDL non imposés.
    K_red = np.zeros((2 * Nn - N_BC_U, 2 * Nn - N_BC_U))
    k = 0
    for j in range(2 * Nn):
        if Row_Column_elimin[j] == 0:
            for i in range(2 * Nn - N_BC_U):
                K_red[i][k] = K_red_inter[i][j]
            k = k + 1

    # 4.g Vecteur des forces réduites F_red, même procédure que K_red_inter.
    #     Ajout par rapport à l'énoncé : un déplacement imposé non nul U_imp[j]
    #     crée l'effort K[i][j]·U_imp[j], qu'on passe dans le second membre.
    #     Dans le cas de l'énoncé, U_imp = 0 et ce terme est nul.
    F_red = np.zeros(2 * Nn - N_BC_U)
    k = 0
    for i in range(2 * Nn):
        if Row_Column_elimin[i] == 0:
            F_red[k] = F_appuis[i]
            for j in range(2 * Nn):
                if Row_Column_elimin[j] == 1:
                    F_red[k] = F_red[k] - K_appuis[i][j] * U_imp[j]
            k = k + 1

    # 5. Résolution de [K_red]{q_red} = {F_red}.
    q_red = np.zeros(2 * Nn - N_BC_U)
    if 2 * Nn - N_BC_U > 0:
        message = "Rigidité réduite numériquement singulière. Vérifier les appuis et les mécanismes."
        if np.linalg.matrix_rank(K_red) < 2 * Nn - N_BC_U:
            raise ValueError(message)
        try:
            q_red = np.linalg.solve(K_red, F_red)
        except np.linalg.LinAlgError as erreur:
            raise ValueError(message) from erreur

    # Vecteur complet q (2Nn) : q_red sur les DDL libres, U_imp sur les DDL imposés.
    q = np.zeros(2 * Nn)
    k = 0
    for i in range(2 * Nn):
        if Row_Column_elimin[i] == 0:
            q[i] = q_red[k]
            k = k + 1
        else:
            q[i] = U_imp[i]
    if changement_base is not None:
        q = changement_base @ q  # retour au repère global

    # 6. Post-traitement (hors énoncé) : réactions R = K q - F, puis efforts.
    reactions = K @ q - F
    deformations = np.zeros(Nr)
    for i in range(Nr):
        n1 = connec[i][0]
        n2 = connec[i][1]
        # Allongement = déplacement relatif projeté sur l'axe de la barre.
        allongement = (cos_orient[i] * (q[2 * n2] - q[2 * n1])
                       + sin_orient[i] * (q[2 * n2 + 1] - q[2 * n1 + 1]))
        deformations[i] = allongement / Length[i]
    contraintes = module * deformations  # σ = E ε
    efforts_normaux = section * contraintes  # N = σ S, positif en traction

    return ResultatTreillis(
        longueurs=Length, orientations=Orient,
        rigidites_elementaires=rigidites_elementaires, K=K, F=F,
        N_BC_U=N_BC_U, Row_Column_elimin=Row_Column_elimin,
        ddl_libres=np.flatnonzero(Row_Column_elimin == 0),
        K_red_inter=K_red_inter, K_red=K_red, F_red=F_red, q_red=q_red,
        deplacements=q.reshape(forme_nodale),
        reactions=reactions.reshape(forme_nodale),
        deformations=deformations, contraintes=contraintes,
        efforts_normaux=efforts_normaux, sections=section.copy(),
    )


def donnees_mini_projet(longueur=1.0, theta=np.pi / 4, module=210e9,
                        section=1e-4, charge=1000.0, alpha=np.pi / 6):
    """Construit le cas de l'énoncé avec le repère X vers le bas, Y vers la droite.

    `theta` et `alpha` sont en radians. Les valeurs par défaut de `module`,
    `section`, `charge` et `alpha` sont choisies pour le test numérique.
    """
    if not np.isfinite(longueur) or longueur <= 0 or not 0 < theta < np.pi / 2:
        raise ValueError("L doit être positif et θ doit être compris entre 0 et π/2 (exclus).")
    if not np.isfinite(charge) or charge < 0 or not np.isfinite(alpha):
        raise ValueError("P doit être une intensité positive ou nulle et α doit être fini.")
    demi_largeur = longueur * np.tan(theta)
    return dict(
        coord=[[longueur, 0], [0, -demi_largeur], [0, 0], [0, demi_largeur]],
        connec=[[0, 1], [0, 2], [0, 3]],  # A–B, A–C, A–D
        module=module,
        section=section,
        bc_u=[[0, 0], [1, 1], [1, 1], [1, 1]],
        val_f=[[charge * np.cos(alpha), charge * np.sin(alpha)], [0, 0], [0, 0], [0, 0]],
    )


def _afficher_rigidite(nom, matrice):
    # L'unité MN/m permet de garder des colonnes compactes dans le cas de référence.
    print(f"\n{nom} ({matrice.shape[0]} × {matrice.shape[1]}, MN/m)")
    coefficients = np.where(matrice == 0, 0.0, matrice) / 1e6
    for ligne in coefficients:
        print(" ".join(f"{valeur:9.5g}" for valeur in ligne))


def _afficher_details(donnees, resultat, noms_noeuds, noms_barres):
    coord = np.asarray(donnees["coord"], dtype=float)
    connec = np.asarray(donnees["connec"])
    n_noeuds = len(noms_noeuds)
    print(f"1. Coordonnées (m) dans l'ordre {', '.join(noms_noeuds)}\n{coord}")
    print(f"\n   Connectivité (indices à partir de 0)\n{connec}")
    module = np.broadcast_to(np.asarray(donnees["module"], dtype=float).ravel(), resultat.sections.shape)
    diametre = donnees.get("diametre")
    if diametre is not None:
        print(f"\n   Diamètres fournis (m) {np.asarray(diametre).ravel()}")
        print("   Sections des barres circulaires pleines calculées par S = πD²/4.")
    print(f"\n{'Barre':<14}{'E (Pa)':>14}{'S (m²)':>14}")
    for nom, E, S in zip(noms_barres, module, resultat.sections):
        print(f"{nom:<14}{E:14.6g}{S:14.6g}")

    print("\n2. Longueur et orientation de chaque barre")
    print("   ΔX et ΔY sont les écarts de coordonnées du premier nœud vers le second.")
    print("   La longueur vaut √(ΔX² + ΔY²). L’angle est mesuré depuis +X, positif vers +Y.")
    print("   La fonction atan2(ΔY, ΔX) utilise les deux signes pour trouver le bon quadrant.")
    print(f"\n{'Barre':<14}{'ΔX (m)':>12}{'ΔY (m)':>12}{'Longueur (m)':>15}{'Angle (°)':>12}")
    ecarts = coord[connec[:, 1]] - coord[connec[:, 0]]
    angles = np.rad2deg(resultat.orientations)
    for nom, (dx, dy), longueur, angle in zip(noms_barres, ecarts, resultat.longueurs, angles):
        print(f"{nom:<14}{dx:12.6g}{dy:12.6g}{longueur:15.6g}{angle:12.6g}")

    print("\n3. Matrices de rigidité (1 MN/m = 10⁶ N/m)")
    print("   Les lignes et colonnes suivent X, Y pour chaque nœud. Les matrices de barre")
    print("   suivent la connectivité. La matrice globale suit l’ordre des coordonnées.")
    for nom, matrice in zip(noms_barres, resultat.rigidites_elementaires):
        _afficher_rigidite(f"   Barre {nom}", matrice)
    _afficher_rigidite("   Matrice globale K", resultat.K)

    print("\n4. Forces nodales et conditions aux limites")
    print(f"\n{'Composante globale':<22}{'Force (N)':>14}")
    for i, force in enumerate(resultat.F):
        noeud, direction = divmod(i, 2)
        nom_force = f"F_{noms_noeuds[noeud]}{'XY'[direction]}"
        print(f"{nom_force:<22}{force:14.6g}")

    axes = "XY"
    angles_appuis = donnees.get("angles_appuis")
    if angles_appuis is not None:
        axes = "xy"
        print("\n   Les déplacements ci-dessous sont exprimés dans les axes locaux d’appui.")
    noms_ddl = [f"U_{nom}{axe}" for nom in noms_noeuds for axe in axes]
    ddl_libres = set(resultat.ddl_libres)
    valeurs = donnees.get("val_u")
    valeurs = np.zeros(2 * n_noeuds) if valeurs is None else np.asarray(valeurs, dtype=float).ravel()
    print(f"\n{'Nœud':<14}{axes[0]:>18}{axes[1]:>18}")
    for noeud, nom in enumerate(noms_noeuds):
        etats = ["libre" if i in ddl_libres else f"imposé à {valeurs[i]:g} m"
                 for i in (2 * noeud, 2 * noeud + 1)]
        print(f"{nom:<14}{etats[0]:>18}{etats[1]:>18}")

    print(f"\n   N_BC_U = {resultat.N_BC_U} (nombre de déplacements imposés)")
    print("\n   Row_Column_elimin (1 = ligne et colonne à éliminer)")
    for nom, valeur in zip(noms_ddl, resultat.Row_Column_elimin):
        print(f"   {nom:<14}{valeur:>4}")

    inconnues = [noms_ddl[i] for i in resultat.ddl_libres]
    print(f"\n   Déplacements à résoudre ({len(inconnues)} sur {2 * n_noeuds}) "
          f"{', '.join(inconnues) or 'aucun'}.")
    print("   Les déplacements imposés sont connus. Les réactions sont calculées ensuite.")
    if inconnues:
        _afficher_rigidite("   K_red_inter : lignes des DDL libres", resultat.K_red_inter)
        _afficher_rigidite("   K_red : colonnes des DDL libres", resultat.K_red)
        print("\n   Second membre F_red (N)")
        for nom, force in zip(inconnues, resultat.F_red):
            print(f"   {nom:<14}{force:14.6g}")

    print("\n5. Déplacements calculés q_red (m)")
    for nom, valeur in zip(inconnues, resultat.q_red):
        print(f"   {nom:<14}{valeur:14.8g}")
    if not inconnues:
        print("   Tous les déplacements sont imposés.")


def main():
    parser = argparse.ArgumentParser(
        description="Résout le treillis du mini-projet ou un treillis fourni en JSON.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--cas", help="fichier JSON de données, par exemple exemple_treillis.json")
    parser.add_argument("--longueur", type=float, help="hauteur L en m (défaut : 1)")
    parser.add_argument("--theta", type=float, help="angle des diagonales en degrés (45)")
    parser.add_argument("--module", type=float, help="module d'Young en Pa (210e9)")
    parser.add_argument("--section", type=float, help="section en m² (1e-4)")
    parser.add_argument("--charge", type=float, help="intensité P en N (1000)")
    parser.add_argument("--alpha", type=float, help="angle de la charge en degrés (30)")
    parser.add_argument("--details", action="store_true", help="affiche les matrices et vecteurs intermédiaires")
    parametres = vars(parser.parse_args())
    cas = parametres.pop("cas", None)
    details = parametres.pop("details", False)
    if cas is not None and parametres:
        parser.error("Avec --cas, régler les paramètres du treillis dans le fichier JSON.")
    try:
        if cas is None:
            for angle in ("theta", "alpha"):
                if angle in parametres:
                    parametres[angle] = np.deg2rad(parametres[angle])
            donnees = donnees_mini_projet(**parametres)
        else:
            with open(cas, encoding="utf-8") as fichier:
                donnees = json.load(fichier)
            if not isinstance(donnees, dict):
                raise ValueError("Le fichier de données doit contenir un objet JSON.")
        resultat = resoudre_treillis(**donnees)
    except (OSError, TypeError, ValueError) as erreur:
        parser.error(str(erreur))

    if cas is None:
        noms_noeuds = list("ABCD")
        noms_barres = ["AB", "AC", "AD"]
    else:
        noms_noeuds = [f"N{i}" for i in range(len(donnees["coord"]))]
        noms_barres = [f"{noms_noeuds[i]}-{noms_noeuds[j]}" for i, j in donnees["connec"]]
    if details:
        _afficher_details(donnees, resultat, noms_noeuds, noms_barres)

    print("\nDéplacements et réactions nodales")
    print(f"{'Nœud':<5} {'UX (mm)':>12} {'UY (mm)':>12} {'RX (N)':>12} {'RY (N)':>12}")
    for nom, u, r in zip(noms_noeuds, resultat.deplacements, resultat.reactions):
        print(f"{nom:<5} {1e3*u[0]:12.6f} {1e3*u[1]:12.6f} {r[0]:12.3f} {r[1]:12.3f}")
    print("\nEfforts dans les barres (positif en traction)")
    print(f"{'Barre':<5} {'N (N)':>12} {'σ (MPa)':>12}")
    for nom, effort, contrainte in zip(noms_barres, resultat.efforts_normaux, resultat.contraintes):
        print(f"{nom:<5} {effort:12.3f} {contrainte/1e6:12.3f}")


if __name__ == "__main__":
    main()
