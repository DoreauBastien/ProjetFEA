"""Calcul paramétrable utilisé pour tracer les cas de treillis."""

from types import SimpleNamespace

import numpy as np


def donnees_mini_projet():
    # 1. Données de l'énoncé, dans l'ordre A, B, C, D.
    P, alpha = 1000., np.deg2rad(30)
    return {
        "coord": [[1., 0.], [0., -1.], [0., 0.], [0., 1.]],
        "connec": [(0, 1), (0, 2), (0, 3)],
        "module": 210e9,
        "section": 1e-4,
        "bc_u": [[0, 0], [1, 1], [1, 1], [1, 1]],
        "val_f": [[P * np.cos(alpha), P * np.sin(alpha)], [0, 0], [0, 0], [0, 0]],
    }


def _par_barre(valeurs, nombre, nom):
    valeurs = np.asarray(valeurs, dtype=float).ravel()
    if valeurs.size not in (1, nombre):
        raise ValueError(f"{nom} doit avoir une valeur commune ou une valeur par barre.")
    if not np.isfinite(valeurs).all() or np.any(valeurs <= 0):
        raise ValueError(f"{nom} doit contenir des valeurs finies et positives.")
    return np.broadcast_to(valeurs, (nombre,))


def resoudre_treillis(coord, connec, module, section=None, *,
                      diametre=None, bc_u=None, liaisons=None, val_f=None):
    """Calcule un treillis plan avec des appuis bloqués à zéro suivant X ou Y."""
    coord = np.asarray(coord, dtype=float)
    barres = np.asarray(connec)
    if coord.ndim != 2 or coord.shape[1] != 2 or len(coord) < 2 or not np.isfinite(coord).all():
        raise ValueError("coord doit contenir au moins deux paires (X, Y) finies.")
    if (barres.ndim != 2 or barres.shape[1] != 2 or len(barres) == 0
            or not np.issubdtype(barres.dtype, np.integer)
            or np.any(barres < 0) or np.any(barres >= len(coord))):
        raise ValueError("connec doit contenir des paires d’indices de nœuds valides.")
    if (section is None) == (diametre is None):
        raise ValueError("Donner soit section en m², soit diametre en m.")
    E = _par_barre(module, len(barres), "Le module d’Young")
    if diametre is not None:
        D = _par_barre(diametre, len(barres), "Le diamètre")
        section = np.pi * D**2 / 4
    S = _par_barre(section, len(barres), "La section")

    n = 2 * len(coord)
    K = np.zeros((n, n))
    longueurs, orientations, rigidites, elements = [], [], [], []
    for e, (i, j) in enumerate(barres):
        # 2. Géométrie de la barre i → j.
        dx, dy = coord[j] - coord[i]
        L = np.hypot(dx, dy)
        if L == 0 or not np.isfinite(L):
            raise ValueError(f"La barre {i}-{j} doit avoir une longueur finie et positive.")
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
    if (bc_u is None) == (liaisons is None):
        raise ValueError("Donner soit bc_u, soit liaisons.")
    if liaisons is not None:
        types = {"libre": (0, 0), "bloque": (1, 1), "libre_x": (0, 1), "libre_y": (1, 0)}
        bc_u = [types[liaison] for liaison in liaisons]
    blocages = np.asarray(bc_u)
    forces = np.zeros_like(coord) if val_f is None else np.asarray(val_f, dtype=float)
    if blocages.shape != coord.shape or not np.isin(blocages, [0, 1]).all():
        raise ValueError("Indiquer les deux composantes libres ou bloquées de chaque nœud.")
    if forces.shape != coord.shape or not np.isfinite(forces).all():
        raise ValueError("val_f doit donner une paire de forces finies par nœud.")
    F = forces.ravel()
    libres = np.flatnonzero(blocages.ravel() == 0)
    K_red = K[np.ix_(libres, libres)]
    F_red = F[libres]

    # 5. Résolution puis calcul des réactions et des efforts.
    U = np.zeros(n)
    if len(libres):
        if np.linalg.matrix_rank(K_red) < len(libres):
            raise ValueError("Matrice réduite singulière. Vérifier la stabilité du treillis.")
        U[libres] = np.linalg.solve(K_red, F_red)
    R = K @ U - F
    N = np.array([k * (b @ U[ddl]) for ddl, b, k in elements])
    return SimpleNamespace(
        longueurs=longueurs, orientations=orientations, rigidites=rigidites,
        K=K, F=F, K_red=K_red, F_red=F_red, q_red=U[libres],
        deplacements=U.reshape(-1, 2), reactions=R.reshape(-1, 2), efforts_normaux=N,
    )
