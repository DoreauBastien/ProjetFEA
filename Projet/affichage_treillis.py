"""Affichage graphique d'un treillis résolu par ProjetFEA.

    python affichage_treillis.py                         cas de l'énoncé
    python affichage_treillis.py --cas exemple_treillis.json
    python affichage_treillis.py --png treillis.png      enregistre l'image

Nœuds colorés selon la liaison, barres avec leur effort normal (+ traction,
- compression), forces appliquées en flèches, déformée amplifiée en pointillés.
"""

import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import numpy as np

from ProjetFEA import donnees_mini_projet, resoudre_treillis

# (Ux imposé, Uy imposé) -> (nom dans la légende, couleur)
LIAISONS = {
    (0, 0): ("libre", "tab:green"),
    (1, 1): ("bloqué", "tab:red"),
    (0, 1): ("glisse en x (y bloqué)", "tab:orange"),
    (1, 0): ("glisse en y (x bloqué)", "tab:purple"),
}
TYPES = {"libre": (0, 0), "bloque": (1, 1), "libre_x": (0, 1), "libre_y": (1, 0)}


def tracer(donnees, resultat, noms, repere_enonce=False, fichier=None):
    coord = np.asarray(donnees["coord"], dtype=float)
    connec = np.asarray(donnees["connec"])
    if "bc_u" in donnees:
        bc = np.asarray(donnees["bc_u"], dtype=int)
    else:
        bc = np.array([TYPES[l] for l in donnees["liaisons"]])
    forces = np.zeros_like(coord)
    if donnees.get("val_f") is not None:
        forces = np.asarray(donnees["val_f"], dtype=float)

    # Repère de l'énoncé : X vers le bas, Y vers la droite.
    # À l'écran on trace donc (Y, -X) pour retrouver le dessin du sujet.
    def ecran(points):
        points = np.atleast_2d(points)
        return np.column_stack((points[:, 1], -points[:, 0])) if repere_enonce else points

    taille = np.ptp(coord, axis=0).max()

    # Déformée amplifiée pour que le déplacement max fasse 10 % de la taille.
    q = resultat.deplacements
    amplif = 0.1 * taille / np.abs(q).max() if np.abs(q).max() > 0 else 0
    deformee = coord + amplif * q

    fig, ax = plt.subplots(figsize=(9, 6))

    # Barres + effort normal au milieu.
    for (a, b), N in zip(connec, resultat.efforts_normaux):
        P = ecran(coord[[a, b]])
        D = ecran(deformee[[a, b]])
        ax.plot(P[:, 0], P[:, 1], color="black", lw=2.5, zorder=1)
        ax.plot(D[:, 0], D[:, 1], "--", color="gray", lw=1, zorder=1)
        milieu = P.mean(axis=0)
        ax.text(milieu[0], milieu[1], f"{N:+.0f} N", ha="center", va="center",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="none"))

    # Forces appliquées : flèches de longueur proportionnelle à l'intensité.
    fmax = np.hypot(forces[:, 0], forces[:, 1]).max()
    for n, f in enumerate(forces):
        if fmax == 0 or not np.any(f):
            continue
        depart = ecran(coord[n])[0]
        fleche = ecran(f)[0] / fmax * 0.25 * taille
        ax.annotate("", xy=depart + fleche, xytext=depart, annotation_clip=False, zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2))
        ax.plot(*(depart + fleche), alpha=0)  # pour que le cadre englobe la flèche
        ax.text(*(depart + fleche), f" {np.hypot(*f):.0f} N", color="tab:blue")

    # Nœuds, couleur = type de liaison.
    for n, p in enumerate(ecran(coord)):
        nom, couleur = LIAISONS[tuple(bc[n])]
        ax.scatter(*p, s=150, color=couleur, edgecolor="black", zorder=3, label=nom)
        ax.annotate(noms[n], p, textcoords="offset points", xytext=(8, 8), fontweight="bold")

    # Légende sans doublons.
    ax.plot([], [], "--", color="gray", label=f"déformée (×{amplif:.0f})")
    ax.plot([], [], color="tab:blue", lw=2, label="force appliquée")
    poignees, etiquettes = ax.get_legend_handles_labels()
    uniques = dict(zip(etiquettes, poignees))
    ax.legend(uniques.values(), uniques.keys(), loc="upper left", bbox_to_anchor=(1, 1))

    ax.set_aspect("equal")
    ax.margins(0.2)
    # Grille : traits forts aux graduations, traits légers entre deux.
    # Le pas s'adapte à la taille du treillis.
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", color="0.75", lw=0.8)
    ax.grid(which="minor", color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    if repere_enonce:
        # À l'écran l'axe vertical vaut -X : on réaffiche X, positif vers le bas.
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{-v + 0:g}"))
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("X (m), vers le bas")
    else:
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
    ax.set_title("Treillis : liaisons, efforts normaux (+ traction) et forces")
    if fichier:
        fig.savefig(fichier, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Affiche le treillis résolu.")
    parser.add_argument("--cas", help="fichier JSON (sinon cas de l'énoncé)")
    parser.add_argument("--png", help="enregistre l'image au lieu de l'afficher")
    args = parser.parse_args()

    if args.cas:
        with open(args.cas, encoding="utf-8") as f:
            donnees = json.load(f)
        noms = [f"N{i}" for i in range(len(donnees["coord"]))]
    else:
        donnees = donnees_mini_projet()
        noms = list("ABCD")

    resultat = resoudre_treillis(**donnees)
    tracer(donnees, resultat, noms, repere_enonce=args.cas is None, fichier=args.png)


if __name__ == "__main__":
    main()
