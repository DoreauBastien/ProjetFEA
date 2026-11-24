"""Vérifications analytiques du solveur de treillis (python -m unittest discover -s Projet)."""

import json
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from ProjetFEA import donnees_mini_projet, resoudre_treillis


class TestTreillis(unittest.TestCase):
    def test_mini_projet(self):
        donnees = donnees_mini_projet()
        resultat = resoudre_treillis(**donnees)

        # À 45°, les diagonales ont la même raideur projetée sur X et Y.
        # Leurs couplages XY, de signes opposés, s'annulent au nœud A.
        ea = 210e9 * 1e-4
        racine_2 = np.sqrt(2)
        c = 1 / racine_2
        force = np.array([500 * np.sqrt(3), 500])
        rigidite = ea * np.diag([1 + c, c])
        u = force / rigidite.diagonal()
        ux, uy = u
        efforts = ea * np.array([(ux + uy) / 2, ux, (ux - uy) / 2])

        assert_allclose(resultat.longueurs, [racine_2, 1, racine_2])
        assert_allclose(resultat.orientations, np.pi * np.array([-3 / 4, 1, 3 / 4]))
        assert_allclose(resultat.K, resultat.K.T)
        assert_array_equal(resultat.ddl_libres, [0, 1])
        assert_allclose(resultat.K_red, rigidite, atol=1e-8)
        assert_allclose(resultat.F_red, force)
        assert_allclose(resultat.deplacements[0], u)
        assert_array_equal(resultat.deplacements[1:], np.zeros((3, 2)))
        assert_allclose(resultat.efforts_normaux, efforts)
        assert_allclose(resultat.contraintes, efforts / 1e-4)
        directions = [[-c, -c], [-1, 0], [-c, c]]
        reactions = efforts[:, None] * directions
        assert_allclose(resultat.reactions[1:], reactions, atol=1e-10)
        assert_allclose(resultat.reactions[0], 0, atol=1e-10)
        assert_allclose(resultat.reactions.sum(axis=0), -force)
        bilan = resultat.reactions + resultat.F.reshape(-1, 2)
        x, y = np.asarray(donnees["coord"]).T
        self.assertAlmostEqual(np.sum(x * bilan[:, 1] - y * bilan[:, 0]), 0)

    def test_treillis_json_diametres_et_liaisons(self):
        # Ce triangle de référence reste fixe pendant les essais avec --cas.
        donnees = json.loads("""{
            "coord": [[0, 0], [2, 0], [1, 1]],
            "connec": [[0, 1], [0, 2], [1, 2]],
            "module": 210000000000,
            "diametre": [0.016, 0.012, 0.010],
            "liaisons": ["bloque", "libre_x", "libre"],
            "val_f": [[0, 0], [0, 0], [0, -1000]]
        }""")
        resultat = resoudre_treillis(**donnees)

        # L'équilibre aux nœuds donne une traction P/2 dans la base et une
        # compression P/√2 dans chaque diagonale. Les allongements de ces
        # barres donnent ensuite les déplacements nodaux par projection.
        charge = 1000
        racine_2 = np.sqrt(2)
        sections = np.pi * (np.array([0.016, 0.012, 0.010]) / 2)**2
        ea = 210e9 * sections
        ux1 = charge / ea[0]
        allongement_gauche, allongement_droite = -charge / ea[1:]
        ux2 = (ux1 + racine_2 * (allongement_gauche - allongement_droite)) / 2
        uy2 = (-ux1 + racine_2 * (allongement_gauche + allongement_droite)) / 2

        assert_allclose(resultat.sections, sections)
        assert_array_equal(resultat.ddl_libres, [2, 4, 5])
        assert_allclose(resultat.deplacements, [[0, 0], [ux1, 0], [ux2, uy2]])
        assert_allclose(resultat.efforts_normaux, [charge / 2, -charge / racine_2, -charge / racine_2])
        assert_allclose(resultat.reactions, [[0, 500], [0, 500], [0, 0]], atol=1e-10)

    def test_td4_exercice_2_appui_incline(self):
        longueur, module, section, charge = 1, 210e9, 6e-4, 1e6
        deplacement_unitaire = charge * longueur / (module * section)
        resultat = resoudre_treillis(
            [[0, 0], [0, longueur], [longueur, longueur]],
            [[0, 1], [1, 2], [0, 2]],
            module, [[section], [section], [section * np.sqrt(2)]],
            bc_u=[[1, 1], [0, 1], [0, 1]],
            val_f=[[0, 0], [charge, 0], [0, 0]],
            angles_appuis=[0, 0, np.pi / 4],
        )

        # Le nœud 3 glisse à 45°. La barre 2 reprend P en compression et
        # l'équilibre tangent au nœud 3 donne N3 = P/√2 en traction.
        assert_allclose(resultat.deplacements, deplacement_unitaire * np.array([
            [0, 0], [1.5, 0], [0.5, 0.5],
        ]))
        assert_allclose(resultat.efforts_normaux, [0, -charge, charge / np.sqrt(2)])
        assert_allclose(resultat.reactions, charge * np.array([
            [-0.5, -0.5], [0, 0], [-0.5, 0.5],
        ]), atol=1e-8)

    def test_deplacement_impose_et_charge_sur_appui(self):
        resultat = resoudre_treillis(
            [[0, 0], [1, 0]], [[1, 0]], 1e6, 0.01,
            bc_u=[[1, 1], [0, 1]], val_u=[[0.001, 0], [0, 0]],
            val_f=[[5, 0], [20, 100]], bc_f=[[1, 0], [1, 0]],
        )
        assert_allclose(resultat.F_red, [30])
        assert_allclose(resultat.deplacements, [[0.001, 0], [0.003, 0]])
        assert_allclose(resultat.reactions, [[-25, 0], [0, 0]], atol=1e-12)
        assert_allclose(resultat.efforts_normaux, [20])

    def test_tous_les_degres_de_liberte_imposes(self):
        resultat = resoudre_treillis(
            [[0, 0], [1, 0]], [[0, 1]], 1e6, 0.01,
            bc_u=[[1, 1], [1, 1]], val_u=[[0, 0], [0.001, 0]],
        )
        self.assertEqual(resultat.K_red.shape, (0, 0))
        assert_allclose(resultat.reactions, [[-10, 0], [10, 0]])

    def test_mecanisme(self):
        with self.assertRaisesRegex(ValueError, "singulière"):
            resoudre_treillis(
                [[0, 0], [1, 1]], [[0, 1]], 1e6, 0.01,
                bc_u=[[1, 1], [0, 0]],
            )


if __name__ == "__main__":
    unittest.main()
