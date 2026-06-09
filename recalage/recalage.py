#!/usr/bin/env python3

import os
import glob
import cv2
import numpy as np
import shutil

def creer_masque(img_gray, ratio_ciel=0.45, seuil_neige=230):
    """Active le bas de l'image (roche) et désactive le haut (ciel) et le blanc (neige).
        Augmenter le ratio_ciel si le ciel représente plus de 45% de l'image en hauteur.
        Augmenter le seuil_neige pour être moins restrictif (un seuil à 210 par exemple
         pourrait garder la neige à l'ombre.)."""
    masque = np.zeros_like(img_gray)
    limite_ciel = int(img_gray.shape[0] * ratio_ciel)
    masque[limite_ciel:, :] = 255 # on active toute la partie basse (le repère commence du haut)

    # on regarde s'il y a trop de neige
    masque_filtre_neige = masque.copy()
    masque_filtre_neige[img_gray > seuil_neige] = 0 # on désactive les pixels trop clairs (neige)

    # si 80% du masque avec la neige est à 0, on ne la filtre pas
    nb_pix_rest = np.count_nonzero(masque_filtre_neige)
    ratio = nb_pix_rest / np.count_nonzero(masque)
    if ratio > 0.2 :
        return masque_filtre_neige
    else:
        return masque

def skyline_backup(img_gray):
    """Lorsque le recalage n'a pas fonctionné avec SIFT, car pas assez de keypoints ou 
    transformation aberrante, on utilise la ligne de crête pour recaler l'image."""
    lines = cv2.Canny(img_gray, )



def recalage_panorama(chemin_reference, dossier_entree, dossier_sortie):
    """Fonction principale de recalage"""

    os.makedirs(dossier_sortie, exist_ok=True) #crée dossier_sortie s'il n'existe pas

    # ETAPE 1: TRAITEMENT IMAGE DE REFERENCE

    # on la charge
    img_ref = cv2.imread(chemin_reference)
    if img_ref is None:
        print(f"Erreur : Impossible de lire {chemin_reference}")
        return

    # on met en gris car les fonctions cv2 fonctionnent mieux en gris
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    hauteur_ref, largeur_ref = gray_ref.shape

    # création du masque pour ignorer la neige et le ciel
    masque_ref = creer_masque(gray_ref)

    # CLAHE pour le contre-jour, pour voir les détails de l'ombre
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_ref = clahe.apply(gray_ref)

    # initialisation de SIFT pour trouver les keypoints
    sift = cv2.SIFT_create(nfeatures=5000)

    # on passe le masque à l'algorithme : il ne cherchera des points qu'en bas sur du sombre
    keypoints_ref, descripteurs_ref = sift.detectAndCompute(gray_ref, mask=masque_ref)

    # matcher pour pouvoir comparer avec les autres images
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    # ETAPE 2: TRAITEMENT DES AUTRES IMAGES

    # on récupère toutes les photos
    fichiers_a_traiter = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        fichiers_a_traiter.extend(glob.glob(os.path.join(dossier_entree, ext)))
        # si jamais on a des photos ecrites en .JPG, .JPEG ...
        fichiers_a_traiter.extend(glob.glob(os.path.join(dossier_entree, ext.upper())))

    # boucle de traitement
    for chemin_image in fichiers_a_traiter:
        nom_fichier = os.path.basename(chemin_image)
        chemin_sauvegarde = os.path.join(dossier_sortie, nom_fichier)

        # si on retombe sur l'image de référence alors on va à l'image suivante
        if os.path.abspath(chemin_image) == os.path.abspath(chemin_reference):
            continue

        print(f"Traitement de {nom_fichier}...")
        img_align = cv2.imread(chemin_image)
        if img_align is None:
            continue

        # on remet en gris, même raison qu'avant
        gray_align = cv2.cvtColor(img_align, cv2.COLOR_BGR2GRAY)

        # masque pour enlever la neige et le ciel
        masque_align = creer_masque(gray_align)

        # on applique le filtre clahe
        gray_align = clahe.apply(gray_align)

        # on recalcule les keypoints pour pouvoir ensuite comparer avec imgref
        keypoints_align, descripteurs_align = sift.detectAndCompute(gray_align, mask=masque_align)

        # si on a trouvé aucun descripteur, alors on renvoie l'image de base
        if descripteurs_align is None:
            print("   [Échec] Aucun point détecté.")
            shutil.copy(chemin_image, chemin_sauvegarde)
            continue

        # on demande de trouver les 2 points de imgref les plus ressemblants,
        #  pour chaque keypoint de l'image en cours.
        matches = matcher.knnMatch(descripteurs_align, descripteurs_ref, k=2)

        bons_matches = []
        for match_candidats in matches:
            if len(match_candidats) == 2:
                m, n = match_candidats
                # test de ratio de Lowe, le premier choix m doit être au moins 30%
                # meilleur que le deuxième choix n, sinon on n'est pas sûr de la correspondance.
                if m.distance < 0.7 * n.distance:
                    bons_matches.append(m)

        # si on n'a pas trouvé assez de correspondance, on renvoie l'image de base
        if len(bons_matches) < 6:
            print(f"   [Échec] Pas assez de repères fixes trouvés ({len(bons_matches)}).")
            shutil.copy(chemin_image, chemin_sauvegarde)
            continue

        # on rentre les coordonnées des points trouvés dans deux tableaux
        points_align = np.zeros((len(bons_matches), 2), dtype=np.float32)
        points_ref = np.zeros((len(bons_matches), 2), dtype=np.float32)

        for i, match in enumerate(bons_matches):
            points_align[i, :] = keypoints_align[match.queryIdx].pt
            points_ref[i, :] = keypoints_ref[match.trainIdx].pt

        # Dernier filtre, la caméra subit que de légères perturbations
        # on cherche une transformation affine entre les 2 images
        # ransacReprojThreshold=2.0 force une tolérance d'erreur de 2 pixels max.
        matrice_rigide, _ = cv2.estimateAffinePartial2D(
            points_align,
            points_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0
        )

        if matrice_rigide is not None:
            # la matrice affine en 2D est composée comme suit:
            # [ a   b   tx ]
            # [ c   d   ty ]
            # avec a = scos(rot) et c = ssin(rot) et rot~0
            a = matrice_rigide[0, 0] # échelle et cosinus de rotation (devrait être ~1.0)
            c = matrice_rigide[1, 0] # sinus de rotation (devrait être ~0.0)
            tx = matrice_rigide[0, 2] # translation horizontale en pixels
            ty = matrice_rigide[1, 2] # translation verticale en pixels

            # on vérifie le décalage (on interdit un mouvement > 200px)
            mouvement_max = 200
            translation_logique = (abs(tx) < mouvement_max) and (abs(ty) < mouvement_max)

            # on vérifie la rotation et l'échelle
            # a doit rester très proche de 1 (on tolère 5% de déformation)
            # c doit rester très proche de 0 (pas de rotation folle)
            rotation_logique = (0.95 < a < 1.05) and (abs(c) < 0.1)

            # tout est normal, on applique le recalage
            if translation_logique and rotation_logique:
                img_alignee_finale = cv2.warpAffine(img_align, matrice_rigide, (largeur_ref, hauteur_ref))
                cv2.imwrite(chemin_sauvegarde, img_alignee_finale)
                print(f"   [Succès] Alignée (Décalage: X={int(tx)}px, Y={int(ty)}px)")
            # sinon, on renvoie l'image de base
            else:
                print("   [REJET] Calcul aberrant détecté. Sauvegarde de l'image brute.")
                shutil.copy(chemin_image, chemin_sauvegarde)

        else:
            # si l'algorithme n'a trouvé aucune matrice mathématique
            print("   [Échec] Calcul impossible. Sauvegarde de l'image brute.")
            shutil.copy(chemin_image, chemin_sauvegarde)

    print("\n--- TERMINÉ ---")

# changer ici les dossiers et image de ref
if __name__ == "__main__":
    IMAGE_REF = "/Users/placideneuilly/Desktop/stage-neige/data/echant-juin-octobre-2023/2023-06-14-12-00-03-001.jpg"
    DOSSIER_SOURCE = "/Users/placideneuilly/Desktop/stage-neige/data/echant-juin-octobre-2023/"
    DOSSIER_RESULTAT = "/Users/placideneuilly/Desktop/stage-neige/recalage/resultats_alignes/"

    recalage_panorama(IMAGE_REF, DOSSIER_SOURCE, DOSSIER_RESULTAT)
    shutil.copy(IMAGE_REF, DOSSIER_RESULTAT)
