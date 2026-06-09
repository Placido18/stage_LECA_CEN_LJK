#!/usr/bin/env python3

import os
import subprocess

def decoupage(input_rep, output_path):
    """extrait avec projection rectiligne le centre de tous les panoramas du
    dossier input_rep"""

    os.makedirs(output_path, exist_ok=True) #crée dossier_sortie s'il n'existe pas

    for file in os.listdir(input_rep):

        file_path = os.path.join(input_rep, file)
        name_output = f"{file[5:10]}-centre.jpg"
        output = os.path.join(output_path, name_output)

        commande = [
            "ffmpeg", "-y",
            "-i", file_path, 
            "-vf", "v360=input=cylindrical:output=rectilinear:ih_fov=210:yaw=0:h_fov=80:v_fov=40", 
            "-frames:v", "1", 
            output
        ]

        subprocess.run(commande, check=False)


if __name__=="__main__":

    OUTPUT = "/Users/placideneuilly/Desktop/stage-neige/decoup/test"
    SOURCE = "/Users/placideneuilly/Desktop/stage-neige/data/echant-juin-octobre-2023"

    decoupage(SOURCE, OUTPUT)
