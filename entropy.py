import cv2
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os

def compute_frame_entropy(frame):
    """
    Calcule l'entropie d'une frame vidéo en niveaux de gris.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversion en niveaux de gris
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256), density=True)
    return entropy(hist)  # Entropie basée sur la distribution des intensités

def analyze_video_entropy(video_paths, output_dir="output_graphs"):
    """
    Analyse l'entropie frame par frame pour une liste de vidéos et sauvegarde les graphiques.
    
    :param video_paths: Liste des chemins des vidéos à analyser.
    :param output_dir: Répertoire où sauvegarder les graphiques.
    """
    # Création du répertoire de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Erreur : impossible d'ouvrir la vidéo {video_path}.")
            continue
        
        frame_entropy = []  # Liste pour stocker l'entropie de chaque frame
        timestamps = []  # Liste des timestamps des frames
        
        fps = cap.get(cv2.CAP_PROP_FPS)  # Framerate de la vidéo
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Nombre total de frames
        duration = frame_count / fps  # Durée totale de la vidéo
        video_name = os.path.basename(video_path)  # Nom de la vidéo sans le chemin
        
        print(f"Analyse de la vidéo : {video_name}")
        print(f"Nombre de frames : {frame_count}, FPS : {fps}, Durée : {duration:.2f} secondes")
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calcul de l'entropie pour chaque frame
            entropy_value = compute_frame_entropy(frame)
            frame_entropy.append(entropy_value)
            
            # Enregistrement du timestamp (temps en secondes)
            timestamps.append(frame_idx / fps)
            
            # Affichage de la progression
            if frame_idx % int(fps) == 0:  # Affiche toutes les secondes
                print(f"Frame {frame_idx}/{frame_count} analysée...")
            
            frame_idx += 1
        
        cap.release()
        
        # Calcul des statistiques
        min_entropy = min(frame_entropy)
        max_entropy = max(frame_entropy)
        mean_entropy = np.mean(frame_entropy)
        
        # Tracé du graphique
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, frame_entropy, label="Entropie", color='blue', linewidth=1.5)
        plt.axhline(mean_entropy, color='red', linestyle='--', linewidth=1, label=f"Moyenne : {mean_entropy:.2f}")
        plt.axhline(min_entropy, color='green', linestyle='--', linewidth=1, label=f"Min : {min_entropy:.2f}")
        plt.axhline(max_entropy, color='orange', linestyle='--', linewidth=1, label=f"Max : {max_entropy:.2f}")
        plt.ylim(bottom=0,top=8)
        plt.title(f"Évolution de l'entropie - {video_name}")
        plt.xlabel("Temps (secondes)")
        plt.ylabel("Entropie")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        # Sauvegarde du graphique
        output_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_entropy.png")
        plt.savefig(output_path)
        plt.close()  # Fermer la figure pour libérer de la mémoire
        
        print(f"Graphique sauvegardé : {output_path}")

# Exemple d'utilisation
video_paths = ["per_title/serie_3.mp4","per_title/jt_20h_1.ts"]  # Liste des vidéos à analyser
analyze_video_entropy(video_paths, output_dir="output_graphs")
