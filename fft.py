import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def process_frame_with_fft(frame):
    """
    Applique une FFT 2D sur une image, retourne l'image originale avec le spectre FFT incrusté.
    """
    # Conversion en niveaux de gris
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Application de la FFT
    f = np.fft.fft2(gray_frame)
    fshift = np.fft.fftshift(f)  # Mettre le zéro fréquentiel au centre
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Échelle logarithmique pour visualisation
    
    # Création de l'image FFT avec matplotlib
    fig, ax = plt.subplots()
    ax.imshow(magnitude_spectrum, cmap='gray')
    ax.set_title('FFT Magnitude Spectrum')
    ax.set_xlabel('Frequency (u)')
    ax.set_ylabel('Frequency (v)')
    
    # Ajout d'axes avec échelle
    num_ticks = 5  # Nombre de graduations sur chaque axe
    h, w = gray_frame.shape
    x_ticks = np.linspace(0, w, num_ticks)
    y_ticks = np.linspace(0, h, num_ticks)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{int(val - w // 2)}' for val in x_ticks])  # Fréquences centrées
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(val - h // 2)}' for val in y_ticks])  # Fréquences centrées
    
    # Sauvegarde temporaire de l'image FFT pour incrustation
    temp_fft_path = 'temp_fft.png'
    fig.savefig(temp_fft_path, bbox_inches='tight')
    plt.close(fig)
    
    # Chargement de l'image FFT pour incrustation
    fft_img = cv2.imread(temp_fft_path)
    fft_img_resized = cv2.resize(fft_img, (frame.shape[1] // 2, frame.shape[0] // 2))
    os.remove(temp_fft_path)
    
    # Incrustation de l'image FFT sur la frame originale
    frame[0:fft_img_resized.shape[0], 0:fft_img_resized.shape[1]] = fft_img_resized
    
    return frame

def process_video(input_path, output_path):
    """
    Traite une vidéo en appliquant la FFT à chaque frame et en sauvegardant le résultat.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Erreur : Impossible de lire la vidéo {input_path}")
        return
    
    # Lecture des propriétés de la vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Création du writer pour la vidéo de sortie
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Traitement de la frame {frame_count}")
        processed_frame = process_frame_with_fft(frame)
        out.write(processed_frame)
        frame_count += 1
    
    cap.release()
    out.release()
    print(f"Traitement terminé. Vidéo sauvegardée sous {output_path}")

# Utilisation du script
input_video = 'short.ts'  # Remplacez par le chemin de votre vidéo
output_video = 'short_fft_video.mp4'  # Chemin pour la vidéo de sortie

process_video(input_video, output_video)
