# Liste des labels correspondant aux classes
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
          'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
          'V', 'W', 'X', 'Y', 'Z']

# Taille de l'image pour le modèle
IMAGE_SIZE = 50 # Ajuster selon la taille que tu as utilisée pour l'entraînement

TRAIN_DATA_PATH = "datasets/asl"

NUM_OF_LETTERS = 36
NUM_OF_CHANNELS = 1  # Grayscale
NUM_OF_DENSE_LAYER_NODES = (IMAGE_SIZE * IMAGE_SIZE) // 2
