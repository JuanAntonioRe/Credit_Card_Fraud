import os, sys
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código
import platform

operative_system = platform.system()

# Define executable extensions
if operative_system == 'Windows':
    extension = ".exe"
else:
    extension = ""
    
# Preprocess --------------------------------------------------
os.system(f"python{extension} preprocessing/a01_preprocess.py")

os.system(f"python{extension} preprocessing/a02_split_train_test.py")

# Model
os.system(f"python{extension} models/b01_models_creation.py")