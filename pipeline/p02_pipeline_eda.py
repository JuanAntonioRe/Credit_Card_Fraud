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

# EDA ----------------------------------------------------------
os.system(f"python{extension} eda/histogram_of_balance.py")

os.system(f"python{extension} eda/mean_median_balance.py")

os.system(f"python{extension} eda/new_table.py")