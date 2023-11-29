import os

# Ruta al directorio que contiene los archivos y carpetas
directorios = ['/Users/blancacanocamarero/repositorios/TFM/Kernel-methods-comparatives/Basic-measures/analysis',
              '/Users/blancacanocamarero/repositorios/TFM/Kernel-methods-comparatives/Basic-measures/analysis/CPU-SMALL/plot']

for directorio in directorios:
# Itera a través de los archivos y carpetas en el directorio
    for nombre in os.listdir(directorio):
        # Comprueba si el nombre contiene la palabra "CPU_SMALL"
        if 'CPU_SMALL' in nombre:
            # Crea el nuevo nombre reemplazando "CPU_SMALL" por "CPU-SMALL"
            nuevo_nombre = nombre.replace('CPU_SMALL', 'CPU-SMALL')
            # Obtiene la ruta completa del archivo o carpeta original
            ruta_original = os.path.join(directorio, nombre)
            # Obtiene la ruta completa del nuevo nombre
            nuevo_path = os.path.join(directorio, nuevo_nombre)
            # Renombra el archivo o carpeta
            os.rename(ruta_original, nuevo_path)
            print(f'Renombrado: {nombre} -> {nuevo_nombre}')
