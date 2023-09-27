#!/bin/bash

# Obtener la lista de archivos ignorados por git
ignored_files=$(cat ../.gitignore)

# Eliminar cada archivo ignorado
for file in $ignored_files; do
    if [ -f "$file" ] || [ -d "$file" ]; then
        rm -rf "$file"
        echo "Archivo eliminado: $file"
    fi
done
