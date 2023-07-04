#!/bin/sh

# Create enviroment
python3 -m venv bcano_python_ml
source ./bcano_python_ml/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

