# General configuration for paths and parameters 
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data')
STUDENT_PROFILE_FILE = os.path.join(DATA_PATH, "perfil_academicos.csv")
ENROLLMENT_FILE = os.path.join(DATA_PATH, "matriculas_udesc.csv")

MIN_SUPPORT = 0.005
MIN_CONFIDENCE = 0.10
