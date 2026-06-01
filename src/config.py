import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IS_CLOUD = os.environ.get('IS_CLOUD', 'False') == 'True'

DATA_PATH = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'output') 
STUDENT_PROFILE_FILE = os.path.join(DATA_PATH, "perfil_academicos.csv")
ENROLLMENT_FILE = os.path.join(DATA_PATH, "matriculas_udesc.csv")
CACHE_FILE = os.path.join(OUTPUT_PATH, "regras_cacheadas.pkl")

MIN_SUPPORT = 0.005
MIN_CONFIDENCE = 0.10