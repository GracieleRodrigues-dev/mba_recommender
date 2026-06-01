import pandas as pd
from src.config import STUDENT_PROFILE_FILE, ENROLLMENT_FILE

class DataLoader:
    @staticmethod
    def load_student_profiles():
        return pd.read_csv(STUDENT_PROFILE_FILE, sep=';', encoding='latin1')
        
    @staticmethod
    def load_enrollments():    
        return pd.read_csv(ENROLLMENT_FILE, sep=';', encoding='latin1')