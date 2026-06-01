import os
import pandas as pd
import streamlit as st
from src.config import STUDENT_PROFILE_FILE, ENROLLMENT_FILE, IS_CLOUD, CACHE_FILE

class DataLoader:
    
    @staticmethod
    def load_student_profiles():
        if IS_CLOUD and not os.path.exists(STUDENT_PROFILE_FILE):
            return pd.DataFrame() 
            
        return pd.read_csv(STUDENT_PROFILE_FILE, sep=';', encoding='latin1')
        
    @staticmethod
    def load_enrollments():    
        if IS_CLOUD and not os.path.exists(ENROLLMENT_FILE):
            return pd.DataFrame()
            
        return pd.read_csv(ENROLLMENT_FILE, sep=';', encoding='latin1')

    @staticmethod
    @st.cache_data 
    def load_cached_rules():
        if os.path.exists(CACHE_FILE):
            return pd.read_pickle(CACHE_FILE)
        else:
            st.error(f"Arquivo de regras não encontrado no caminho: {CACHE_FILE}")
            return None