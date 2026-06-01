import streamlit as st
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle

try:
    from src.config import IS_CLOUD
except ImportError:
    IS_CLOUD = os.environ.get('IS_CLOUD', 'False') == 'True'

from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.apriori_model import AprioriModel
from src.recommender import CourseRecommender
from src.visualizer import Visualizer
from src.exploratory_analysis import ExploratoryAnalysis

st.set_page_config(page_title="Analise Curricular - MBA", layout="wide")
st.title("Sistema de Recomendacao e Analise Curricular")
st.markdown("Estudo de Caso baseado em Market Basket Analysis (MBA), focado no planejamento curricular.")
st.info("""
**Desenvolvido por Graciele Rodrigues** | Trabalho de Conclusão de Curso (TCC) - **UDESC** (Universidade do Estado de Santa Catarina).  
Para mais detalhes técnicos e acesso ao código-fonte, visite o [repositório no GitHub](https://github.com/GracieleRodrigues-dev/mba_recommender).
""")
st.sidebar.header("Configuracoes do Algoritmo")

if IS_CLOUD:
    st.sidebar.info("Modo Nuvem Ativo: Operando com dados cacheados. A mineração de novas regras está desativada.")
    min_support_input = 0.005
    min_confidence_input = 0.10
    modo_regras = "Utilizar regras em cache"
else:
    st.sidebar.caption("Ajuste a sensibilidade matematica da busca historica.")
    min_support_input = st.sidebar.number_input("Suporte Minimo", value=0.005, format="%.4f")
    min_confidence_input = st.sidebar.number_input("Confianca Minima", value=0.10, format="%.2f")
    st.sidebar.markdown("---")
    st.sidebar.header("Gerenciamento de Regras")
    modo_regras = st.sidebar.radio("Defina a fonte dos dados:", ("Utilizar regras em cache", "Gerar regras novamente"))

CACHE_PATH = "output/regras_cacheadas.pkl"

@st.cache_data
def carregar_dados_completos(modo, support, confidence):
    if IS_CLOUD:
        if not os.path.exists(CACHE_PATH):
            st.error("Erro crítico: Arquivo de regras não encontrado na nuvem!")
            return [], [], pd.DataFrame()
            
        with open(CACHE_PATH, "rb") as f:
            rules_df = pickle.load(f)
            
        disciplinas_unicas = set()
        for itemset in pd.concat([rules_df['antecedents'], rules_df['consequents']]):
            for item in itemset:
                disciplinas_unicas.add(item.split('_')[0])
                
        disciplinas_list = sorted(list(disciplinas_unicas))
        transactions = [] 
        
        return disciplinas_list, transactions, rules_df
        
    else:
        enrollments = DataLoader.load_enrollments()
        enrollments = DataPreprocessor.clean_enrollments(enrollments)
        transactions = DataPreprocessor.to_transactions(enrollments)    
    
        disciplinas_unicas = set()
        for t in transactions:
            for item in t:
                disciplinas_unicas.add(item.split('_')[0])
        disciplinas_list = sorted(list(disciplinas_unicas))
        
        if modo == "Gerar regras novamente" or not os.path.exists(CACHE_PATH):
            with st.spinner("Minerando dados com algoritmo Apriori. Aguarde..."):
                model = AprioriModel(min_support=support, min_confidence=confidence)
                rules_df = model.fit(transactions)
                os.makedirs("output", exist_ok=True)
                with open(CACHE_PATH, "wb") as f:
                    pickle.dump(rules_df, f)
        else:
            with open(CACHE_PATH, "rb") as f:
                rules_df = pickle.load(f)
                
        return disciplinas_list, transactions, rules_df

disciplinas_list, transactions, rules_df = carregar_dados_completos(modo_regras, min_support_input, min_confidence_input)

aba_recomendacao, aba_exploratoria = st.tabs(["Simulador de Grade", "Analise Exploratoria do Curso"])

vis = Visualizer()

with aba_exploratoria:
    if IS_CLOUD:
        st.warning("A Análise Exploratória requer acesso aos dados brutos. Por conformidade de segurança (LGPD), esta aba está desativada no ambiente em Nuvem.")
    else:
        ExploratoryAnalysis.render_dashboard(transactions, rules_df, vis)

with aba_recomendacao:
    st.markdown("### Simulador de Grade e Analise Preditiva")
    with st.expander("Instrucoes de Uso"):
        st.markdown("""
        1. **Tamanho da Grade:** Selecione a quantidade total de disciplinas que planeja cursar no semestre.
        2. **Disciplinas Base:** Insira as materias obrigatorias ou prioritarias.
        3. **Analise:** O sistema sugerira disciplinas complementares seguras e alertara sobre combinacoes perigosas.
        """)

    col_a, col_b = st.columns([1, 2])

    with col_a:
        tamanho_alvo = st.number_input("Quantidade total de materias desejada:", min_value=2, max_value=8, value=4, step=1)

    with col_b:
        cenario_selecionado = st.multiselect(
            "Materias base (prioritarias):",
            options=disciplinas_list,
            default=["35CDI", "35BAD"] if "35CDI" in disciplinas_list else []
        )

    if st.button("Executar Simulacao"):
        if not cenario_selecionado:
            st.warning("Por favor, selecione ao menos uma disciplina.")
        elif len(cenario_selecionado) > tamanho_alvo:
            st.error("A quantidade de disciplinas selecionadas nao pode ser maior que o tamanho total.")
        else:
            st.markdown("---")
            recommender = CourseRecommender(rules_df)
            target_set = set(cenario_selecionado)
            
            a_geral, a_aprov, a_reprov, best, worst = recommender.simulate_scenario(target_set, int(tamanho_alvo))
            
            if a_geral is None or a_geral.empty:
                st.warning("Nenhum historico estatisticamente relevante encontrado para esta combinacao base.")
            else:
                st.markdown("### Visao Geral Historica")
                
                st.markdown("**1. Grades mais comuns escolhidas no passado**")
                st.dataframe(vis.format_scenarios_to_dataframe(a_geral), use_container_width=True)
                
                st.markdown("**2. Grades que geraram reprovacoes no passado**")
                if not a_reprov.empty:
                    st.dataframe(vis.format_scenarios_to_dataframe(a_reprov, is_risk=True), use_container_width=True)
                else:
                    st.info("Nao foram detectadas combinacoes de alto risco nos registros estatisticos base.")
                    
                st.markdown("---")
                st.markdown(f"### Projecao e Recomendacao (Grade de {tamanho_alvo} Materias)")
                
                if best is not None:
                    st.success("**Grade Recomendada (Maior chance de aprovacao)**")
                    todas_materias_best = list(best["success_itemset"])
                    todas_materias_best.sort(key=lambda x: ("_Reprovado" in x, x))
                    st.markdown(f"#### {'{' + ', '.join(todas_materias_best) + '}'}")
                    st.write(f"Confianca estatistica: {best['success_conf']:.4f}")
                else:
                    st.warning(f"Nao ha dados consistentes para recomendar uma grade de sucesso de tamanho {tamanho_alvo}.")

                st.markdown("---")

                if worst is not None:
                    st.error("**Alerta de Risco (Evite esta combinacao)**")
                    todas_materias_worst = list(worst["risk_itemset"])
                    todas_materias_worst.sort(key=lambda x: ("_Reprovado" in x, x))
                    st.markdown(f"#### {'{' + ', '.join(todas_materias_worst) + '}'}")
                    st.write(f"Confianca estatistica do risco: {worst['risk_conf']:.4f}")
                else:
                    st.success(f"O sistema nao identificou alertas criticos de combinacoes diferentes para este volume de disciplinas.")