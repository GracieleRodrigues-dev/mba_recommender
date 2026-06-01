import streamlit as st
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.apriori_model import AprioriModel
from src.recommender import CourseRecommender
from src.visualizer import Visualizer
from src.exploratory_analysis import ExploratoryAnalysis


st.set_page_config(page_title="Analise Curricular - MBA", layout="wide")
st.title("Sistema de Recomendacao e Analise Curricular")
st.markdown("Estudo de Caso baseado em Market Basket Analysis (MBA), focado no planejamento curricular.")

st.sidebar.header("Configuracoes do Algoritmo")
st.sidebar.caption("Ajuste a sensibilidade matematica da busca historica.")

min_support_input = st.sidebar.number_input(
    "Suporte Minimo", 
    value=0.005, 
    format="%.4f",
    help="Define a frequencia minima que uma combinacao deve ter no historico. Valores menores encontram combinacoes raras; valores maiores exigem que a combinacao seja muito comum."
)
min_confidence_input = st.sidebar.number_input(
    "Confianca Minima", 
    value=0.10, 
    format="%.2f",
    help="Indica a precisao ou probabilidade da regra. Ex: 0.80 significa que 80% dos alunos que cursaram essas materias juntos tiveram o resultado indicado."
)

st.sidebar.markdown("---")
st.sidebar.header("Gerenciamento de Regras")
modo_regras = st.sidebar.radio(
    "Defina a fonte dos dados:",
    ("Utilizar regras em cache", "Gerar regras novamente")
)

CACHE_PATH = "output/regras_cacheadas.pkl"

@st.cache_data
def carregar_dados_iniciais():
    enrollments = DataLoader.load_enrollments()
    total_alunos = enrollments['MatriculaAluno'].nunique()
    print(f"Total de alunos únicos na base: {total_alunos}")
    enrollments = DataPreprocessor.clean_enrollments(enrollments)
    transactions = DataPreprocessor.to_transactions(enrollments)    
    print(f"Total de cestas (transacoes) geradas: {len(transactions)}")

    disciplinas_unicas = set()
    for t in transactions:
        for item in t:
            disciplinas_unicas.add(item.split('_')[0])
            
    return sorted(list(disciplinas_unicas)), transactions

disciplinas_list, transactions = carregar_dados_iniciais()

def obter_regras(modo, support, confidence, transacoes):
    if modo == "Gerar regras novamente" or not os.path.exists(CACHE_PATH):
        with st.spinner("Minerando dados com algoritmo Apriori. Aguarde..."):
            model = AprioriModel(min_support=support, min_confidence=confidence)
            regras = model.fit(transacoes)
            os.makedirs("output", exist_ok=True)
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(regras, f)
        return regras
    else:
        with open(CACHE_PATH, "rb") as f:
            regras = pickle.load(f)
        return regras

rules_df = obter_regras(modo_regras, min_support_input, min_confidence_input, transactions)

aba_recomendacao, aba_exploratoria = st.tabs(["Simulador de Grade", "Analise Exploratoria do Curso"])

vis = Visualizer()

with aba_exploratoria:
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
        tamanho_alvo = st.number_input(
            "Quantidade total de materias desejada:", 
            min_value=2, max_value=8, value=4, step=1,
            help="Informe com quantas materias no total voce quer fechar a sua grade neste semestre."
        )

    with col_b:
        cenario_selecionado = st.multiselect(
            "Materias base (prioritarias):",
            options=disciplinas_list,
            default=["35CDI", "35BAD"],
            help="Selecione as disciplinas que voce JA DECIDIU cursar (ex: materias atrasadas ou pre-requisitos essenciais)."
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
                st.caption("Atencao: Estas sao as combinacoes gerais que historicamente causaram reprovacoes conjuntas.")
                if not a_reprov.empty:
                    st.dataframe(vis.format_scenarios_to_dataframe(a_reprov, is_risk=True), use_container_width=True)
                else:
                    st.info("Nao foram detectadas combinacoes de alto risco nos registros estatisticos base.")
                    
                st.markdown("---")
                st.markdown(f"### Projecao e Recomendacao (Grade de {tamanho_alvo} Materias)")
                st.caption(f"Para completar a sua grade com {tamanho_alvo} materias, o algoritmo filtrou o historico e separou a escolha mais segura (historico de sucesso) da escolha mais perigosa (historico de reprovacao).")
                
                if best is not None:
                    st.success("**Grade Recomendada (Maior chance de aprovacao)**")
                    todas_materias_best = list(best["success_itemset"])
                    todas_materias_best.sort(key=lambda x: ("_Reprovado" in x, x))
                    texto_best = "{" + ", ".join(todas_materias_best) + "}"
                    
                    st.markdown(f"#### {texto_best}")
                    st.write(f"**Por que esta recomendacao?** Historicamente, ao adicionar estas materias especificas, os alunos obtiveram a maior taxa de aprovacao simultanea (Confianca estatistica: {best['success_conf']:.4f}).")
                else:
                    st.warning(f"Nao ha dados consistentes para recomendar uma grade de sucesso de tamanho {tamanho_alvo}.")

                st.markdown("---")

                if worst is not None:
                    st.error("**Alerta de Risco (Evite esta combinacao)**")
                    todas_materias_worst = list(worst["risk_itemset"])
                    todas_materias_worst.sort(key=lambda x: ("_Reprovado" in x, x))
                    texto_worst = "{" + ", ".join(todas_materias_worst) + "}"
                    
                    st.markdown(f"#### {texto_worst}")
                    st.write(f"**Por que este alerta?** O algoritmo identificou que preencher a sua grade com estas materias gerou um altissimo volume de reprovacoes no passado (Confianca estatistica do risco: {worst['risk_conf']:.4f}).")
                else:
                    st.success(f"O sistema nao identificou alertas criticos de combinacoes diferentes para este volume de disciplinas.")