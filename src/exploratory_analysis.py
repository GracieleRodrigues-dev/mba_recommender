import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

class ExploratoryAnalysis:
    @staticmethod
    def render_dashboard(transactions: list, rules_df: pd.DataFrame, vis_instance):
        st.header("Análise Global do Histórico do Curso")
        st.caption("Visão exploratória do comportamento geral das matrículas e das regras extraídas pelo modelo Apriori.")
        
        st.markdown("---")
        
        # ==========================================
        # GRÁFICO 1: BARRAS HORIZONTAIS
        # ==========================================
        st.subheader("1. Frequência de Matrículas (Suporte Individual)")
        
        todos_itens = [item for transacao in transactions for item in transacao]
        df_itens = pd.Series(todos_itens).value_counts().reset_index()
        df_itens.columns = ['Item', 'Quantidade']
        
        total_transacoes = len(transactions)
        df_itens['Suporte'] = df_itens['Quantidade'] / total_transacoes
        
        df_aprovados = df_itens[df_itens['Item'].str.contains('_Aprovado')].copy()
        df_aprovados['Disciplina'] = df_aprovados['Item'].str.replace('_Aprovado', '')
        
        df_aprovados = df_aprovados.sort_values('Suporte', ascending=True).tail(20)
        
        fig_barras = px.bar(
            df_aprovados, 
            x='Suporte', 
            y='Disciplina', 
            orientation='h',
            title="Top 20 Disciplinas com Maior Suporte de Aprovação",
            color='Suporte',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_barras, use_container_width=True)

        st.markdown("---")

        # ==========================================
        # GRÁFICO 2: CHORD DIAGRAM
        # ==========================================
        st.subheader("2. Conexões Frequentes entre Disciplinas (Chord Diagram)")
        st.caption("Filtre pela fase do curso para analisar quais disciplinas os alunos mais cursam em conjunto (linhas mais grossas representam maior frequência histórica).")
        
        # Filtro Interativo
        fases_opcoes = ["Todas as Fases", "1ª Fase", "2ª Fase", "3ª Fase", "4ª Fase", "5ª Fase", "6ª Fase", "7ª Fase", "8ª Fase"]
        fase_selecionada = st.selectbox("Selecione a fase para focar o diagrama:", fases_opcoes)
        
        regras_simples = rules_df[
            (rules_df['antecedents'].apply(len) == 1) & 
            (rules_df['consequents'].apply(len) == 1) &
            (rules_df['confidence'] > 0.10)
        ].copy()
        
        if fase_selecionada != "Todas as Fases":
            # Extrai o numero da fase selecionada (Ex: "3ª Fase" -> "3")
            num_fase = fase_selecionada[0]
            
            # Filtra as regras onde o antecedente OU o consequente pertencem a fase escolhida
            regras_simples = regras_simples[
                regras_simples['antecedents'].apply(lambda x: list(x)[0].startswith(num_fase)) |
                regras_simples['consequents'].apply(lambda x: list(x)[0].startswith(num_fase))
            ]
        
        # Limita às 60 conexões mais fortes daquela fase para não poluir visualmente o diagrama
        regras_simples = regras_simples.sort_values(by="support", ascending=False).head(60)
        
        if not regras_simples.empty:
            html_cordas = vis_instance.generate_chord_html_string(regras_simples, "grafo_cordas_temp")
            components.html(html_cordas, height=900, scrolling=True)
        else:
            st.info("O modelo não encontrou conexões com relevância estatística suficiente para esta fase específica.")

        st.markdown("---")

        # ==========================================
        # GRÁFICO 3: DISPERSÃO (SCATTER PLOT)
        # ==========================================
        st.subheader("3. Distribuição Geral das Regras (Confiança vs. Lift)")
        st.caption("Mapeamento da qualidade do modelo. Pontos mais altos e mais à direita representam as regras mais fortes extraídas.")
        
        regras_plot = rules_df.copy()
        regras_plot['Antecedentes'] = regras_plot['antecedents'].apply(lambda x: ", ".join(list(x)))
        regras_plot['Consequentes'] = regras_plot['consequents'].apply(lambda x: ", ".join(list(x)))
        
        fig_scatter = px.scatter(
            regras_plot,
            x="confidence",
            y="lift",
            size="support",
            color="lift",
            hover_data=["Antecedentes", "Consequentes"],
            labels={"confidence": "Confiança", "lift": "Lift", "support": "Suporte"},
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)