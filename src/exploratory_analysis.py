import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

class ExploratoryAnalysis:
    @staticmethod
    def render_dashboard(transactions: list, rules_df: pd.DataFrame, vis_instance):
        st.header("Análise Global do Histórico do Curso")
        st.caption("Visão exploratória do comportamento geral das matrículas e das regras extraídas pelo modelo Apriori.")
        
        # ==========================================
        # 1. PROCESSAMENTO ÚNICO DOS DADOS
        # ==========================================
        todos_itens = [item for transacao in transactions for item in transacao]
        df = pd.DataFrame(todos_itens, columns=['Item'])
        
        # Separa Disciplina e Status
        df['Disciplina'] = df['Item'].str.replace('_Aprovado', '').str.replace('_Reprovado', '')
        df['Status'] = df['Item'].apply(lambda x: 'Aprovado' if '_Aprovado' in x else 'Reprovado')
        
        total_transacoes = len(transactions)

        # ==========================================
        # 2. ABA DE VISUALIZAÇÃO DE MATRÍCULAS
        # ==========================================
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["Matrículas Totais", "Aprovações", "Reprovações"])

        with tab1:
            st.subheader("Volume Total de Matrículas por Disciplina")
            df_total = df.groupby('Disciplina').size().reset_index(name='Quantidade')
            df_total['Suporte'] = df_total['Quantidade'] / total_transacoes
            df_top = df_total.sort_values('Suporte', ascending=True).tail(20)
            
            fig = px.bar(df_top, x='Suporte', y='Disciplina', orientation='h', 
                         title="Top 20 Disciplinas - Volume Total", 
                         color='Suporte', color_continuous_scale='Viridis')
            fig.update_layout(yaxis=dict(automargin=True), margin=dict(l=150), height=600)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Top 20 Disciplinas com Mais Aprovações")
            df_aprov = df[df['Status'] == 'Aprovado'].groupby('Disciplina').size().reset_index(name='Quantidade')
            df_aprov['Suporte'] = df_aprov['Quantidade'] / total_transacoes
            df_aprov = df_aprov.sort_values('Suporte', ascending=True).tail(20)
            
            fig = px.bar(df_aprov, x='Suporte', y='Disciplina', orientation='h', 
                         title="Top 20 Disciplinas - Aprovações", 
                         color='Suporte', color_continuous_scale='Greens')
            fig.update_layout(yaxis=dict(automargin=True), margin=dict(l=150), height=600)
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Top 20 Disciplinas com Mais Reprovações")
            df_reprov = df[df['Status'] == 'Reprovado'].groupby('Disciplina').size().reset_index(name='Quantidade')
            df_reprov['Suporte'] = df_reprov['Quantidade'] / total_transacoes
            df_reprov = df_reprov.sort_values('Suporte', ascending=True).tail(20)
            
            fig = px.bar(df_reprov, x='Suporte', y='Disciplina', orientation='h', 
                         title="Top 20 Disciplinas - Reprovações", 
                         color='Suporte', color_continuous_scale='Reds')
            fig.update_layout(yaxis=dict(automargin=True), margin=dict(l=150), height=600)
            st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # 3. CHORD DIAGRAM (Conexões)
        # ==========================================
        st.markdown("---")
        st.subheader("4. Conexões Frequentes entre Disciplinas")
        
        fase_selecionada = st.selectbox("Selecione a fase para focar o diagrama:", 
                                        ["Todas as Fases", "1ª Fase", "2ª Fase", "3ª Fase", "4ª Fase", "5ª Fase", "6ª Fase", "7ª Fase", "8ª Fase"])
        
        regras_simples = rules_df[(rules_df['antecedents'].apply(len) == 1) & (rules_df['consequents'].apply(len) == 1)].copy()
        
        if fase_selecionada != "Todas as Fases":
            num_fase = fase_selecionada[0]
            regras_simples = regras_simples[regras_simples['antecedents'].apply(lambda x: list(x)[0].startswith(num_fase)) | 
                                            regras_simples['consequents'].apply(lambda x: list(x)[0].startswith(num_fase))]
        
        regras_simples = regras_simples.sort_values(by="support", ascending=False).head(60)
        
        if not regras_simples.empty:
            html_cordas = vis_instance.generate_chord_html_string(regras_simples, "grafo_cordas")
            components.html(html_cordas, height=900, scrolling=True)
        else:
            st.info("Dados insuficientes para esta fase.")

        # ==========================================
        # 4. SCATTER PLOT (Dispersão)
        # ==========================================
        st.markdown("---")
        st.subheader("5. Distribuição das Regras (Confiança vs. Lift)")
        
        regras_plot = rules_df.copy()
        regras_plot['Antecedentes'] = regras_plot['antecedents'].apply(lambda x: ", ".join(list(x)))
        regras_plot['Consequentes'] = regras_plot['consequents'].apply(lambda x: ", ".join(list(x)))
        
        fig_scatter = px.scatter(regras_plot, x="confidence", y="lift", size="support", color="lift",
                                 hover_data=["Antecedentes", "Consequentes"], color_continuous_scale='Reds')
        st.plotly_chart(fig_scatter, use_container_width=True)