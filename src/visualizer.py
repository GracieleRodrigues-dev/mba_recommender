import os
import pandas as pd
from d3blocks import D3Blocks
import logging

# Oculta avisos não essenciais da biblioteca d3blocks no terminal
logging.getLogger("d3blocks").setLevel(logging.ERROR)

class Visualizer:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def format_rules_to_dataframe(self, df_rules: pd.DataFrame, report_type="success") -> pd.DataFrame:
        """
        Formata o DataFrame de regras clássicas para exibição tabular limpa.
        """
        if df_rules.empty:
            return pd.DataFrame()

        dados = []
        for _, row in df_rules.iterrows():
            itemset = row["consequents"]
            
            if report_type == "success":
                itens_formatados = [item.split('_')[0] for item in itemset]
            else:
                itens_formatados = list(itemset)

            texto_itemset = "{" + ", ".join(itens_formatados) + "}"
            
            dados.append({
                "Tamanho do Itemset": row["consequent_size"],
                "Disciplinas Associadas": texto_itemset,
                "Suporte": f"{row['support']:.4f}",
                "Confianca": f"{row['confidence']:.4f}",
                "Lift": f"{row['lift']:.4f}"
            })
            
        return pd.DataFrame(dados)

    def generate_chord_html_string(self, rules: pd.DataFrame, temp_filename: str) -> str:
        """
        Gera o código HTML do diagrama de cordas (Chord Diagram) usando d3blocks.
        """
        pares = []
        for _, row in rules.iterrows():
            for ant in row['antecedents']:
                for con in row['consequents']:
                    pares.append({
                        'source': ant.split('_')[0],
                        'target': con.split('_')[0],
                        'weight': row['support']
                    })
                    
        df_pares = pd.DataFrame(pares)
        if df_pares.empty:
            return "<p style='color:red;'>Dados insuficientes para gerar o grafico de conexoes.</p>"

        # Agrupa os pesos das conexões. 
        df_pares = df_pares.groupby(['source', 'target'], as_index=False)['weight'].sum()
        df_pares = df_pares.sort_values(by='weight', ascending=False)

        temp_path = os.path.join(self.output_dir, f"{temp_filename}.html")

        try:
            d3 = D3Blocks(chart='chord')
            d3.chord(df_pares, filepath=temp_path, showfig=False)
            
            with open(temp_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            html_content = html_content.replace(
                '<body>', 
                '<body style="background-color: white; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 800px; border-radius: 8px;">'
            )
            
            os.remove(temp_path)
            return html_content
        except Exception as e:
            return f"<p>Erro na geracao do grafico: {e}</p>"
        
    def format_scenarios_to_dataframe(self, df_stats: pd.DataFrame, is_risk: bool = False) -> pd.DataFrame:
        """
        Formata o dicionário de estatísticas processado pelo recomendador 
        (Contrast Set Mining) para exibição tabular limpa.
        """
        try:
            if df_stats is None or df_stats.empty:
                return pd.DataFrame()

            dados = []
            for _, row in df_stats.iterrows():
                # Define qual itemset carregar dependendo do tipo de relatório (Sucesso ou Risco)
                if is_risk and row.get("risk_support", 0) > 0:
                    itemset = list(row["risk_itemset"])
                    sup = row["risk_support"]
                    conf = row["risk_conf"]
                else:
                    itemset = list(row["success_itemset"]) if row.get("success_support", 0) > 0 else list(row.get("risk_itemset", []))
                    sup = row.get("success_support", row.get("risk_support", 0))
                    conf = row.get("success_conf", row.get("risk_conf", 0))

                # Ordena para exibir matérias aprovadas primeiro
                itemset.sort(key=lambda x: ("_Reprovado" in x, x))
                texto_itemset = "{" + ", ".join(itemset) + "}"
                
                aprovacoes = sum(1 for i in itemset if "_Aprovado" in i)
                reprovacoes = sum(1 for i in itemset if "_Reprovado" in i)
                
                dados.append({
                    "Tamanho": row["size"],
                    "Grade Mapeada": texto_itemset,
                    "Aprov.": aprovacoes,
                    "Reprov.": reprovacoes,
                    "Suporte": f"{sup:.4f}",
                    "Confianca": f"{conf:.4f}"
                })
                
            return pd.DataFrame(dados)
        except Exception as e:
            return pd.DataFrame([{"Aviso": "Erro na formatação dos dados.", "Detalhe": str(e)}])