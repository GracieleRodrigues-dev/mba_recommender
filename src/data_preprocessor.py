import pandas as pd
import numpy as np

class DataPreprocessor:
    @staticmethod
    def clean_enrollments(enrollments: pd.DataFrame) :
        # 1. Remover matrículas sem código de disciplina ou sem resultado final
        enrollments = enrollments.dropna(subset=["CodDis", "SituacaoResultadoDisciplina"])
        
        # 2. Filtrar apenas o curso de Engenharia de Software
        enrollments = enrollments[enrollments['CodCursoDaDisciplina'].str.contains('ESO-ESO', case=False, na=False)]

        # 3. Filtrar apenas ingressos até 2023/2 para evitar registros da nova grade
        enrollments = enrollments[enrollments['PeriodoIngresso'] <= '2023/2']
        
        # 4. Padronizar optativas de acordo com a grade ESO171
        # Optativas Tipo I (6ª fase)
        optativas_tipo1 = ['65DSE', '65DDM', '65PRW', '65DSM']
        enrollments['CodDis'] = enrollments['CodDis'].replace(optativas_tipo1, '65OPT1')
        
        # Optativas Tipo II (7ª fase)
        optativas_tipo2 = ['75DED', '75GCO', '75DJO', '75DSC']
        enrollments['CodDis'] = enrollments['CodDis'].replace(optativas_tipo2, '75OPT2')
        
        # Optativas Tipo III (8ª fase)
        optativas_tipo3 = ['85ESE', '85EAG', '85EAS']
        enrollments['CodDis'] = enrollments['CodDis'].replace(optativas_tipo3, '85OPT3')

        # 5. Remover disciplinas que fogem do padrão de cesta (Estágio, TCCs e Atividades Complementares)
        disciplinas_isoladas = ['85ESS', '75TCC1', '85TCC2', '85ATC']
        enrollments = enrollments[~enrollments['CodDis'].isin(disciplinas_isoladas)]

        # -------------------------------------------------------------------------
        # 6. Limpar sufixos de versão de grade curricular (ex: 15FES23 -> 15FES)
        # O regex r'\d{2}$' procura e apaga exatamente 2 dígitos no final da string.
        # Isso preserva matérias originais que terminam com 1 dígito, como '35PRO2' e '45PIN1'.
        # -------------------------------------------------------------------------
        enrollments['CodDis'] = enrollments['CodDis'].str.replace(r'\d{2}$', '', regex=True)

        # 7. Binarizar a Situação (Aprovado vs Reprovado)
        # Tudo que não for exatamente 'Aprovado' vira 'Reprovado' (Incompleto, Frequência, etc)
        enrollments['Status_Final'] = np.where(enrollments['SituacaoResultadoDisciplina'] == 'Aprovado', 'Aprovado', 'Reprovado')
        
        return enrollments

    @staticmethod
    def to_transactions(enrollments: pd.DataFrame) :
        # 1. Criar o Item de Transação concatenando a sigla com o resultado
        enrollments['Item_Transacao'] = enrollments['CodDis'].astype(str) + '_' + enrollments['Status_Final']

        # 2. Agrupar em cestas por aluno e período
        transactions = (
            enrollments.groupby(["MatriculaAluno", "PeriodoDaTurma"])["Item_Transacao"]
            .apply(list).tolist()
        )
        return transactions