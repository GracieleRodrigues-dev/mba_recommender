# Sistema de Recomendação Acadêmica (MBA e Apriori)
Este repositório contém a implementação da Prova de Conceito (PoC) de um sistema de recomendação de disciplinas para o curso de Engenharia de Software da UDESC. O sistema utiliza técnicas de Market Basket Analysis (MBA) e o algoritmo Apriori para minerar padrões frequentes em registros históricos de matrículas. A ferramenta processa dados acadêmicos para sugerir grades curriculares otimizadas, visando maximizar a probabilidade de aprovação e mitigar riscos de sobrecarga e retenção escolar.

## Prova de Conceito (PoC)
A aplicação está hospedada e disponível para acesso remoto: [https://mba-recommender.streamlit.app/](https://mba-recommender.streamlit.app/)

## Arquitetura e Configuração
O sistema foi desenvolvido em Python e utiliza a biblioteca MLxtend para a mineração de regras de associação. A aplicação conta com um mecanismo de execução condicional controlado pela variável de ambiente `IS_CLOUD`. Em ambiente de desenvolvimento local, a variável deve ser configurada como `False` para permitir a leitura dos dados brutos. Em produção (nuvem), deve ser definida como `True` para utilizar o cache numérico serializado.

## Pré-requisitos
* Python 3.9 ou superior
* Git

## Instalação e Execução Local
### 1. Clonagem do Repositório
`git clone https://github.com/GracieleRodrigues-dev/mba_recommender.git`
`cd mba_recommender`

### 2. Configuração do Ambiente
**No Linux:**
`python3 -m venv venv`
`source venv/bin/activate`
**No Windows:**
`python -m venv venv`
`.\venv\Scripts\activate`

### 3. Instalação das Dependências
`pip install -r requirements.txt`

### 4. Execução da Aplicação
`python -m streamlit run src/app.py`

O sistema disponibilizará um endereço local (geralmente `http://localhost:8501`) para acesso via navegador.
## Estrutura dos Módulos
* `config.py`: Gestão de variáveis globais e modo de execução.
* `data_loader.py`: Rotinas de ingestão de dados.
* `data_preprocessor.py`: Limpeza e formatação transacional.
* `apriori_model.py`: Motor de mineração de regras (algoritmo Apriori).
* `recommender.py`: Motor lógico de recomendação baseado em classes (CAR).
* `visualizer.py`: Renderização de componentes gráficos.
* `app.py`: Orquestrador da interface (Streamlit).
