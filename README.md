# English Version
![GitHub](https://img.shields.io/github/license/willianadb/gold_purification_prediction)

## Taxi Driver Allocation Optimization

### Project Overview

This project aims to develop a predictive model for taxi demand and optimize the allocation of taxi drivers based on historical demand data. The model uses machine learning techniques to predict the number of taxi requests in future time intervals and helps optimize driver allocation.

### Project Structure

- **taxi_driver_allocation_optimization/**
  - **data/**
    - `taxi.csv` - Original dataset
    - `novos_dados_taxi.csv` - Synthetic data for predictions
  - **models/**
    - `decision_tree_pipeline.pkl` - Trained Decision Tree model
  - **notebooks/**
    - `taxi_driver_allocation_optimization.ipynb` - Main notebook with analysis, training, and evaluation
  - **scripts/**
    - `train_model.py` - Script for training the model
    - `predict_new_data.py` - Script for generating new data and predictions
  - `README.md` - Project overview (this file)
  - `requirements.txt` - Project dependencies

### Dataset

The project uses a dataset of historical taxi demand data. The dataset contains the following columns:

- `datetime`: The timestamp of the demand entry.
- `num_orders`: The number of taxi requests during that time period.

### Model

The model used in this project is a Decision Tree Regressor trained to predict the number of taxi requests based on features like the day, month, hour, and previous demand (lag features).

**Features**

- `month`: Month
- `day`: Day
- `dayofweek`: Day of the week
- `hour`: Hour of the day
- `lag_1`, `lag_2`, `lag_3`: Lag values (previous demands)
- `rolling_mean`: Rolling mean of previous demands

### Scripts

1. **train_model.py**  
   This script trains the Decision Tree model using the dataset and saves it as a `.pkl` file.

2. **predict_new_data.py**  
   This script generates new synthetic data and makes predictions using the trained model.

### Instructions

1. **Install Dependencies**  
   Install the project dependencies by running the following command:

    ```bash
    pip install -r requirements.txt
    ```

2. **Train the Model**  
   To retrain the model, run the `train_model.py` script:

    ```bash
    python scripts/train_model.py
    ```

3. **Generate Predictions**  
   To generate predictions on new data, run the `predict_new_data.py` script:

    ```bash
    python scripts/predict_new_data.py
    ```

4. **Open the Notebook**  
   You can open the Jupyter notebook `taxi_driver_allocation_optimization.ipynb` to view the complete analysis and modeling process.

### Results

The Decision Tree model achieved the best performance with an RMSE of 22.66, making it the most effective model for predicting taxi demand.

### Future Improvements

- **Hyperparameter Tuning**: Try fine-tuning the hyperparameters of the model to achieve better results.
- **Other Models**: Explore other machine learning models such as Gradient Boosting or Neural Networks.
- **Real-Time Data**: Integrate real-time taxi demand data for more practical applications.

# Versão em Português
## Otimização de Alocação de Motoristas de Táxi
### Visão Geral do Projeto

Este projeto visa desenvolver um modelo preditivo para a demanda de táxis e otimizar a alocação de motoristas com base em dados históricos. O modelo usa técnicas de aprendizado de máquina para prever o número de pedidos de táxi em intervalos de tempo futuros, ajudando na otimização da alocação de motoristas.

### Estrutura do Projeto

- **taxi_driver_allocation_optimization/**
  - **data/**
    - `taxi.csv` - Conjunto de dados original
    - `novos_dados_taxi.csv` - Dados sintéticos para previsões
  - **models/**
    - `decision_tree_pipeline.pkl` - Modelo treinado de Árvore de Decisão
  - **notebooks/**
    - `taxi_driver_allocation_optimization.ipynb` - Notebook principal com análise, treinamento e avaliação
  - **scripts/**
    - `train_model.py` - Script para treinar o modelo
    - `predict_new_data.py` - Script para gerar novos dados e previsões
  - `README.md` - Visão geral do projeto (este arquivo)
  - `requirements.txt` - Dependências do projeto

### Conjunto de Dados

O projeto utiliza um conjunto de dados de demanda histórica de táxis. O conjunto de dados contém as seguintes colunas:

- `datetime`: Data e hora do registro de demanda.
- `num_orders`: O número de pedidos de táxi durante aquele período.

### Modelo

O modelo utilizado neste projeto é um Regressor de Árvore de Decisão, treinado para prever o número de pedidos de táxi com base em variáveis como o dia, mês, hora e demanda anterior (lag).

**Features**

- `month`: Mês
- `day`: Dia
- `dayofweek`: Dia da semana
- `hour`: Hora do dia
- `lag_1`, `lag_2`, `lag_3`: Valores de atraso (demanda anterior)
- `rolling_mean`: Média móvel dos últimos 5 períodos

### Scripts

1. **train_model.py**  
   Este script treina o modelo de Árvore de Decisão usando o conjunto de dados e o salva como um arquivo `.pkl`.

2. **predict_new_data.py**  
   Este script gera novos dados sintéticos e faz previsões usando o modelo treinado.

### Instruções

1. **Instalar Dependências**  
   Instale as dependências do projeto rodando o comando:

    ```bash
    pip install -r requirements.txt
    ```

2. **Treinar o Modelo**  
   Para treinar novamente o modelo, execute o script `train_model.py`:

    ```bash
    python scripts/train_model.py
    ```

3. **Gerar Previsões**  
   Para gerar previsões em novos dados, execute o script `predict_new_data.py`:

    ```bash
    python scripts/predict_new_data.py
    ```

4. **Abrir o Notebook**  
   Você pode abrir o notebook Jupyter `taxi_driver_allocation_optimization.ipynb` para ver a análise completa e o processo de modelagem.

### Resultados

O modelo de Árvore de Decisão obteve o melhor desempenho com um RMSE de 22,66, sendo o modelo mais eficaz para prever a demanda de táxi.

### Melhorias Futuras

- **Ajuste de Hiperparâmetros**: Tentar ajustar os hiperparâmetros do modelo para obter melhores resultados.
- **Outros Modelos**: Explorar outros modelos de aprendizado de máquina, como Gradient Boosting ou Redes Neurais.
- **Dados em Tempo Real**: Integrar dados de demanda de táxi em tempo real para aplicações mais práticas.
