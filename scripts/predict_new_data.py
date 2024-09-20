{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "c5ed4146",
   "metadata": {},
   "source": [
    "# predict_new_data.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "ce82ce69",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importação das bibliotecas necessárias\n",
    "import pandas as pd\n",
    "import joblib\n",
    "import os\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "6886c53e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Função para criar as features como foi definido no projeto principal\n",
    "def make_features(data, max_lag, rolling_mean_size):\n",
    "    data['month'] = data.index.month\n",
    "    data['day'] = data.index.day\n",
    "    data['dayofweek'] = data.index.dayofweek\n",
    "    data['hour'] = data.index.hour\n",
    "    for lag in range(1, max_lag + 1):\n",
    "        data[f'lag_{lag}'] = data['num_orders'].shift(lag)\n",
    "    data['rolling_mean'] = data['num_orders'].shift().rolling(rolling_mean_size).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "012a6648",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Função para carregar o modelo já treinado\n",
    "def load_model(filepath):\n",
    "    return joblib.load(filepath)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "248665bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Função para gerar novos dados sintéticos (apenas para testes)\n",
    "def generate_new_data(filepath):\n",
    "    # Gerar uma série de datas futuras\n",
    "    date_range = pd.date_range(start=\"2018-09-01\", end=\"2018-09-15\", freq=\"H\")\n",
    "    \n",
    "    # Criar um dataframe com datas e preencher com dados sintéticos (números de pedidos)\n",
    "    new_data = pd.DataFrame({\n",
    "        'datetime': date_range,\n",
    "        'num_orders': np.random.randint(0, 120, size=len(date_range))  # Gerando pedidos aleatórios\n",
    "    })\n",
    "    \n",
    "    # Salvar os novos dados como CSV\n",
    "    new_data.to_csv(filepath, index=False)\n",
    "    \n",
    "    print(\"Arquivo de novos dados gerado com sucesso.\")\n",
    "    return new_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "4fc1a7dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Função principal para fazer previsões com novos dados\n",
    "def predict_new_data(new_data_filepath, model_filepath):\n",
    "    # Carregar os dados novos\n",
    "    new_data = pd.read_csv(new_data_filepath, parse_dates=['datetime'], index_col='datetime')\n",
    "    \n",
    "    # Ordenar os dados e criar as features\n",
    "    new_data = new_data.sort_values(by='datetime')\n",
    "    make_features(new_data, max_lag=3, rolling_mean_size=5)\n",
    "    \n",
    "    # Remover valores nulos\n",
    "    new_data = new_data.dropna()\n",
    "    \n",
    "    # Separar as features (sem a coluna alvo 'num_orders')\n",
    "    X_new = new_data.drop('num_orders', axis=1, errors='ignore')  # Não deve dar erro se não houver a coluna 'num_orders'\n",
    "    \n",
    "    # Carregar o modelo\n",
    "    pipeline = load_model(model_filepath)\n",
    "    \n",
    "    # Fazer previsões\n",
    "    predictions = pipeline.predict(X_new)\n",
    "    \n",
    "    # Adicionar as previsões ao dataframe original\n",
    "    new_data['predictions'] = predictions\n",
    "    \n",
    "    # Retornar o dataframe com as previsões\n",
    "    return new_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "4ae2db00",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Arquivo de novos dados gerado com sucesso.\n"
     ]
    }
   ],
   "source": [
    "# Exemplo de uso\n",
    "if __name__ == \"__main__\":\n",
    "    # Definir o caminho do novo conjunto de dados e do modelo treinado\n",
    "    new_data_filepath = r'D:\\GitHub\\taxi_driver_allocation_optimization\\data/novos_dados_taxi.csv'  # Caminho do arquivo de dados\n",
    "    model_filepath = r'D:\\GitHub\\taxi_driver_allocation_optimization\\models/decision_tree_pipeline.pkl'  # Caminho do modelo salvo\n",
    "    \n",
    "    # Gerar novos dados sintéticos (apenas para teste)\n",
    "    generate_new_data(new_data_filepath)\n",
    "    \n",
    "    # Fazer as previsões\n",
    "    predicted_data = predict_new_data(new_data_filepath, model_filepath)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "89be25e5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Previsões das primeiras linhas:\n",
      "\n",
      "                     num_orders  month  day  dayofweek  hour  predictions\n",
      "datetime                                                                 \n",
      "2018-09-01 05:00:00          29      9    1          5     5    29.000000\n",
      "2018-09-01 06:00:00          18      9    1          5     6    22.826531\n",
      "2018-09-01 07:00:00          55      9    1          5     7    15.279070\n",
      "2018-09-01 08:00:00          21      9    1          5     8    42.000000\n",
      "2018-09-01 09:00:00         117      9    1          5     9    18.685714\n"
     ]
    }
   ],
   "source": [
    "# Exibir as primeiras linhas do dataframe com as previsões de forma mais simples e organizada\n",
    "print(\"\\nPrevisões das primeiras linhas:\\n\")\n",
    "print(predicted_data[['num_orders', 'month', 'day', 'dayofweek', 'hour', 'predictions']].head().to_string(index=True))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aac231e2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": true
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
