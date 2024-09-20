{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8ca2c1ca",
   "metadata": {},
   "source": [
    "# train_model.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "2f8610bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "7c7a841b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Caminho para o modelo já salvo\n",
    "model_dir = r'D:\\GitHub\\taxi_driver_allocation_optimization\\modelos'\n",
    "model_path = os.path.join(model_dir, 'decision_tree_pipeline.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "3f5ebd13",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Carregar o modelo já treinado\n",
    "pipeline = joblib.load(model_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "1426901e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Modelo carregado com sucesso.\n"
     ]
    }
   ],
   "source": [
    "print(\"Modelo carregado com sucesso.\")"
   ]
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
