{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Análise do Dataset Titanic - Tratamento e Análise de Dados com Pandas\n",
    "\n",
    "Este notebook realiza:\n",
    "1. Descrição completa dos dados\n",
    "2. Tratamento de qualidade dos dados\n",
    "3. 5 perguntas de análise respondidas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importação das bibliotecas necessárias\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Configurações de exibição\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.max_rows', 50)\n",
    "pd.set_option('display.width', None)\n",
    "\n",
    "# Carregar o dataset\n",
    "df = pd.read_csv('Base de Dados Titanic.csv')\n",
    "print(\"Dataset carregado com sucesso!\")\n",
    "print(f\"Formato do dataset: {df.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. DESCRIÇÃO DOS DADOS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"=\" * 70)\n",
    "print(\"1. DESCRIÇÃO DOS DADOS\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "# Informações básicas\n",
    "print(\"\\n--- Informações Gerais ---\")\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"\\n--- Primeiras 5 linhas ---\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"\\n--- Últimas 5 linhas ---\")\n",
    "df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"\\n--- Estatísticas Descritivas (Variáveis Numéricas) ---\")\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"\\n--- Estatísticas Descritivas (Variáveis Categóricas) ---\")\n",
    "df.describe(include=['object'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.1 Análise de Valores Nulos\n",
    "print(\"\\n--- Análise de Valores Nulos ---\")\n",
    "missing_data = pd.DataFrame({\n",
    "    'Coluna': df.columns,\n",
    "    'Qtd Nulos': df.isnull().sum().values,\n",
    "    'Percentual Nulos (%)': (df.isnull().sum() / len(df) * 100).values\n",
    "})\n",
    "missing_data = missing_data[missing_data['Qtd Nulos'] > 0].sort_values('Percentual Nulos (%)', ascending=False)\n",
    "missing_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.2 Análise de Valores Distintos\n",
    "print(\"\\n--- Contagem de Valores Únicos por Coluna ---\")\n",
    "unique_counts = pd.DataFrame({\n",
    "    'Coluna': df.columns,\n",
    "    'Valores Únicos': df.nunique().values,\n",
    "    'Tipo de Dado': df.dtypes.values\n",
    "})\n",
    "unique_counts = unique_counts.sort_values('Valores Únicos')\n",
    "unique_counts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.3 Distribuição das Variáveis Categóricas\n",
    "print(\"\\n--- Distribuição das Variáveis Categóricas ---\")\n",
    "categorical_columns = ['Survived', 'Pclass', 'Sex', 'Embarked']\n",
    "for col in categorical_columns:\n",
    "    if col in df.columns:\n",
    "        print(f\"\\n{col}:\")\n",
    "        print(df[col].value_counts(dropna=False))\n",
    "        print(f\"\\nPercentuais:\")\n",
    "        print(df[col].value_counts(normalize=True, dropna=False) * 100)\n",
    "        print(\"-\" * 30)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1.4 Estatísticas Detalhadas das Variáveis Numéricas\n",
    "print(\"\\n--- Estatísticas Detalhadas das Variáveis Numéricas ---\")\n",
    "numerical_columns = ['Age', 'SibSp', 'Parch', 'Fare']\n",
    "for col in numerical_columns:\n",
    "    if col in df.columns:\n",
    "        print(f\"\\n{col}:\")\n",
    "        print(f\"  Mínimo: {df[col].min():.2f}\")\n",
    "        print(f\"  Máximo: {df[col].max():.2f}\")\n",
    "        print(f\"  Média: {df[col].mean():.2f}\")\n",
    "        print(f\"  Mediana: {df[col].median():.2f}\")\n",
    "        print(f\"  Desvio Padrão: {df[col].std():.2f}\")\n",
    "        print(f\"  Variância: {df[col].var():.2f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. TRATAMENTO DE QUALIDADE DOS DADOS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"\\n\" + \"=\" * 70)\n",
    "print(\"2. TRATAMENTO DE QUALIDADE DOS DADOS\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "# Criar uma cópia para tratamento\n",
    "df_clean = df.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2.1 Verificar e remover colunas constantes (apenas um valor)\n",
    "print(\"\\n--- Verificando Colunas Constantes ---\")\n",
    "constant_columns = []\n",
    "for col in df_clean.columns:\n",
    "    if df_clean[col].nunique() == 1:\n",
    "        constant_columns.append(col)\n",
    "        print(f\"Coluna '{col}' tem apenas um valor: {df_clean[col].iloc[0]}\")\n",
    "\n",
    "if constant_columns:\n",
    "    df_clean = df_clean.drop(columns=constant_columns)\n",
    "    print(f\"\\n✓ Colunas constantes removidas: {constant_columns}\")\n",
    "else:\n",
    "    print(\"✓ Nenhuma coluna constante encontrada.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2.2 Tratamento de Valores Nulos\n",
    "print(\"\\n--- Tratamento de Valores Nulos ---\")\n",
    "\n",