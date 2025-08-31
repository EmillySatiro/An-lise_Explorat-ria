import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

adult_df = pd.read_csv("Bases/adult.csv")


adult_df.rename(columns={
    'age': 'Idade', 'gender': 'Sexo', 'workclass': 'Tipo_de_Emprego', 'fnlwgt': 'Peso_da_Amostra',
    'education': 'Educacao', 'educational-num': 'Nivel_de_Educacao', 'marital-status': 'Estado_Civil',
    'occupation': 'Profissao', 'relationship': 'Relacao_Familiar', 'race': 'Raca',
    'capital-gain': 'Ganho_de_Capital', 'capital-loss': 'Perda_de_Capital',
    'hours-per-week': 'Horas_por_Semana', 'native-country': 'Pais_de_Origem', 'income': 'Renda'
}, inplace=True)

adult_df['Sexo'] = adult_df['Sexo'].replace(
    {'Male': 'Masculino', 'Female': 'Feminino'})
adult_df['Renda'] = adult_df['Renda'].replace(
    {'>50K': 'Acima de 50K', '<=50K': 'Até 50K'})

quantidade_pessoas = len(adult_df)
elementos_unicos = []
for col in adult_df.columns:
    #desconsidera 'Desconhecido' e valores nulos
    if adult_df[col].dtype == 'object':
        n = adult_df[col].dropna().loc[adult_df[col] != 'Desconhecido'].nunique()
    else:
        n = adult_df[col].dropna().nunique()
    elementos_unicos.append([col, n])
elementos_unicos = pd.DataFrame(elementos_unicos, columns=['Coluna', 'Quantidade de Elementos Únicos'])

resumo = pd.concat([
    pd.DataFrame([['Total de Pessoas', quantidade_pessoas]], columns=['Coluna', 'Quantidade de Elementos Únicos']),
    elementos_unicos
], ignore_index=True)

print("\nResumo da base de dados (desconsiderando campos vazios ou 'Desconhecido'):")
print(resumo)

# Tratar valores ausentes
adult_df.replace(" ?", pd.NA, inplace=True)
for col in adult_df.select_dtypes(include="object").columns:
    adult_df[col] = adult_df[col].fillna("Desconhecido")

# Garantir tipos numéricos
numeric_cols = ['Idade', 'Nivel_de_Educacao',
                'Horas_por_Semana', 'Ganho_de_Capital', 'Perda_de_Capital']
adult_df[numeric_cols] = adult_df[numeric_cols].astype(float)

# Winsorization para tratar outliers
for col in numeric_cols:
    adult_df[col] = winsorize(adult_df[col], limits=[0.05, 0.05])

# Converter renda para numérico
adult_df['Renda_num'] = adult_df['Renda'].map(
    {'Até 50K': 0, 'Acima de 50K': 1})

# Distribuição Geral das Variáveis

categorical_cols = ['Sexo', 'Raca', 'Estado_Civil', 'Tipo_de_Emprego',
                    'Pais_de_Origem', 'Educacao', 'Profissao', 'Renda']
for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(y=col, data=adult_df,
                  order=adult_df[col].value_counts().index, palette='Set2')
    plt.title(f'Contagem de pessoas por {col}')
    plt.xlabel("Contagem")
    plt.ylabel(col)
    plt.show()

# Histogramas variáveis numéricas
for col in ['Idade', 'Nivel_de_Educacao', 'Horas_por_Semana', 'Ganho_de_Capital', 'Perda_de_Capital']:
    plt.figure(figsize=(8, 5))
    sns.histplot(adult_df[col], bins=30, kde=True, color='skyblue')
    plt.title(f'Distribuição de {col}')
    plt.xlabel(col)
    plt.ylabel("Contagem")
    plt.show()


# Análise de Renda
for col in ['Sexo', 'Educacao', 'Profissao', 'Raca', 'Estado_Civil']:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, hue='Renda', data=adult_df, palette='Set2',
                  order=adult_df[col].value_counts().index)
    plt.title(f'Renda por {col}')
    plt.xlabel(col)
    plt.ylabel("Contagem")
    plt.xticks(rotation=45)
    plt.show()

# Boxplot apenas para Idade por Renda
plt.figure(figsize=(6, 4))
sns.boxplot(x='Renda', y='Idade', data=adult_df, palette='Pastel1')
plt.title('Idade por Renda (outliers tratados)')
plt.show()

# Correlações

plt.figure(figsize=(8, 6))
sns.heatmap(adult_df[numeric_cols+['Renda_num']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlação entre variáveis numéricas e Renda")
plt.show()


# Interações Entre Variáveis

# Profissão x Sexo
plt.figure(figsize=(12, 6))
sns.countplot(x='Profissao', hue='Sexo', data=adult_df, palette='Set2',
              order=adult_df['Profissao'].value_counts().index)
plt.title("Ocupação por Sexo")
plt.xticks(rotation=45)
plt.show()

# Profissão x Escolaridade
plt.figure(figsize=(12, 6))
sns.countplot(x='Profissao', hue='Educacao', data=adult_df, palette='Set3',
              order=adult_df['Profissao'].value_counts().index)
plt.title("Ocupação por Escolaridade")
plt.xticks(rotation=45)
plt.show()

# Raça x Renda
plt.figure(figsize=(6, 4))
sns.countplot(x='Raca', hue='Renda', data=adult_df, palette='Set1')
plt.title("Distribuição de Renda por Raça")
plt.show()

# Feature Engineering

# Faixa etária
adult_df['Faixa_Etaria'] = pd.cut(
    adult_df['Idade'], bins=[0, 25, 60, 100], labels=['Jovem', 'Adulto', 'Idoso'])

# Categorizar horas de trabalho
adult_df['Horas_Categ'] = adult_df['Horas_por_Semana'].apply(
    lambda h: 'Part-time' if h < 35 else ('Full-time' if h <= 40 else 'Over-time'))

# Ganhos líquidos
adult_df['Ganho_Liquido'] = adult_df['Ganho_de_Capital'] - \
    adult_df['Perda_de_Capital']


print("Proporção de Renda Acima de 50K por Sexo:\n",
      adult_df.groupby('Sexo')['Renda_num'].mean())
print("\nProporção de Renda Acima de 50K por Escolaridade:\n", adult_df.groupby(
    'Educacao')['Renda_num'].mean().sort_values(ascending=False))
print("\nProporção de Renda Acima de 50K por Ocupação:\n", adult_df.groupby(
    'Profissao')['Renda_num'].mean().sort_values(ascending=False))
