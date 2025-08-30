import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler


sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10,6)


adult_df = pd.read_csv("Bases/adult.csv")

adult_df.rename(columns={
    'age': 'Idade',
    'workclass': 'Tipo_de_Emprego',
    'fnlwgt': 'Peso_da_Amostra',
    'education': 'Educacao',
    'educational-num': 'Nivel_de_Educacao',
    'marital-status': 'Estado_Civil',
    'occupation': 'Profissao',
    'relationship': 'Relacao_Familiar',
    'race': 'Raca',
    'gender': 'Sexo',
    'capital-gain': 'Ganho_de_Capital',
    'capital-loss': 'Perda_de_Capital',
    'hours-per-week': 'Horas_por_Semana',
    'native-country': 'Pais_de_Origem',
    'income': 'Renda'
}, inplace=True)


adult_df['Sexo'] = adult_df['Sexo'].replace({'Male': 'Masculino', 'Female': 'Feminino'})
adult_df['Renda'] = adult_df['Renda'].replace({'>50K': 'Acima de 50K', '<=50K': 'Até 50K'})


# tratamento de valores ausentes ou inconsistentes
print("\n=== Verificando valores ausentes ===")
print(adult_df.isnull().sum())

# No Adult dataset
adult_df.replace(" ?", pd.NA, inplace=True)
print("\n=== Valores nulos (após substituição) ===")
print(adult_df.isnull().sum())

for col in adult_df.select_dtypes(include="object").columns:
    adult_df[col] = adult_df[col].fillna("Desconhecido")



# Detecção e tratamento de outliers

numeric_cols = adult_df.select_dtypes(include='number').columns
for col in ['Idade','Horas_por_Semana','Ganho_de_Capital','Perda_de_Capital']:
    Q1 = adult_df[col].quantile(0.25)
    Q3 = adult_df[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = adult_df[(adult_df[col] < limite_inferior) | (adult_df[col] > limite_superior)]
    print(f"Coluna {col}: {outliers.shape[0]} outliers detectados")
  # adult_df = adult_df[(adult_df[col] >= limite_inferior) & (adult_df[col] <= limite_superior)]


# conversão de variáveis categóricas para numéricas

le = LabelEncoder()
for col in adult_df.select_dtypes(include="object").columns:
    adult_df[col] = le.fit_transform(adult_df[col])

print("\n=== Dataset após conversão categórica ===")
print(adult_df.head())



# normalização dos dados (padrão Z-score)
scaler = StandardScaler()
adult_df[numeric_cols] = scaler.fit_transform(adult_df[numeric_cols])

print("\n=== Estatísticas das colunas numéricas normalizadas ===")
print(adult_df[numeric_cols].describe())


# crrelação entre variáveis numéricas

plt.figure()
sns.heatmap(adult_df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Mapa de correlação entre colunas numéricas (normalizadas)")
plt.show()
