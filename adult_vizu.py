import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder, StandardScaler

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Carregar a base
adult_df = pd.read_csv("Bases/adult.csv")

# Renomear colunas
adult_df.rename(columns={
    'age': 'Idade', 'gender': 'Sexo', 'workclass': 'Tipo_de_Emprego', 'fnlwgt': 'Peso_da_Amostra',
    'education': 'Educacao', 'educational-num': 'Nivel_de_Educacao', 'marital-status': 'Estado_Civil',
    'occupation': 'Profissao', 'relationship': 'Relacao_Familiar', 'race': 'Raca',
    'capital-gain': 'Ganho_de_Capital', 'capital-loss': 'Perda_de_Capital',
    'hours-per-week': 'Horas_por_Semana', 'native-country': 'Pais_de_Origem', 'income': 'Renda'
}, inplace=True)

# Ajustes nos valores de texto
adult_df['Sexo'] = adult_df['Sexo'].replace({'Male': 'Masculino', 'Female': 'Feminino'})
adult_df['Renda'] = adult_df['Renda'].replace({'>50K': 'Acima de 50K', '<=50K': 'Até 50K'})

# Percentual de zeros
percent_ganho_zero = (adult_df['Ganho_de_Capital'] == 0).mean() * 100
percent_perda_zero = (adult_df['Perda_de_Capital'] == 0).mean() * 100
print(f"% de zeros em Ganho_de_Capital: {percent_ganho_zero:.2f}%")
print(f"% de zeros em Perda_de_Capital: {percent_perda_zero:.2f}%")

# Resumo elementos únicos
quantidade_pessoas = len(adult_df)
elementos_unicos = []
for col in adult_df.columns:
    if adult_df[col].dtype == 'object':
        n = adult_df[col].dropna().loc[adult_df[col] != 'Desconhecido'].nunique()
    else:
        n = adult_df[col].dropna().nunique()
    elementos_unicos.append([col, n])
elementos_unicos = pd.DataFrame(elementos_unicos, columns=['Coluna', 'Respostas Diferentes'])

resumo = pd.concat([
    pd.DataFrame([['Total de Pessoas', quantidade_pessoas]], columns=['Coluna', 'Respostas Diferentes']),
    elementos_unicos
], ignore_index=True)

print("\nResumo da base de dados (desconsiderando campos vazios ou 'Desconhecido'):")
print(resumo)

# Tratar valores ausentes
adult_df.replace(" ?", pd.NA, inplace=True)
for col in adult_df.select_dtypes(include="object").columns:
    adult_df[col] = adult_df[col].fillna("Desconhecido")

# Garantir tipos numéricos
numeric_cols = ['Idade', 'Nivel_de_Educacao', 'Horas_por_Semana', 'Ganho_de_Capital', 'Perda_de_Capital']
adult_df[numeric_cols] = adult_df[numeric_cols].astype(float)

# Winsorization para tratar outliers
for col in numeric_cols:
    adult_df[col] = winsorize(adult_df[col], limits=[0.05, 0.05])

# LabelEncoder para variáveis categóricas (exceto Renda)
categorical_cols = ['Sexo', 'Raca', 'Estado_Civil', 'Tipo_de_Emprego', 'Pais_de_Origem', 'Educacao', 'Profissao']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    adult_df[col + "_Cod"] = le.fit_transform(adult_df[col])
    le_dict[col] = le
    # Imprimir mapeamento no terminal
    print(f"\nMapeamento dos códigos da coluna '{col}':")
    for code, category in enumerate(le.classes_):
        print(f"{code} -> {category}")

# Normalização das variáveis numéricas
scaler = StandardScaler()
adult_df[numeric_cols] = scaler.fit_transform(adult_df[numeric_cols])

# Converter Renda para numérico
adult_df['Renda_num'] = adult_df['Renda'].map({'Até 50K': 0, 'Acima de 50K': 1})

# Distribuição Geral das Variáveis (mantendo números nos gráficos)
plot_cols_cod = [col + "_Cod" for col in categorical_cols] + ['Renda_num']
for col in plot_cols_cod:
    plt.figure(figsize=(8, 5))
    sns.countplot(y=col, data=adult_df, palette='Set2', order=adult_df[col].value_counts().index)
    plt.title(f'Contagem de pessoas por {col}')
    plt.xlabel("Contagem")
    plt.ylabel(col)
    plt.show()

# Histogramas variáveis numéricas
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(adult_df[col], bins=30, kde=True, color='skyblue')
    plt.title(f'Distribuição de {col}')
    plt.xlabel(col)
    plt.ylabel("Contagem")
    plt.show()

# Análise de Renda por categoria (mostrando números nos gráficos)
for col in plot_cols_cod[:-1]:  # exclui Renda_num
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, hue='Renda_num', data=adult_df, palette='Set2',
                  order=adult_df[col].value_counts().index)
    plt.title(f'Renda por {col}')
    plt.xlabel(col)
    plt.ylabel("Contagem")
    plt.show()

# Boxplot apenas para Idade por Renda
plt.figure(figsize=(6, 4))
sns.boxplot(x='Renda_num', y='Idade', data=adult_df, palette='Pastel1')
plt.title('Idade por Renda (outliers tratados)')
plt.show()

# Correlações
plt.figure(figsize=(8, 6))
sns.heatmap(adult_df[numeric_cols + ['Renda_num']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlação entre variáveis numéricas e Renda")
plt.show()

# Feature Engineering
adult_df['Faixa_Etaria'] = pd.cut(adult_df['Idade'], bins=[-3, -1, 1], labels=['Jovem', 'Adulto', 'Idoso'])
adult_df['Horas_Categ'] = adult_df['Horas_por_Semana'].apply(
    lambda h: 'Part-time' if h < 35 else ('Full-time' if h <= 40 else 'Over-time'))
adult_df['Ganho_Liquido'] = adult_df['Ganho_de_Capital'] - adult_df['Perda_de_Capital']

# Proporção de Renda
print("Proporção de Renda Acima de 50K por Sexo:\n", adult_df.groupby('Sexo_Cod')['Renda_num'].mean())
print("\nProporção de Renda Acima de 50K por Escolaridade:\n",
      adult_df.groupby('Educacao_Cod')['Renda_num'].mean().sort_values(ascending=False))
print("\nProporção de Renda Acima de 50K por Ocupação:\n",
      adult_df.groupby('Profissao_Cod')['Renda_num'].mean().sort_values(ascending=False))

plt.figure(figsize=(12, 6))
sns.scatterplot(
    x=adult_df['Idade_original'],  # coluna original, crie antes de normalizar
    y='Estado_Civil_Cod',
    hue='Renda_num',
    data=adult_df,
    palette={0: 'skyblue', 1: 'salmon'},
    alpha=0.6
)
plt.title('Renda por Idade e Estado Civil')
plt.xlabel('Idade')
plt.ylabel('Estado Civil')
plt.legend(title='Renda', labels=['Até 50K', 'Acima de 50K'])
plt.yticks(ticks=range(len(le_dict['Estado_Civil'].classes_)), labels=le_dict['Estado_Civil'].classes_)
plt.show()
