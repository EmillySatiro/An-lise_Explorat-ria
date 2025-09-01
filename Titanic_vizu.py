import pandas as pd

titanic_df = pd.read_csv("Bases/Titanic-Dataset.csv")

titanic_df.rename(columns={
    'PassengerId': 'ID_Passageiro',
    'Survived': 'Sobreviveu',
    'Pclass': 'Classe',
    'Name': 'Nome',
    'Sex': 'Sexo',
    'Age': 'Idade',
    'SibSp': 'Irmaos_Conjuges',
    'Parch': 'Pais_Filhos',
    'Ticket': 'Bilhete',
    'Fare': 'Tarifa',
    'Cabin': 'Cabine',
    'Embarked': 'Embarque'
}, inplace=True)

import matplotlib.pyplot as plt
# Preencher valores ausentes de idade apenas para quem respondeu (ignorando quem não respondeu)
# Aqui, vamos ignorar quem não respondeu (ou seja, manter os NaNs e não preencher)
# Remover outliers de idade usando o método IQR apenas para quem respondeu

idades_respondidas = titanic_df['Idade'].dropna()
q1 = idades_respondidas.quantile(0.25)
q3 = idades_respondidas.quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

# Filtrar o DataFrame para remover outliers apenas entre quem respondeu
titanic_sem_outliers = titanic_df[
    (titanic_df['Idade'].isna()) | 
    ((titanic_df['Idade'] >= limite_inferior) & (titanic_df['Idade'] <= limite_superior))
]

# Boxplot de idade sem separar por sexo e sem outliers, apenas para quem respondeu
plt.figure(figsize=(6,6))
titanic_sem_outliers['Idade'].dropna().to_frame().boxplot(column='Idade')
plt.title('Idade Passageiros')
plt.suptitle('')
plt.ylabel('Idade')
plt.tight_layout()
plt.show()


# Tratamento dos valores pagos pelo pessoal da primeira classe

# Selecionar tarifas da primeira classe e remover valores ausentes
valores_primeira_classe = titanic_df[titanic_df['Classe'] == 1]['Tarifa'].dropna()

# Remover outliers usando o método IQR
q1_tarifa = valores_primeira_classe.quantile(0.25)
q3_tarifa = valores_primeira_classe.quantile(0.75)
iqr_tarifa = q3_tarifa - q1_tarifa
limite_inferior_tarifa = q1_tarifa - 1.5 * iqr_tarifa
limite_superior_tarifa = q3_tarifa + 1.5 * iqr_tarifa

valores_primeira_classe_sem_outliers = valores_primeira_classe[
    (valores_primeira_classe >= limite_inferior_tarifa) & (valores_primeira_classe <= limite_superior_tarifa)
]

print("Valores pagos pelo pessoal da primeira classe (em euro, sem outliers):")
print(valores_primeira_classe_sem_outliers)

# Boxplot dos valores pagos pelo pessoal da primeira classe (em euro, sem outliers)
plt.figure(figsize=(6,6))
plt.boxplot(valores_primeira_classe_sem_outliers)
plt.title('Boxplot da Tarifa (€) - Primeira Classe')
plt.ylabel('Tarifa (€)')
plt.tight_layout()
plt.show()

# Gráfico de barras: Sobreviventes e não sobreviventes por classe
sobreviventes_por_classe = titanic_df.groupby(['Classe', 'Sobreviveu']).size().unstack(fill_value=0)

sobreviventes_por_classe.plot(kind='bar', figsize=(8,6))
plt.title('Sobreviventes e Não Sobreviventes por Classe')
plt.xlabel('Classe')
plt.ylabel('Número de Passageiros')
plt.legend(['Não Sobreviveu', 'Sobreviveu'], title='Status')
plt.tight_layout()
plt.show()



# Gráfico de barras: Sobreviventes por classe e sexo
sobreviventes_classe_sexo = titanic_df[titanic_df['Sobreviveu'] == 1].groupby(['Classe', 'Sexo']).size().unstack(fill_value=0)

print("Sobreviventes por classe e sexo:")
print(sobreviventes_classe_sexo)

sobreviventes_classe_sexo.plot(kind='bar', figsize=(8,6))
plt.title('Sobreviventes por Classe e Sexo')
plt.xlabel('Classe')
plt.ylabel('Número de Sobreviventes')
plt.legend(title='Sexo')
plt.tight_layout()
plt.show()

# Gráfico de barras: Quantidade de pessoas por classe
quantidade_por_classe = titanic_df['Classe'].value_counts().sort_index()
quantidade_por_classe.plot(kind='bar', color='teal')
plt.title('Quantidade de Pessoas por Classe')
plt.xlabel('Classe')
plt.ylabel('Quantidade de Passageiros')
plt.tight_layout()
plt.show()

# Gráfico de barras: Não sobreviventes por classe e sexo
nao_sobreviventes_classe_sexo = titanic_df[titanic_df['Sobreviveu'] == 0].groupby(['Classe', 'Sexo']).size().unstack(fill_value=0)

print("Quantidade de não sobreviventes por classe e sexo:")
print(nao_sobreviventes_classe_sexo)

# Definir cores: homem = azul, mulher = laranja
cores = ['royalblue', 'orange']
nao_sobreviventes_classe_sexo.plot(kind='bar', figsize=(8,6), color=cores)
plt.title('Não Sobreviventes por Classe e Sexo')
plt.xlabel('Classe')
plt.ylabel('Número de Não Sobreviventes')
plt.legend(title='Sexo')
plt.tight_layout()
plt.show()

# Gráfico de barras: Quantidade de pessoas para cada prefixo do bilhete

# Extrair prefixo (ou 'Sem Prefixo' se não houver)
def extrair_prefixo(bilhete):
    partes = str(bilhete).split()
    return partes[0] if not partes[0].isdigit() else 'Sem Prefixo'

titanic_df['Prefixo_Bilhete'] = titanic_df['Bilhete'].apply(extrair_prefixo)

contagem_por_prefixo = titanic_df['Prefixo_Bilhete'].value_counts()

contagem_por_prefixo.plot(kind='bar', color='cornflowerblue')
plt.title('Quantidade de Pessoas por Prefixo do Bilhete')
plt.xlabel('Prefixo do Bilhete')
plt.ylabel('Quantidade de Pessoas')
plt.tight_layout()
plt.show()

# Calcular o tamanho das famílias (incluindo o próprio passageiro)
titanic_df['Tamanho_Familia'] = titanic_df['Irmaos_Conjuges'] + titanic_df['Pais_Filhos'] 

# Exibir estatísticas descritivas do tamanho das famílias
print("Estatísticas do tamanho das famílias:")
print(titanic_df['Tamanho_Familia'].describe())

# Gráfico de barras: Distribuição do tamanho das famílias
titanic_df['Tamanho_Familia'].value_counts().sort_index().plot(kind='bar', color='mediumseagreen')
plt.title('Distribuição do Tamanho das Famílias')
plt.xlabel('Tamanho da Família')
plt.ylabel('Quantidade de Passageiros')
plt.tight_layout()
plt.show()

# Distribuição por família e quantidade delas em cada classe

# Criar um identificador de família usando o sobrenome (última palavra antes da vírgula no nome)
titanic_df['Sobrenome'] = titanic_df['Nome'].str.extract(r'(^[^,]+)')

# Contar famílias distintas por classe
familias_por_classe = titanic_df.groupby('Classe')['Sobrenome'].nunique()

print("Quantidade de famílias distintas por classe:")
print(familias_por_classe)


# Exibir todos os títulos únicos presentes na coluna 'Nome'
# Extrair títulos dos nomes
titanic_df['Titulo'] = titanic_df['Nome'].str.extract(r',\s*([^\.]*)\.')

# Ver títulos únicos encontrados
titulos_unicos = titanic_df['Titulo'].unique()
print("Títulos únicos encontrados nos nomes dos passageiros:")
print(titulos_unicos)

# Dicionário para renomear títulos para algo mais compreensível
mapa_titulos = {
    'Mr': 'Senhor',
    'Mrs': 'Senhora',
    'Miss': 'Senhorita',
    'Ms': 'Senhorita',
    'Mlle': 'Senhorita',
    'Mme': 'Senhora',
    'Master': 'Menino',
    'Dr': 'Doutor',
    'Rev': 'Reverendo',
    'Col': 'Militar',
    'Major': 'Militar',
    'Capt': 'Militar',
    'Sir': 'Nobre',
    'Lady': 'Nobre',
    'Don': 'Nobre',
    'Jonkheer': 'Nobre',
    'the Countess': 'Nobre',
    'Dona': 'Nobre'
}

# Renomear os títulos agrupando por significado
titanic_df['Titulo_Traduzido'] = titanic_df['Titulo'].map(mapa_titulos).fillna(titanic_df['Titulo'])

# Ver títulos traduzidos únicos
titulos_traduzidos_unicos = titanic_df['Titulo_Traduzido'].unique()
print("Títulos traduzidos agrupados:")
print(titulos_traduzidos_unicos)

# Gráfico de barras: Quantidade de pessoas por título traduzido agrupado
titanic_df['Titulo_Traduzido'].value_counts().plot(kind='bar', color='slateblue')
plt.title('Quantidade de Pessoas por Título (Agrupado)')
plt.xlabel('Título')
plt.ylabel('Quantidade de Passageiros')
plt.tight_layout()
plt.show()


# Quantidade de pessoas por título em cada classe
titulos_por_classe = titanic_df.groupby(['Classe', 'Titulo_Traduzido']).size().unstack(fill_value=0)
print("Quantidade de pessoas por título em cada classe:")
print(titulos_por_classe)

# Gráfico de barras agrupadas: pessoas por título em cada classe
ax = titulos_por_classe.plot(kind='bar', figsize=(10,7), colormap='tab20')
plt.title('Quantidade de Pessoas por Título em Cada Classe')
plt.xlabel('Classe')
plt.ylabel('Quantidade de Passageiros')
plt.legend(title='Título', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.yticks(range(0, int(titulos_por_classe.values.max()) + 11, 10))
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Adiciona linhas de grade horizontais
plt.tight_layout()
plt.show()