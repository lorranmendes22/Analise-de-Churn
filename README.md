

# Analise-de-Churn


## Sumário

1. [Importação dos Dados](#importacao-dos-dados)
2. [Pré-processamento e Transformação dos Dados](#pre-processamento-e-transformacao-dos-dados)
3. [Modelagem e Avaliação](#modelagem-e-avaliacao)
   - [Support Vector Classifier (SVC)](#support-vector-classifier-svc)
   - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
   - [Naive Bayes](#naive-bayes)
4. [Conclusão](#conclusao)

## Importação dos Dados

Os dados são importados diretamente de um arquivo CSV disponível publicamente. O dataset contém informações sobre clientes e se eles cancelaram ou não o contrato com a empresa.

```python
import pandas as pd

# Carregar o dataset
url = 'https://raw.githubusercontent.com/Gustavoice/Churn/main/Customer-Churn.csv'
dados = pd.read_csv(url)
dados.head()
```

## Pré-processamento e Transformação dos Dados

1. **Conversão de Variáveis Categóricas para Numéricas**: As variáveis categóricas são convertidas em variáveis binárias e dummies para facilitar o treinamento dos modelos.

2. **Balanceamento dos Dados**: Usamos a técnica de SMOTE (Synthetic Minority Over-sampling Technique) para balancear a classe de saída (`Churn`).

```python
from imblearn.over_sampling import SMOTE

# Separar features e target
X = dados_final.drop('Churn', axis=1)
y = dados_final['Churn']

# Balancear o dataset
smt = SMOTE(random_state=123)
X, y = smt.fit_resample(X, y)
```

## Modelagem e Avaliação

### Support Vector Classifier (SVC)

1. **Importação de Bibliotecas**: Utilizamos `SVC` para criar um modelo de classificador baseado em vetores de suporte.

2. **Normalização dos Dados**: Os dados são normalizados usando `StandardScaler`.

3. **Treinamento e Avaliação**: O modelo é treinado e avaliado com o conjunto de dados de teste.

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Normalizar dados
norm = StandardScaler()
X_normalizado = norm.fit_transform(X)

# Separar dados de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(X_normalizado, y, test_size=0.25, random_state=123)

# Criar e treinar o modelo
svc = SVC(gamma='auto', random_state=123)
svc.fit(x_treino, y_treino)

# Fazer previsões
previsoes_svc = svc.predict(x_teste)
print("Acurácia do SVC:", accuracy_score(y_teste, previsoes_svc) * 100, "%")
```

### K-Nearest Neighbors (KNN)

1. **Importação de Bibliotecas**: Utilizamos `KNeighborsClassifier` para criar um modelo baseado em vizinhos mais próximos.

2. **Treinamento e Avaliação**: O modelo é treinado e avaliado com o conjunto de dados de teste.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Criar e treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knn.fit(x_treino, y_treino)

# Fazer previsões
previsoes_knn = knn.predict(x_teste)
print("Acurácia do KNN:", accuracy_score(y_teste, previsoes_knn) * 100, "%")
```

### Naive Bayes

1. **Importação de Bibliotecas**: Utilizamos `MultinomialNB` para criar um modelo baseado no Teorema de Naive Bayes.

2. **Treinamento e Avaliação**: O modelo é treinado e avaliado com o conjunto de dados de teste.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Criar e treinar o modelo Naive Bayes
naive_bayes = MultinomialNB()
naive_bayes.fit(x_treino, y_treino)

# Fazer previsões
previsoes_nb = naive_bayes.predict(x_teste)
print("Acurácia do Naive Bayes:", accuracy_score(y_teste, previsoes_nb) * 100, "%")
```
