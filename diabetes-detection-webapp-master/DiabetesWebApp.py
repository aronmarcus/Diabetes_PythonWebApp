# Aplicação em Python para detectar se uma pessoa tem diabetes ou não, usando Machine Learning!

#Importando os pacotes Python
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Cabeçalho
st.header('                 DIABETES - PYTHON WEB APP')

#Título e subtítulo
st.write("""
# DETECÇÃO DE DIABETES
# Aplicação em Python para detectar se uma pessoa tem diabetes ou não, usando Machine Learning!
# """)

#Exibir imagem
image = Image.open('img/neural_networks.gif')
st.image(image, caption='Diabetes Neural Networks', use_column_width=True)

#Carregar dataset csv
df = pd.read_csv('Dataset/diabetes.csv')

#Definir subheader de informações dos dados
# st.subheader('Data Information')
#
# #Exibir tabelas
# st.dataframe(df)
#
# #Exibir estatísticas
# st.write(df.describe())
#
# #Exibir gráfico
#chart = st.bar_chart(df)

#Dividir os dados em variáveis X independentes e Y dependentes
X = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values

#Dividir os dados para treinamento e testes
X_train ,X_test, y_train ,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#Obter os dados de entrada do usuário
def get_user_input():
    pregnancies = st.sidebar.slider('Gravidez',0,17,1)
    glucose = st.sidebar.slider('Glicose',0,199,117)
    blood_pressure = st.sidebar.slider('Pressão Arterial',0,199,117)
    skin_thickness = st.sidebar.slider('Espessura de Pele',0,99,23)
    insulin = st.sidebar.slider('Insulina',0.0,846.0,30.0)
    bmi = st.sidebar.slider('Índice_de_Massa_Corpórea',0.0,67.1,32.0)
    diabetes_pedigree_function = st.sidebar.slider('Histórico de Diabetes na família',0.078,2.42,0.3725)
    age = st.sidebar.slider('Idade',21,81,29)

    user_data = ({
                'Gravidez' : [pregnancies],
                'Glicose' : [glucose],
                'Pressão Arterial' : [blood_pressure],
                'Espessura de Pele' : [skin_thickness],
                'Insulina' : [insulin],
                'IMC' : [bmi],
                'Histórico de Diabetes na família' : [diabetes_pedigree_function],
                'Idade' : [age]
                })
#Transformar os dados em um DataFrame
    features = pd.DataFrame(user_data)
    return features


user_input = get_user_input()

st.subheader('Entrada do usuário')
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train,y_train)
#Modelo de acurácia
st.subheader('Modelo teste de Precisão!')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(X_test)) * 100)+'%')


prediction = RandomForestClassifier.predict(user_input)

#Resultado
st.subheader('Você tem Diabetes?: Sim(1) Não(0) ')
st.write(int(prediction))
