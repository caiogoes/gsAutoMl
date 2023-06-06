#!pip install pycaret
#!pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import *
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

st.set_page_config( page_title = 'GS - Auto ML',
                    page_icon = './logo_fiap.png',
                    layout = 'wide',
                    initial_sidebar_state = 'expanded')

st.title('Sistema Integrado de Produtos e Estabelecimentos Agropecuários - Fertilizante')

with st.sidebar:
    c1, c2 = st.columns(2)
    c1.image('./logo_fiap.png', width = 100)
    c2.write('')
    c2.subheader('GS Auto ML - Fiap')

    st.subheader('Fonte dos dados de entrada: CSV')

    st.info('Upload do CSV')
    file = st.file_uploader('Selecione o arquivo CSV', type='csv')

#Tela principal
if file:
    #carregamento do CSV
    Xtest = pd.read_csv(file, sep=',')
    df_data = Xtest.copy()

    le = preprocessing.LabelEncoder()
    for col in Xtest.columns:
        Xtest[col] = le.fit_transform(Xtest[col])

    #carregamento / instanciamento do modelo pkl
    mdl_lgbm = load_model('model')

    #predict do modelo
    ypred = predict_model(mdl_lgbm, data = Xtest, raw_score = True)

    with st.expander('Visualizar CSV carregado:', expanded = True):
        c1, _ = st.columns([2,4])
        qtd_linhas = c1.slider('Visualizar quantas linhas do CSV:', 
                                min_value = 5, 
                                max_value = df_data.shape[0], 
                                step = 10,
                                value = 5)
        st.dataframe(df_data.head(qtd_linhas))

    with st.expander('Visualizar Predições:', expanded = False):
        c1, c2, c3 = st.columns([2,2,2])

        qtd_ativos = len(ypred[ypred["prediction_label"]=="Ativo"])
        qtd_cancelados = len(ypred[ypred["prediction_label"]=="Cancelado"])
        qtd_suspensos = len(ypred[ypred["prediction_label"]=="Suspenso"])

        c1.metric('Qtd ativos', value = qtd_ativos)
        c2.metric('Qtd cancelados', value = qtd_cancelados)
        c3.metric('Qtd suspensos', value = qtd_suspensos)

        def color_pred(val):
            color = 'olive' if val == 'Ativo' else 'red' if val == "Cancelado" else 'yellow'
            return f'background-color: {color}'

        df_view = pd.concat([df_data, ypred["prediction_label"], ypred["prediction_score_Ativo"], ypred["prediction_score_Cancelado"], ypred["prediction_score_Suspenso"]], axis = 1)

        st.dataframe(df_view.style.applymap(color_pred, subset = ['prediction_label']))
else:
    st.warning('Arquivo CSV não foi carregado')
