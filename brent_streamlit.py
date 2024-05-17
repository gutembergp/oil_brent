import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# For SARIMA model
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose # For time series decomposition
from pmdarima import auto_arima

# For LSTM model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model  # Allows load a previously saved model.

# To evaluate the models
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Títulos
st.write('# Grupo 69')
st.write('# Oil Brent ')

# Subindo e tratandos os dados
raw_list = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view', 
                      encoding='UTF-8')

df_brent = raw_list[2]
df_brent.columns = ['date','brent_crude_oil']
df_brent = df_brent.drop(0)
df_brent['date'] = pd.to_datetime(df_brent['date'],format='%d/%m/%Y')
df_brent = df_brent.astype({'brent_crude_oil':'float64'})
df_brent['brent_crude_oil'] = df_brent['brent_crude_oil'] / 100
chart_data = df_brent
df_brent = df_brent.set_index('date')
df_brent.sort_values(by='date', inplace=True)
df_brent.drop_duplicates(inplace=True)
df_brent = df_brent.squeeze()
df_brent = df_brent.asfreq('B')
df_brent.fillna(method='ffill', inplace=True)

data = df_brent

# Criando o layout da aplicação
tab0, tab1, tab2, tab3 = st.tabs(["Histórico Preço Oil Brent", "Previsão SARIMA", "Forecast LSTM", "Conclusão"])

with tab0:
    st.write('### Preço do Oil Brent (USD)')

    chart_data.sort_values(by='date', inplace=True)  
    st.line_chart(chart_data, x="date", y="brent_crude_oil")

with tab1:
   st.write('### Previsão Preço Oil Brent (15 dias)')
   st.write('### Método SARIMA')

   #Data Split
   train_data = data.iloc[:len(data) - 30]
   test_data = data.iloc[len(data) - 30:]
   
   # Best Model -> (3,1,1)x(0,0,0,0)
   sarima_model_eval = SARIMAX(train_data, order=(3,1,1), seasonal_order=(0, 0, 0, 0))
   estimator_eval = sarima_model_eval.fit()

   # Gets forecast for evaluation
   preds = estimator_eval.forecast(len(test_data))

   # Sets the model
   sarima_model_forecast = SARIMAX(data, order=(3,1,1), seasonal_order=(0,0,0,0))
   estimator_forecast = sarima_model_forecast.fit()

   # Makes predictions
   steps_ahead = 15
   st.write('Prevendo os próximos {} days: \n'.format(steps_ahead))
   
   forecasts = estimator_forecast.forecast(steps_ahead)
   ci = estimator_forecast.conf_int()

   df_forecast = pd.DataFrame(forecasts)
   df_forecast.index = pd.to_datetime(df_forecast.index, format = '%Y-%m-%d').strftime('%d/%m/%Y')
   df_forecast.rename(columns={"predicted_mean": "Forecast ARIMA"}, inplace=True)

   st.write(df_forecast)
   
   rmse = np.sqrt(mean_squared_error(test_data.values, preds.values))
   mae = mean_absolute_error(test_data.values, preds.values)
   mape = mean_absolute_percentage_error(test_data.values, preds.values)
   st.write('Root Mean Square Error (RMSE): ', rmse) 
   st.write('Mean Absolute Error (MAE): ', mae)
   st.write('Mean Absolute Percentage Error (MAPE): ', mape)
  

with tab2:
   st.write('### Previsão Preço Oil Brent (15) dias)')
   st.write('### Método LSTM')

   # Data Split
   train_data = data.iloc[:len(data) - 30]
   test_data = data.iloc[len(data) - 30:]

   # Reshapes the data to feed the model
   full_data_lstm = data.values.reshape(-1, 1)
   train_data_lstm = train_data.values.reshape(-1, 1)
   test_data_lstm = test_data.values.reshape(-1, 1)

   # Defines train and test sets
   X_train = []
   y_train = []
   ws = 30 # Window size: indicates the number of previous time steps. The more, may lead to higher accuracy, but increases complexity and training time.

   for i in range(ws, len(train_data_lstm)):
      X_train.append(train_data_lstm[i - ws: i])
      y_train.append(train_data_lstm[i])

   X_train, y_train = np.array(X_train), np.array(y_train)
   
   import joblib
   from joblib import load
   model = joblib.load('brent.joblib')

   # Model Testing
   prediction_set = []
   batch_one = train_data_lstm[-ws:]
   new_batch = batch_one.reshape((1, ws, 1))

   for i in range(len(test_data)):
      pred = model.predict(new_batch, verbose=False)[0]
      prediction_set.append(pred)
      new_batch = np.append(new_batch[:, 1:, :], [[pred]], axis=1)

   prediction_set = [i[0] for i in prediction_set]
   predictions = pd.Series(prediction_set, index=test_data.index)

   # Makes the predictions 
   prediction_set = []
   batch_one = full_data_lstm[-ws:]
   new_batch = batch_one.reshape((1, ws, 1))
   days_to_forecast = 15

   st.write('Prevendo os próximos {} dias: '.format(days_to_forecast))

   for i in range(days_to_forecast):
      pred = model.predict(new_batch, verbose=False)[0]
      prediction_set.append(pred)
      new_batch = np.append(new_batch[:, 1:, :], [[pred]], axis=1)

   prediction_set = [i[0] for i in prediction_set] 
   date_range = pd.date_range(test_data.index[-1], periods=days_to_forecast, freq='B')   
   forecast = pd.Series(prediction_set, index=date_range, name='Forecast')

   df_forecast_lstm = pd.DataFrame(forecast)
   df_forecast_lstm.index = pd.to_datetime(df_forecast_lstm.index, format = '%Y-%m-%d').strftime('%d/%m/%Y')
   st.write(df_forecast_lstm)

   rmse_lstm = np.sqrt(mean_squared_error(test_data_lstm, predictions))
   mae_lstm = mean_absolute_error(test_data_lstm, predictions)
   mape_lstm = mean_absolute_percentage_error(test_data_lstm, predictions)
   st.write('Root Mean Square Error (RMSE): ', rmse_lstm) 
   st.write('Mean Absolute Error (MAE): ', mae_lstm)
   st.write('Mean Absolute Percentage Error (MAPE): ', mape_lstm)
  
with tab3:

   st.write('### Conclusão')
   st.write('Modelar o preço futuro do Oil Brent é uma tarefa complexa devido a vários fatores.')
   st.write('1. Volatilidade dos Preços Internacionais como eventos geopolíticos, mudanças na oferta e demanda, flutuações cambiais e outros fatores imprevisíveis')
   st.write('2. Dependência de Fatores Externos: Os preços do petróleo estão sujeitos a decisões políticas e econômicas de grandes produtores, como a OPEP (Organização dos Países Exportadores de Petróleo) e outros países. Eventos inesperados, como desastres naturais, conflitos armados ou mudanças nas políticas governamentais, podem afetar drasticamente os preços.')
   st.write('3. Transição Energética e Pressões Ambientais - A crescente conscientização sobre as mudanças climáticas e a busca por fontes de energia mais limpas estão pressionando a indústria do petróleo.')
   st.write('4. Diversidade de Usos do Petróleo - O petróleo é usado em uma variedade de setores, incluindo transporte, indústria, geração de energia e produtos químicos. Modelar a demanda futura em cada um desses setores é desafiador, pois eles têm dinâmicas diferentes.')
   st.write('5. Incertezas Econômicas e Tecnológicas - Mudanças na economia global, avanços tecnológicos e inovações disruptivas podem afetar a demanda por petróleo.Por exemplo, adoção de veículos elétricos. ') 
   
   
   st.write('### Análise das Previsões')

   d = {' ' : ['RMSE','MAE', 'MAPE'] , 'SARIMA': [rmse, mae,mape], 'LSTM': [rmse_lstm, mae_lstm ,mape_lstm]}
   df = pd.DataFrame(data=d)

   st.write(df)

   st.write('Observamos que o valores de Root Mean Square Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE)  e o forecast para os próximos 15 dias são muito próximos para os dois métodos.')
  
   df_lstm = pd.DataFrame(forecast)
   df_lstm.index = pd.to_datetime(df_lstm.index, format = '%Y-%m-%d').strftime('%d/%m/%Y')
   df_lstm.rename(columns={"Forecast": "Forecast LSTM"}, inplace=True)

   df_forecast.rename(columns={"Forecast": "Forecast SARIMA"}, inplace=True)

   df_comp = pd.concat([df_forecast, df_lstm], axis=1)
   df_comp.index = pd.to_datetime(df_comp.index, format = '%d/%m/%Y')
   df_comp.sort_index(inplace = True)
   df_comp.index = pd.to_datetime(df_comp.index, format = '%d/%m/%Y').strftime('%d/%m/%Y')
   st.write(df_comp)

   st.write('O método LSTM utiliza mais recursos computacionais mas não tem um impacto muito grande no processamento desse volume de dados')
   st.write('Portando os dois métodos podem ser considerados para o Forecast do Valor do Oil Brent')