from typing import Optional,List
from pydantic import BaseModel
from fastapi import Depends, FastAPI
import class_operation_binance as OpBI
import pandas as pd
#from starlette.responses import JSONResponse
import httpx
#import io
import matplotlib.pyplot as plt
#import plotly.express as px
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

api = FastAPI(
    title="OPA API Binance",
    description="API to process bianance cryptocurrency ",
    version="1.0.1"
)



@api.get('/get_price_short_information/', name='get_price_short_information')
async def prepare_plot_ouverture(symbole,frequency,date_debut,date_fin):
  """_summary_\n
      Recherche les enregistrements dans la base locale qui ont une valeur de "close_time" comprise entre la plage de dates\n

    Args:\n
        symbole (_String_): paire de trading, exemple BTCUSDT : Cette  permet d'échanger du Bitcoin (BTC) contre le Tether (USDT), 
        qui est une stablecoin indexée sur le dollar américain.
        frequency : 1d 
        date_debut (_IsoDate_): date de debut
        date_fin (_IsoDate_): date de fin
  Returns:\n
      _type_: Json _enregistrements qui ont une valeur de "close_time" comprise entre la plage de dates_ pour le couple de currency demandé
      {"close_time": 1, "close": 1,"open": 1 }
  """
  Op1 = OpBI.Operation_Binance()
  data= Op1.prepare_plot_ouverture(symbole,frequency,date_debut,date_fin)
  data.drop('_id' , axis=1, inplace = True)
  return data.to_dict(orient="records")



@api.get('/get_price_complete/', name='get_all information about couple of currency ')
async def get_historical_kline_for_rangeofday(symbole,date_debut,date_fin):
  """_summary_\n
       Recherche les enregistrements dans la base locale qui ont une valeur de "close_time" comprise entre la plage de dates\n
       
  Args:\n
        symbole (_String_): paire de trading, exemple BTCUSDT : Cette  permet d'échanger du Bitcoin (BTC) contre le Tether (USDT), 
        qui est une stablecoin indexée sur le dollar américain.
        date_debut (_IsoDate_): date de debut
        date_fin (_IsoDate_): date de fin
  Returns:\n
      _type_: Json _enregistrements qui ont une valeur de "close_time" comprise entre la plage de dates_ pour le couple de currency demandé
      "symbole": symbole ,"datetime" : to_datetime(element[0], unit ='ms'),
                                "open" : element[1],
                                "high":element[2],
                                "low":element[3],
                                "close":element[4],
                                "volume":element[5],
                                "close_time":to_datetime(element[6], unit ='ms'),
                                "qav":element[7],
                                "num_trades":element[8],
                                "taker_base_vol":element[9],
                                "num_trades":element[10],
                                "ignore":element[11],
                                "frequency" : interval
  """
  Op1 = OpBI.Operation_Binance()
  data= Op1.get_historical_kline_for_rangeofday(symbole,date_debut,date_fin)
  data.drop('_id' , axis=1, inplace = True)
  
  return data.to_dict(orient="records")


@api.get('/get_all_data_from_local_database/', name='get_all_data_from_local_database')
async def get_all_data():
  """_summary_
      Get all data from dataBase
      Returns:
          _type_: Json All data_from local data base
      """
  Op1 = OpBI.Operation_Binance()
  data= Op1.get_all_data()
  data.drop('_id' , axis=1, inplace = True)
  return data.to_dict(orient="records")

@api.get('/get_statistique/', name='Avoir les statistiques standards pour une paire donnée sur une plage de dates ')
def get_statistique(symbole,date_debut,date_fin):
  Op1 = OpBI.Operation_Binance()
  data= Op1.get_statistique(symbole,date_debut,date_fin)
  
  return data

@api.delete('/delete_all_data/', name ='delete all data from current data base')
async def remove_all_data ():
  """_summary_\n
  remove all data from database

  Returns:
      nada
  """
  Op1 = OpBI.Operation_Binance()
  non_empty_df= Op1.remove_all_data()
  if non_empty_df.empty:
    return {"message": "les données ont été supprimées."}
  else:
    return {"message": "aucune donnée n'a été supprimée"}

@api.put('/new_data', name ='get data from binance and push it in database')
async def push_data_in_local_data_base (symbole, nbofdays:int, frequence:str):

  Op1 = OpBI.Operation_Binance()
  trade=Op1.get_histo_price(symbole, nbofdays , frequence)
  Message=Op1.preprocessing_and_storage(trade,symbole,frequence)
  return Message

@api.get('/Prophete_Prediction/', name ='Prophete_Prediction - Forecast')
async def Prophete_Prediction (symbole:str,frequency:str,date_debut:str,date_fin:str,Nb_of_day_Prediction:int):
  """_summary_\n
  prédiction en utilsant la lib prophète

  Args:\n
        symbole (_String_): paire de trading, exemple BTCUSDT : Cette  permet d'échanger du Bitcoin (BTC) contre le Tether (USDT), 
        qui est une stablecoin indexée sur le dollar américain.
        frequency : 1d 
        date_debut (_IsoDate_): date de debut
        date_fin (_IsoDate_): date de fin
        Nb_of_day_Prediction (int): nombre de jour sur lequel nous souhaitons nous projeter  

  Returns:
      _type_: Json contenant le forcast, à mettre dans des plots (Dash)
  """
  base_url = "http://api_binance_alias:3000/get_price_short_information/" 
  url = f"{base_url}?symbole={symbole}&frequency={frequency}&date_debut={date_debut}&date_fin={date_fin}" 
  
  async with httpx.AsyncClient() as client:
      response = await client.get(url)
      data = response.json()  # Convertir la réponse JSON en données Python

      if isinstance(data, list):  # Si la réponse est une liste de dictionnaires
        df_selec = pd.DataFrame(data)
      elif isinstance(data, dict):  # Si la réponse est un dictionnaire
        df_selec = pd.DataFrame([data])
      
      df_select = df_selec.iloc[:, -2:]
      
      
      df_select["close_time"] = df_select["close_time"]
      df_select["close"] = df_select["close"].astype(float)
      
      df_selec = df_selec.rename(columns={'close_time': 'ds', 'close': 'y'})
      my_model = Prophet(interval_width=0.90)
      my_model.fit(df_selec)
      future_dates = my_model.make_future_dataframe(periods=Nb_of_day_Prediction, freq='D')
      forecast = my_model.predict(future_dates)

      return forecast.to_dict(orient="records")

@api.get('/Prophete_Prediction_metrics/', name ='Prophete_Prediction - Metrics')
async def Prophete_Prediction_metrics (symbole:str,frequency:str,date_debut:str,date_fin:str,Nb_of_day_Prediction:int):
  """_summary_\n
  Etudier la fiabilité de prophète sur un ancien jeu de données

  Args:\n
        symbole (_String_): paire de trading, exemple BTCUSDT : Cette  permet d'échanger du Bitcoin (BTC) contre le Tether (USDT), 
        qui est une stablecoin indexée sur le dollar américain.
        frequency : 1d 
        date_debut (_IsoDate_): date de debut
        date_fin (_IsoDate_): date de fin
        Nb_of_day_Prediction (int): nombre de jour sur lequel nous souhaitons nous projeter  

  Returns:
      _type_: Json contenant le forcast, à mettre dans des plots (Dash) pour objectif de comparaison y_test , pred
  """
  
  base_url = "http://api_binance_alias:3000/get_price_short_information/" 
  url = f"{base_url}?symbole={symbole}&frequency={frequency}&date_debut={date_debut}&date_fin={date_fin}" 
  
  async with httpx.AsyncClient() as client:
      response = await client.get(url)
      data = response.json()  # Convertir la réponse JSON en données Python

      if isinstance(data, list):  # Si la réponse est une liste de dictionnaires
        df_selec = pd.DataFrame(data)
      elif isinstance(data, dict):  # Si la réponse est un dictionnaire
        df_selec = pd.DataFrame([data])
      
      df_select = df_selec.iloc[:, -2:]
     
      
      df_select["close_time"] = df_select["close_time"]
      df_select["close"] = df_select["close"].astype(float)
      
      df_selec = df_selec.rename(columns={'close_time': 'ds', 'close': 'y'})
      my_model = Prophet(interval_width=0.9)
      
      train = df_selec.drop(df_selec.index[-Nb_of_day_Prediction:])
      my_model.fit(train)
      future_dates = my_model.make_future_dataframe(periods=Nb_of_day_Prediction, freq='D')
      forecast = my_model.predict(future_dates)
      
      from sklearn.metrics import mean_absolute_error
      
      
      #y_true = df_selec['y'][-Nb_of_day_Prediction:].values, df_selec['ds'][-Nb_of_day_Prediction:].values  # seléction 
      y_true = df_selec['y'][-Nb_of_day_Prediction:].values
      y_pred = forecast['yhat'][-Nb_of_day_Prediction:].values
      date =  df_selec['ds'][-Nb_of_day_Prediction:].values
      y_pousse = y_true.tolist() + date.tolist()
      
      mae = mean_absolute_error(y_true, y_pred)

      result = {
        "mae": mae,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "date" : date.tolist(),
        "y_pousse" : y_pousse
        
    }

      return result

@api.get('/regression_Lineare/', name ='regression_Lineare_Metrics')
async def Prophete_regression_Lineare (symbole:str,frequency:str,date_debut:str,date_fin:str,size_test:float):
    """_summary_\n
    Utilisation de la regression lineaire

    Args:\n
          symbole (_String_): paire de trading, exemple BTCUSDT : Cette  permet d'échanger du Bitcoin (BTC) contre le Tether (USDT), 
          qui est une stablecoin indexée sur le dollar américain.
          frequency : 1d 
          date_debut (_IsoDate_): date de debut
          date_fin (_IsoDate_): date de fin
          

    Returns:\n
        _type_: Json contenant le forcast, à mettre dans des plots (Dash) pour objectif de comparaison y_test , pred
    """
    
    base_url = "http://api_binance_alias:3000/get_price_short_information/" 
    url = f"{base_url}?symbole={symbole}&frequency={frequency}&date_debut={date_debut}&date_fin={date_fin}" 
    #print("cccc",url)
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()  # Convertir la réponse JSON en données Python

        if isinstance(data, list):  # Si la réponse est une liste de dictionnaires
          df_selec = pd.DataFrame(data)
        elif isinstance(data, dict):  # Si la réponse est un dictionnaire
          df_selec = pd.DataFrame([data])
        
        df_select = df_selec.iloc[:, -3:]
        #print(df_select.head())
        
        #df_select["close_time"] = df_select["close_time"]
        df_select["close"] = df_select["close"].astype(float)
        df_select['open']=df_select['open'].astype('double')
        
        df_select ['J-1'] = df_select['close'].shift(periods=1)
        df_select ['J-2'] = df_select['close'].shift(periods=2)
        df_select ['J-3'] = df_select['close'].shift(periods=3)
        df_select ['J-4'] = df_select['close'].shift(periods=4)
        df_select ['target'] = (df_select['close']-df_select ['open' ])/df_select ['open' ]
      
        prediction_for_tomorrow=df_select.tail(1)
        #print('je suis la', prediction_for_tomorrow)
        """prediction_for_tomorrow=df_select.tail(5)
        
        prediction_for_tomorrow ['J-1'] = prediction_for_tomorrow['close'].shift(periods=1)
        prediction_for_tomorrow ['J-2'] = prediction_for_tomorrow['close'].shift(periods=2)
        prediction_for_tomorrow ['J-3'] = prediction_for_tomorrow['close'].shift(periods=3)
        prediction_for_tomorrow ['J-4'] = prediction_for_tomorrow['close'].shift(periods=4)"""
        
        #print(prediction_for_tomorrow)
        df_select=df_select.dropna()
        feats = df_select.drop(['target', 'close_time','close','open'], axis=1)

        df_select['NouvelIndex'] = range(len(df_select)) # pour remplacer la date relation 1 pour 1

# Définir la nouvelle colonne comme index
        df_select.set_index('NouvelIndex', inplace=True)
        target=df_select['close']
        prediction_for_tomorrow = prediction_for_tomorrow.drop(['target', 'close_time','close','open'], axis=1)
        
        X_train, X_test, y_train, y_test=train_test_split(feats, target, test_size=size_test, random_state=42)
        

        reglin=LinearRegression ()
        reglin.fit(X_train, y_train)
        pred_test = reglin.predict(X_test)
        pred_tomorrow = reglin.predict(prediction_for_tomorrow)
        Score_train =reglin.score(X_train, y_train)
        Score_test = reglin.score(X_test, y_test)
        print('Score sur ensemble train', reglin.score(X_train, y_train))
        print('Score sur ensemble test', reglin.score(X_test, y_test))
        
        print (pred_tomorrow)
        result = {
        "score_train": Score_train,
        "Score_test": Score_test,
        "prediction_test": pred_test.tolist(),
        "y_true": y_test.tolist(),
        "pred_tomorrow" : pred_tomorrow.tolist()
        
    }

        return result