import os
from sys import displayhook

from binance.client import Client
from datetime import datetime, timedelta
from pandas import to_datetime
import pandas as pd
from pymongo import MongoClient,errors




class Operation_Binance:
    def __init__(self):
       # export binance_api="8Gt1E5gTGJL69t0o6F2yhJJqpfq5g05TAzv1KFfVKso5HJwj6Dt4gCWC4UuN8rbw"
        #export binance_secret="pDBbL2n8WYsSJeC83sM9fvdijoHDHBsoMO3m9QpyTJVLVyVxQL5Ve9YEQx7OSvGr"
        self.api_key = os.environ.get('binance_api')
        self.api_secret = os.environ.get('binance_secret')
        self.client_binance = Client(self.api_key, self.api_secret)
        #Biance Test URL 
        self.client_binance.API_URL = 'https://api.binance.com/api' #'https://testnet.binance.vision/api'
        #Init MongoClient
        self.MongoCli = MongoClient(
            host="mongo_alias",#127.0.0.1
            port = 27017,
            username = "admin",
            password = "pass")
        self.my_binance_mongo=self.MongoCli["sample"]["batch_binance"]
        self.my_binance_mongo.create_index([("symbole", 1), ("datetime", 1),("frequency",1)], unique=True)
       
        #self.my_binance_mongo.create_index("symbole", "datetime" , unique=True)
        
        
    def get_histo_price(self,symbole, nbofdays,interval):
        """une fonction de récupération de données générique afin de pouvoir avoir les données de n’importe quel marché.
        Obtenir les prix d'un couple de symbole sur les nbofdays derniers jours
        interval peut etre en jour 1d, en semaine 1w, en mois 1M"""
        #Calcul de la date avant nbofdays
        my_date = datetime.now() - timedelta(nbofdays)
        #transformation de la date en integer pour pouvoir utiliser le endpoint de l'API
        date_start = int(my_date.timestamp()*1000)
        date_end   = int(datetime.now().timestamp()*1000)
        # Appel api pour récuperer l'historique des prix
        trades =self.client_binance.get_historical_klines (symbole, interval ,date_start,date_end) 
        ##pprint (trades)
        return trades    
    
     
    def preprocessing_and_storage(self,trades,symbole,interval):
        """ pré-processing pour réorganiser les données sortant de l'API afin qu’elles soient propres.creation d'une liste pour le chargement des données en mode bloc dans MongoDB""" 
        listeA = []
        for element in trades:
            listeA.append({"symbole": symbole ,"datetime" : to_datetime(element[0], unit ='ms'),
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
                                "frequency" : interval})#la frequence daily/monthly, ou weekly etc})
        #injection des données dans MongoDB
        #auto-regressif (regression lineare, en entree la date , prédire , librairie prophète (time series))[une periode fixée pour l'entrainement, (l'année,mois,valeur de cloture M-1 ---M-N)]
        #un modèle de serie temporelle, arimas/Sarima
        #librairie prophète (facteur de saisonalité, tendance, cyclique)
        #
        # un modèle entrainé 
        # Insérez plusieurs documents dans la collection MongoDB
        try:
            result = self.my_binance_mongo.insert_many(listeA)
            return {"message": f"{len(result.inserted_ids)} documents insérés avec succès"}
        except errors.BulkWriteError as e:
            return {"warning": "les doublons n'ont pas été insérés", "details": str(e)}
        except Exception as e:
            return {"error": "Une erreur est survenue lors de l'insertion", "details": str(e)}
        

    def get_all_data (self):
        """_summary_
        Get all data from dataBase
        Returns:
            _type_: _None_
        """
        
        return pd.DataFrame(self.my_binance_mongo.find({}))
    
    
    def remove_all_data (self):
        """remove all data from database
        """
        self.my_binance_mongo.drop() 
        
        
    def get_historical_kline_for_rangeofday (self,symbole,date_debut,date_fin):
        """ # Recherche des enregistrements qui ont une valeur de "close_time" comprise entre la plage de dates

        Args:
           symbole (_String_): paire de trading, exemple BTCUSDT : Cette  permet d'échanger du Bitcoin (BTC) contre le Tether (USDT), qui est une stablecoin indexée sur le dollar américain.
            date_debut (_IsoDate_): date de debut
            date_fin (_IsoDate_): date de fin
        """
        # Recherche des enregistrements qui ont une valeur de "close_time" comprise entre la plage de dates
        #print (date_debut,date_fin)
        result = self.my_binance_mongo.find({"symbole": symbole,"close_time": {"$gte": to_datetime(date_debut), "$lte": to_datetime(date_fin)}})
        # Parcourir les résultats de la recherche
        
        return pd.DataFrame(result)
        
        
    def get_statistique(self,symbole,date_debut,date_fin):
        """_Avoir les statistiques standards pour une paire donnée sur une plage de dates 

        Args:
            symbole (_String_): paire de trading, exemple BTCUSDT : Cette  permet d'échanger du Bitcoin (BTC) contre le Tether (USDT), qui est une stablecoin indexée sur le dollar américain.
            date_debut (_IsoDate_): date de debut
            date_fin (_IsoDate_): date de fin
        """
        res= self.my_binance_mongo.find({"symbole": symbole,"close_time": {"$gte": to_datetime(date_debut), "$lte": to_datetime(date_fin)}})
        data = pd.DataFrame(res)
        #data.drop('_id', axis=1, inplace = True)
        print(data.describe())
        return data[['high','open','symbole','qav']].describe().to_dict()
    
    def prepare_plot(self,symbole,frequency,date_debut,date_fin):
        """_Avoir les statistiques standards pour une paire donnée sur une plage de dates 
        Args:
            symbole (_String_): paire de trading, exemple BTCUSDT : Cette  permet d'échanger du Bitcoin (BTC) contre le Tether (USDT), qui est une stablecoin indexée sur le dollar américain.
            date_debut (_IsoDate_): date de debut
            date_fin (_IsoDate_): date de fin
        """
        projection_fields = {"close_time": 1, "close": 1}
        res= self.my_binance_mongo.find({"symbole": symbole, "frequency" :frequency,  "close_time": {"$gte": to_datetime(date_debut), "$lte": to_datetime(date_fin)}}, projection=projection_fields )
        return pd.DataFrame(res)
        
        
    def prepare_plot_ouverture(self,symbole,frequency,date_debut,date_fin):
        """_Avoir les statistiques standards pour une paire donnée sur une plage de dates 
        Args:
            symbole (_String_): paire de trading, exemple BTCUSDT : Cette  permet d'échanger du Bitcoin (BTC) contre le Tether (USDT), qui est une stablecoin indexée sur le dollar américain.
            date_debut (_IsoDate_): date de debut
            date_fin (_IsoDate_): date de fin
        """
        projection_fields = {"close_time": 1, "close": 1,"open": 1 }
        res= self.my_binance_mongo.find({"symbole": symbole, "frequency" :frequency,  "close_time": {"$gte": to_datetime(date_debut), "$lte": to_datetime(date_fin)}}, projection=projection_fields )
        return pd.DataFrame(res)
        
    def get_symbole(self):
            symbole =self.my_binance_mongo.distinct('symbole')
            ##pprint (trades)\n",
            return symbole 
    
OP = Operation_Binance()
print(OP.get_symbole())