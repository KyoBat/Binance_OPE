from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
#import class_operation_binance as OpBI
from airflow.operators.http_operator import SimpleHttpOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from pymongo import MongoClient
#from my_server import ui_binance as Pom 
my_dag = DAG(
    dag_id='UpdateDataBase',
    description='Le Dag s exécute toutes les jours à minuit',
    tags=['Projet_Binance'],
    schedule_interval='35 18 * * *',
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    },
    catchup=False
)

"""def pos():
    Op1 = OpBI.Operation_Binance()
    trade=Op1.get_histo_price('ETHBUSD', 1 , '1d')
    Op1.preprocessing_and_storage(trade,'ETHBUSD','1d')"""





base_url = 'api_binance_alias:3000'
symbols = ['BTCBUSD', 'ETHBUSD','BNBUSDT', 'ALCXBTC', 'ALGOBTC', 'AAVEBTC',  'ADABTC', 'ADXBTC', 'AGIXBTC',  'ACHBTC', 'ACABTC']
for symbol in symbols:
    task_id = f'api_sensor_task_{symbol.lower()}' 
    endpoint = f'{base_url}/new_data?symbole={symbol}&nbofdays=4000&frequence=1d'

    api_task = SimpleHttpOperator(
        task_id=task_id,
        method='PUT',
        http_conn_id='http_default',
        endpoint=endpoint,
        dag=my_dag,
    )


#Task1
api_task  # >> trigger_task or any other downstream task

if __name__ == "__main__":
    my_dag.cli()