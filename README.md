# Imrann laisses papa travailler
# creating directoriescd 
mkdir clean_data
mkdir raw_files
mkdir dags
mkdir plugin
mkdir logs

sudo chown ubuntu raw_files/ clean_data/ plugin/ dags/ logs/

echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env

docker-compose up airflow-init

wget https://dst-de.s3.eu-west-3.amazonaws.com/airflow_avance_fr/eval/data.csv -O clean_data/data.csv
echo '[]' >> raw_files/null_file.json

# starting docker-compose
docker-compose up -d
![image](https://github.com/KyoBat/Binance_OPE/assets/127420606/acc1f11a-b5c2-4f4a-a2f5-c3893c91117f)
