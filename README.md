# mle-dvc

# Запуск/остановка докера
docker compose up --build
docker compose down 
docker compose restart

# Установка зависимостей
pip install -r requirements.txt


# Права на запись 
sudo chown -R $USER /home/mle_projects/mle_airflow/

# git
sudo git add . && sudo git commit -m "











dvc remote modify my_storage endpointurl \ 