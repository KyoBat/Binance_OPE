FROM ubuntu:18.04


ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8


WORKDIR /my_server
RUN mkdir assets
RUN chmod 777 assets
RUN apt-get update && \
    apt-get install -y python3 python3-pip build-essential python3-dev apt-utils libopenblas-dev python3-venv

COPY requirements.txt .


RUN python3 -m venv /my_venv
ENV PATH="/my_venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install -r requirements.txt
#RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 2000

COPY ui_binance.py .
#COPY class_operation_binance.py .


# Lancement de l'UI
CMD ["python", "/my_server/ui_binance.py"]