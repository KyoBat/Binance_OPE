FROM ubuntu:18.04


ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8


WORKDIR /my_server/

RUN apt-get update && \
    apt-get install -y python3 python3-pip build-essential python3-dev apt-utils libopenblas-dev python3-venv

COPY requirements.txt .


RUN python3 -m venv /my_venv
ENV PATH="/my_venv/bin:$PATH"


# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port 8000 (adjust as needed)
EXPOSE 3000

# Copy your Python files into the container
COPY class_operation_binance.py .
COPY api_binance.py .
#COPY ui_binance.py .

# Define the working directory for the container
WORKDIR /my_server/

# lancement de l'api
CMD ["uvicorn", "api_binance:api", "--host", "0.0.0.0", "--port", "3000"]