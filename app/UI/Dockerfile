FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y procps && pip install -U pip && rm /etc/localtime && ln -s /usr/share/zoneinfo/America/Mexico_City /etc/localtime

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r ./requirements.txt

COPY ./main.py /app/

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]