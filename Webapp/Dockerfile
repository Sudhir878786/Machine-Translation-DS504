FROM python:3

ENV EN_FR_TRANSLATOR_ENDPOINT http://localhost:5001
ENV FR_EN_TRANSLATOR_ENDPOINT http://localhost:5002

WORKDIR /app

COPY ["./Webapp/static", "./Webapp/static"]
COPY ["./Webapp/templates", "./Webapp/templates"]
COPY ["./Webapp/app.py", "./Webapp/app.py"]
COPY ["./Webapp/requirements.txt", "./Webapp/requirements.txt"]

WORKDIR "/app/Webapp"

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD python app.py
