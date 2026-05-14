FROM python:3.14-slim-trixie

WORKDIR /app

COPY ./data_scraping/* .

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "scraper.py"]