FROM python:3.14-slim-trixie

WORKDIR /app

COPY ./data_scraping/bus_data.db requirements_data.txt .

RUN pip install -r requirements_data.txt

ENTRYPOINT ["python3", "scraper.py"]