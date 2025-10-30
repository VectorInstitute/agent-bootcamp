FROM python:3.12-slim
WORKDIR /app

COPY requirements-app.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-app.txt

RUN mkdir -p /app/src/utils/web_search
RUN touch /app/src/utils/__init__.py
COPY __init__.py app.py auth.py db.py /app/src/utils/web_search/

ENV PYTHONPATH=/app/src
CMD ["uvicorn", "utils.web_search.app:app", "--host", "0.0.0.0", "--port", "8080"]
