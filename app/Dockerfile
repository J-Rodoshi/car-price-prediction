FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/ ./code/

EXPOSE 8050

CMD ["python", "code/app.py"]