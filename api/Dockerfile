FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fastapi_app.py .
# REMOVE this line: COPY model.joblib /app/model.joblib # This expects model.joblib to be here
# ADD this new line:
RUN cp model.joblib /app/model.joblib
EXPOSE 8000
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
