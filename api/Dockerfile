FROM python:3.9-slim-buster
WORKDIR /app
# **IMPORTANT CHANGE:** Copy requirements.txt from relative path in build context
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# **IMPORTANT CHANGE:** Copy fastapi_app.py from relative path in build context
COPY api/fastapi_app.py .
# This one is already correct now as model.joblib is at repo root
COPY model.joblib /app/model.joblib
EXPOSE 8000
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
