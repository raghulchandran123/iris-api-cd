# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model into the container at /app
COPY fastapi_app.py .
COPY model.joblib .

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# Command to run the application
# Use gunicorn with uvicorn workers for production-ready deployment
# For simplicity in this assignment, we'll stick to uvicorn directly.
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
