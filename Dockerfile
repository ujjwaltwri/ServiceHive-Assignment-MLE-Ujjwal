# Start from a Python 3.9 base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# The command to run your app using uvicorn
CMD ["uvicorn", "SIC.api.main:app", "--host", "0.0.0.0", "--port", "8000"]