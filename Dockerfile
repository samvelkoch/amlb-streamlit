# Use the official Python 3.10 image as the base image
FROM python:3.11 AS main

# Set the PYTHONUNBUFFERED environment variable
ENV PYTHONUNBUFFERED=1

# Expose port 8088 for the application to run on
EXPOSE 8088

# Set the working directory to /app
WORKDIR /app

# Update the package list, install ca-certificates and local certificate and build-essential packages,
# and clean up the package list cache
RUN apt-get update && \
    apt-get install --no-install-recommends --yes build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
# Upgrade pip, install poetry, configure poetry to not create virtualenvs,
# and install the project dependencies without development dependencies
RUN pip install --upgrade pip &&\
    pip install poetry &&\
    poetry config virtualenvs.create false &&\
    poetry install --only main

COPY . /app
# Set the PYTHONPATH environment variable to include the /app directory
ENV PYTHONPATH=/app/src

#This stage creates dev image for running debugging sessions
# FROM main AS dev

# CMD ["streamlit", "run", "src/app.py"]
