# Use the official Python image
FROM python:latest

# Install the latest AWS CLI version 2
USER root

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    bzip2 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the AWS CLI tools to your image
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip ./aws

# Create a new user and set permissions
RUN useradd -m elina

# Set ownership of the working directory
WORKDIR /app
RUN chown -R elina /app

# Create a virtual environment
RUN python -m venv /venv

# Activate the virtual environment
ENV PATH="/venv/bin:${PATH}"

RUN pip install pandas jupyter numpy scikit-learn boto3 requests beautifulsoup4 scrapy lxml

#switch to non-root user
USER elina

# Copy ETL scripts or Jupyter notebooks into the container
COPY ./data/* /app/rawdata/
COPY ./src/* /app/src/
COPY ./run.sh /app/run.sh
COPY ./README.md /app/README.md

# Expose the Jupyter Notebook port
EXPOSE 8888

# Command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

