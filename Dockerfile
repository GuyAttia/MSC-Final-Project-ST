# Create the image base on the Miniconda3 image
FROM python:3.8-slim

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# Creating the working directory in the container
WORKDIR /FPST
# Copy the local code to the container
COPY . /FPST/.

# Install requirements
RUN /usr/local/bin/python -m pip install --upgrade pip && \
  pip3 install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/FPST/src
ENV PYTHONUNBUFFERED=1