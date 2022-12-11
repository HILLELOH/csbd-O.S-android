FROM python:2.7
RUN apt update && apt install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install scikit-learn==0.18.1 pebble joblib==0.11

RUN git clone https://github.com/HILLELOH/csbd-O.S-android.git cbsd
