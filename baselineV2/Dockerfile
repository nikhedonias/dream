FROM python:3.5

# Do not change
LABEL challenge="covid19"

# Description of the model (max 500 characters)
LABEL description="COVID diagnosis baseline prediction v2"

# Comma-separated list of arbitrary feature names used by the model in desc
# order of importance.
LABEL ranked_features="pre-filter + clustering"

# Comma-separated list of references (URL)
LABEL references="https://www.medrxiv.org/content/10.1101/2020.03.19.20039099v1,https://www.biorxiv.org/content/10.1101/2020.05.20.106401v1"

# Install dependencies
RUN pip install pandas \
  numpy \
  sklearn \
  datetime

# Move to the app folder
RUN mkdir /app
WORKDIR /app

# Copy our python program for training and inference
COPY src/infer_v2.py .


# Copy Bash scripts expected by the IT infrastructure of the EHR DREAM Challenge
COPY infer.sh .

# Add executable permission to Bash scripts
RUN chmod +x infer.sh
