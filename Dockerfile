FROM python:3.8

# Create the app directory and copy the contents
RUN mkdir /app
COPY . /app/
WORKDIR /app/

# Install requirements
RUN pip3 install -r requirements.txt

# Set environment variables
ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

# Initialize Airflow database and create a user
RUN ["bash", "-c", "airflow db init"]
RUN ["bash", "-c", "airflow users create -e pravinbokare10@gmail.com -f Pravin -l Bokare -p admin -r Admin -u admin"]

# Set permissions for start.sh
RUN chmod +x start.sh

# Install AWS CLI
RUN apt update -y && apt install awscli -y

# Specify the entry point and command
ENTRYPOINT [ "/bin/sh" ]
CMD ["./start.sh"]
