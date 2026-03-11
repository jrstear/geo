FROM opendronemap/odm:latest

RUN apt-get update && apt-get install -y awscli && rm -rf /var/lib/apt/lists/*

COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

ENTRYPOINT ["/app/run.sh"]
