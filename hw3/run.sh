sudo chmod 666 /var/run/docker.sock

docker build -t pyspark-image .

docker run -v /home/ubuntu/lab2/DE300_ER/DE300/hw3:/tmp/hw3 -it -p 8888:8888 --name spark-sql-container pyspark-image