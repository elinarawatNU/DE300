sudo chmod 666 /var/run/docker.sock

docker build -f dockerfiles/Dockerfile -t hw2-image .

docker run --name hw2-container -p 8888:8888 -v hw2project:/home/jovyan hw2-image
