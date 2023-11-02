#!/bin/sh
sudo docker container stop tuned-client
sudo docker container remove tuned-client
sudo docker create -p 8181:8181 --ipc host --gpus 1 --volume petals-cache3:/root/.cache --name tuned-client hive-chat-tuned:latest
sudo docker container start tuned-client
