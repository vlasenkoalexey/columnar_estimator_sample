#!/bin/bash

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
MODEL_NAME=${CURRENT_DATE}
MODEL_DIR=models/${MODEL_NAME}/model
DOCKER_FILE_NAME="Dockerfile_tf114"

for i in "$@"
do
case $i in
    --docker-file-name=*)
    DOCKER_FILE_NAME="${i#*=}"
    ;;
    --tensorboard)
    TENSORBOARD=true
    ;;
    --model-name=*)
    MODEL_NAME="${i#*=}"
    ;;
esac
done

DOCKER_IMAGE_TAG="${DOCKER_FILE_NAME,,}-v1"

if [ "$TENSORBOARD" = true ] ; then
    trap "kill 0" SIGINT
    echo "running tensorboard: tensorboard --logdir=${MODEL_DIR}/logs --port=0 --reload_multifile=true"
    tensorboard --logdir=${MODEL_DIR}/logs --port=0 --reload_multifile=true &
fi

echo "Rebuilding docker image $DOCKER_FILE_NAME"
docker build -f $DOCKER_FILE_NAME -t $DOCKER_IMAGE_TAG ./

echo "Running training job in docker image..."
docker run -v ${PWD}/${MODEL_DIR}:/${MODEL_DIR} $DOCKER_IMAGE_TAG python trainer.py --job-dir=/${MODEL_DIR} $@