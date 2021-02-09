#!/bin/bash
DIR=$(dirname $(readlink -f $0))
docker build ${DIR} -t ${USERNAME}/${EXTRACT_STEP_IMAGE}:${TAG}
docker push ${USERNAME}/${EXTRACT_STEP_IMAGE}:${TAG}