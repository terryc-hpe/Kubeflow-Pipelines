#!/bin/bash
DIR=$(dirname $(readlink -f $0))

s2i build ${DIR} seldonio/seldon-core-s2i-python37:1.2.3 ${USERNAME}/${CLEAN_STEP_IMAGE}:${TAG}

docker push ${USERNAME}/${CLEAN_STEP_IMAGE}:${TAG}
