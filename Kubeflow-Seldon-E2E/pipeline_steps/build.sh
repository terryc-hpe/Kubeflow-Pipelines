#!/bin/bash
export EXTRACT_STEP_IMAGE=features_extractor
export CLEAN_STEP_IMAGE=clean_text_transformer
export TOKENIZE_STEP_IMAGE=spacy_tokenizer
export VECTORIZE_STEP_IMAGE=tfidf_vectorizer
export PREDICT_STEP_IMAGE=lr_text_classifier


echo Image Tag?
read TAG

export TAG=${TAG}

download_s2i(){
   s2i > /dev/null 2>&1
   if [ $? -eq 0 ]; then
      echo "s2i Installed..\n"
   else
     echo "s2i Installing....\n"
     wget https://github.com/openshift/source-to-image/releases/download/v1.3.1/source-to-image-v1.3.1-a5a77147-linux-amd64.tar.gz  
     tar xf source-to-image-v1.3.1-a5a77147-linux-amd64.tar.gz 
     cp s2i /bin
     rm -rf source-to-image-v1.3.1-a5a77147-linux-amd64.tar.gz s2i sti
   fi

}

build_pipeline_images(){
  for i in $(ls -d */); 
    do 
      echo "#################################################"
      bash ${i}build_image.sh
      echo "#################################################"
    done
}

FILE=~/.docker/config.json
if [ -f "$FILE" ]; then
    is_login=`cat ~/.docker/config.json  | jq -r ".auths[].auth"`
else
    is_login='' 
fi

if [ -z $is_login ]
then
    echo Enter Username?
    read USERNAME
    export USERNAME=${USERNAME}
    echo Enter Password?
    read PASSWORD
    export PASSWORD=${PASSWORD}
    docker login --username=$USERNAME --password=$PASSWORD $REGISTRY
    if [ $? -eq 0 ]; then
      echo Login Successful.
      download_s2i
      build_pipeline_images
    else
      echo Login Failed.
      exit 1
    fi
else
   echo Enter Username?
   read USERNAME
   export USERNAME=${USERNAME}
   download_s2i
   build_pipeline_images
fi

echo "######################################################"
echo "Docker images for Kubeflow Pipeline steps:\n\n"
echo EXTRACT_STEP_IMAGE=${EXTRACT_STEP_IMAGE}:${TAG}
echo CLEAN_STEP_IMAGE=${CLEAN_STEP_IMAGE}:${TAG}
echo TOKENIZE_STEP_IMAGE=${TOKENIZE_STEP_IMAGE}:${TAG}
echo VECTORIZE_STEP_IMAGE=${VECTORIZE_STEP_IMAGE}:${TAG}
echo PREDICT_STEP_IMAGE=${PREDICT_STEP_IMAGE}:${TAG}
echo "######################################################"
