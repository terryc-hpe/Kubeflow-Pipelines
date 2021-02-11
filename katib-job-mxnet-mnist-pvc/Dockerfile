FROM python:3.7.0

RUN pip3 install mxnet

ADD . /opt/mxnet-mnist

WORKDIR /opt/mxnet-mnist

RUN chgrp -R 0 /opt/mxnet-mnist \
  && chmod -R g+rwX /opt/mxnet-mnist

ENTRYPOINT ["python3", "/opt/mxnet-mnist/mxnet-mnist.py"]
