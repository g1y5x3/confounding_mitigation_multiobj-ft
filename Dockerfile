FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y vim  && \
    apt-get install -y git  && \
    apt-get -y install python3-pip && \
    pip install mlconfound && \
    pip install wandb && \
    pip install pandas && \
    pip install torchinfo && \
    pip install tqdm && \
    pip install numpy &&\
    pip install nibabel &&\
    pip install pyyaml && \
    pip install scikit-learn && \
    pip install pymoo

ADD data/subjects_40_v6.mat /data/subjects_40_v6.mat
