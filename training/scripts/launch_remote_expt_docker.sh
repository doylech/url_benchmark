#!/bin/bash
### Launch training experiment on remote host
### Args:
### 1) Hostname (i.e. user@remote.address.com ) (i.e. should be saved as env var $REMOTE_HOSTNAME)
### 2) Train script file name (i.e. train_model.py)
### 3) Experiment id (i.e. IGS-100) for naming log directory.
### 4) Git branch to use (i.e. ik_expts)
### 5) Path to python (i.e. refers to python in a conda env)

ssh -A $1 "source ~/.bash_profile  \
        && cd ~/src/gendreamer \
        && pwd \
        && git fetch origin \
        && git checkout $4 \
        && git branch \
        && git pull \
        && pwd \
        && mkdir /home/ikauvar/logs/$3 \
        || echo 'logdir already exists' \
        && sh docker_run_py.sh \
        $2  \
        --logdir /logs/$3 \
        >> /home/ikauvar/logs/$3/$3.out 2>&1 "

# OLD
#        --logdir /logs/$3 \
#        --configs defaults dmc \
#        --task $6 \
#        >> /home/ikauvar/logs/$3/$3.out 2>&1 "
# Note: the logdir is the path inside the docker container.
