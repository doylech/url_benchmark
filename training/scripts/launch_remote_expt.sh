#!/bin/bash
### Launch training experiment on remote host
### Args:
### 1) Hostname (i.e. user@remote.address.com ) (i.e. should be saved as env var $REMOTE_HOSTNAME)
### 2) Train script file name (i.e. train_model.py)
### 3) Experiment id (i.e. IGS-100) for naming log directory.
### 4) Git branch to use (i.e. ik_expts)
### 5) Path to python (i.e. refers to python in a conda env)


ssh $1 "source ~/.bash_profile  \
        && cd ~/src/gendreamer/training/scripts \
        && pwd \
        && git fetch origin \
        && git checkout $4 \
        && git branch \
        && git pull \
        && pwd \
        && mkdir ~/logs/$3 \
        || echo 'logdir already exists' \
       ############### && xvfb-run -a -s '-screen 0 1400x900x24' bash \
        && $5 \
        $2  \
        --logdir $3 \
        >> ~/logs/$3/$3.out 2>&1 "