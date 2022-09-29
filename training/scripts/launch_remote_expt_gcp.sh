#!/bin/bash
### Launch training experiment on remote host
### Args:
### 1) Remote address (i.e. in user@remote.address.com ) (i.e. should be saved as env var $REMOTE_HOSTNAME)
### 2) Train script file name (i.e. train_model.py)
### 3) Experiment id (i.e. IGS-100) for naming log directory.
### 4) Git branch to use (i.e. ik_expts)
### 5) Path to python (i.e. refers to python in a conda env)
### 6) Zone of server (i.e. us-central1-a)
### 7) User (login on the remote)

#sshgcp $1 $6 "source ~/.bash_profile  \
gcloud beta compute ssh $7@$1 --zone $6  --project "hai-gcp-artificial" \
      --command "source ~/.bashrc  \
        && source ~/.bash_profile  \
        && echo \$NEPTUNE_API_TOKEN \
        && cd ~/src/gendreamer/training/scripts \
        && pwd \
        && git fetch origin \
        && git checkout $4 \
        && git branch \
        && git pull \
        && pwd \
        && mkdir ~/logs/$3 \
        || echo 'logdir already exists' \
        && PYTHONPATH=/home/cd/src/gendreamer DISPLAY=:0 $5 \
        $2  \
        --logdir /home/$7/logs/$3 \
        >> ~/logs/$3/$3.out 2>&1 "


  ###         && sudo killall Xorg \
  #        && (sudo /usr/bin/X :0 &) \
