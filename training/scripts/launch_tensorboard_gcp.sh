### Launch training experiment on remote host
### Args:
### 1) Hostname (i.e. user@remote.address.com ) (i.e. should be saved as env var $REMOTE_HOSTNAME)
### 2) Zone of server (i.e. us-central1-a)
### 3) Experiment id (i.e. IGS-100) for naming log directory.
### 4) Port for tensorboard ssh

gcloud beta compute ssh ikauvar@$1 --zone $2  --project "hai-gcp-artificial" \
      --command "bash ~/src/gendreamer/setup_tboard.bash $4 $3"
