### Launch training experiment on remote host
### Args:
### 1) Hostname (i.e. user@remote.address.com ) (i.e. should be saved as env var $REMOTE_HOSTNAME)
### 2) Zone of server (i.e. us-central1-a)
### 3) Experiment id (i.e. IGS-100) for naming log directory.
### 4) Checkpoint name
### 5) Path to python (i.e. refers to python in a conda env)

gcloud beta compute ssh ikauvar@$1 --zone $2  --project "hai-gcp-artificial" \
      --command "$5 ~/src/gendreamer/training/analysis_scripts/save_wm_from_agent_ckpt.py --expid $3 --ckpt $4"

