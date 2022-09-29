# Modify and run this script to download checkpoints from remote server

id=DRA-159
server=t4-tf-1
zone=us-central1-a
mkdir ~/logs/$id
#scpgcp $server $zone /home/ikauvar/logs/$id/variables*.pkl ~/logs/$id/.
scpgcp $server $zone /home/ikauvar/logs/$id/variables*15*.pkl ~/logs/$id/.
scpgcp $server $zone /home/ikauvar/logs/$id/variables*30*.pkl ~/logs/$id/.
scpgcp $server $zone /home/ikauvar/logs/$id/for_ckpt ~/logs/$id/.
#scpgcp $server $zone /home/ikauvar/logs/$id/train_replay_0 ~/logs/$id/train_replay_0
scpgcp $server $zone /home/ikauvar/logs/$id/eval_replay_0 ~/logs/$id/train_replay_0
#



#scpgcp $server $zone /home/ikauvar/logs/$id/variables_train_agent.pkl ~/logs/$id/.
#scpgcp $server $zone /home/ikauvar/logs/$id/variables_train_agent_explb.pkl ~/logs/$id/.
#scpgcp $server $zone /home/ikauvar/logs/$id/variables_train_agent_taskb.pkl ~/logs/$id/.
#scpgcp $server $zone /home/ikauvar/logs/$id/variables_train_agent_wm.pkl ~/logs/$id/.


  ids = {  # freerange, black
    'DRE-354': 't4-tf-2', # black
    'DRE-346': 't4-tf-8', # black
    'DRE-348': 't4-tf-9', # black
    'DRE-379': 't4-tf-4',  # black
  }
    #   'DRE-377': 't4-tf-1',
