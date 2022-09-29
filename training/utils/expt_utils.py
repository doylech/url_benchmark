import ruamel.yaml as yaml
import pathlib
import elements
import numpy as np

# import neptune.new as neptune
import neptune

def get_latest_expt_id(neptune_project):
    """
    Queries list of experiments logged in Neptune.ai,
    and gets the final experiment.
    Assumes that the most recent experiment has not been deleted.
    Args:
        neptune_project: Neptune project object
    """
    print('Getting experiments from neptune')
    # run = neptune.get_last_run()
    # id = run.get_url().split('/')[-1]
    # return id

    expts = neptune_project.get_experiments()
    ids = [x.id for x in expts]
    ind = np.argsort(np.array([int(x.split('-')[1]) for x in ids]))[-1]
    id = ids[ind]
    return id
    #
    #
    # # ids = [int(x.id.split('-')[1]) for x in expts]
    # # expt = expts[np.argsort(np.array(id))[-1]]
    # # expt = neptune_project.get_experiments()[-1]
    # # return expt.id


def get_next_expt_id(neptune_project):
    """
    Queries list of experiments logged in Neptune.ai and
    predicts the id of the next experiment.
    Assumes that the most recent experiment has not been deleted.
    """
    prev_expt_id = get_latest_expt_id(neptune_project)
    expt_id = '-'.join([prev_expt_id.split('-')[0],
                        str(int(prev_expt_id.split('-')[-1]) + 1)])
    return expt_id
