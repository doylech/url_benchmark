"""Launch a training experiment either on remote or local machine."""
"""Not yet implemented."""

import subprocess
import os
import inspect
import neptune
import getpass

import training
import training.utils.expt_utils as exu
import training.utils.remote_utils as ru


def launch_expt(neptune_project, expt_fn,
                do_use_remote, which_remote=None,
                use_docker_on_remote=True,
                expt_id=None,
                gcp_zone=None,
                user='ikauvar'):
    """
    Automatically commit code, and launch experiment,
    either locally or on remote machine.
    Note: You must VPN into Stanford network to access a Stanford server.
    Note: You must have an ssh keypair setup with the remote computer,
          and the local and remote computer must have ssh keypairs
          set up with github.

    Args:
        neptune_project: Neptune object (output from neptune.init())
        expt_fn: handle to main() function in expt script.
        do_use_remote: bool. Whether to run locally or on server.
        which_remote: select server. 'node19', 'gse73'
        use_docker_on_remote: bool. Whether to use docker (which may be necessary
                              for full usage of the GPU with proper CUDA version).
        task_name: this is specific to launching Dreamer, specify task here. i.e. 'dmc_walker_walk'
        Note: Remote and local computer must each have
              ssh keypair setup with github, and
              an ssh keypair with each other.
    """
    if do_use_remote:
        use_gcp = False
        if which_remote == 'node19':
            remote_hostname = os.environ.copy()['REMOTE_HOSTNAME19'] # user@remote.address.com, see ~/.bash_profile
            base_path = f'/home/{user}/'
            python_path = f'/home/{user}/miniconda3/envs/gendreamer/bin/python'
            use_docker_on_remote = True
        elif which_remote == 'gse73':
            remote_hostname = os.environ.copy()['REMOTE_HOSTNAME']
            base_path = f'/home/{user}/'
            python_path = f'/home/{user}/anaconda3/envs/gendreamer/bin/python'
            use_docker_on_remote = True
        elif which_remote == 'saal1':
            remote_hostname = os.environ.copy()['REMOTE_HOSTNAMEsaal1']
            base_path = '/home/saal1/'
            python_path = '/home/saal1/miniconda3/envs/gendreamer/bin/python'
            use_docker_on_remote = False
        elif 't4-tf' in which_remote or 'p4-tf' in which_remote or 'v100-tf' in which_remote:
            remote_hostname = which_remote
            base_path = f'/home/{user}/'
            python_path = f'/home/{user}/miniconda3/envs/gendreamer/bin/python'
            use_docker_on_remote = False
            use_gcp = True
            if which_remote.split('-')[0] == 't4':
              ports = {'2f':20, '3f':21, '4f':22, '5f':23, '6f':24, '7f':25, '8f':26, '9f':27,
                       '1': 17, '10': 18, '2':7, '3':8, '4':9, '5':10, '6':11, '7':12, '8':13, '9':14, '11': 15, '12': 16,
                       '2b':15, '3b':16}
            elif which_remote.split('-')[0] == 'v100':
              ports = {'1f': 17}
            port = str(6000 + ports[which_remote.split('-')[-1]])
        elif 'cd-gendreamer' in which_remote:
            remote_hostname = which_remote
            base_path = f'/home/{user}/'
            python_path = f'/home/{user}/miniconda3/envs/gendreamer/bin/python'
            use_docker_on_remote = False
            use_gcp = True
            port = str(6000 + int(which_remote.split('-')[-1]))
    else:
        remote_hostname = None
        python_path = None

    if do_use_remote and remote_hostname is not None:
        if use_gcp:
            ru.test_remote_connection_gcp(remote_hostname, gcp_zone, user)
        else:
            ru.test_remote_connection(remote_hostname)

    if expt_id is None:
        print('---> Getting next_expt_id.')
        expt_id = exu.get_next_expt_id(neptune_project)
        print('---> Committing.')
        commit_id, git_branch = ru.git_commit(expt_id=expt_id)
    else:
        print('---> Committing.')
        commit_id, git_branch = ru.git_commit(expt_id=f'{expt_id} (again)')

    print(commit_id, git_branch)
    expt_script_path = os.path.abspath(inspect.getfile(expt_fn))
    expt_script_name = expt_script_path.split('/')[-1]

    if do_use_remote and remote_hostname is not None:
        print('---> Launching remote experiment.')
        file_dir = os.path.dirname(os.path.realpath(__file__)) # Get current dir (on remote).
        if use_docker_on_remote:
            launch_script_path = os.path.join(file_dir, 'launch_remote_expt_docker.sh')
            # Set path to script to be compatible with docker: it needs to be relative
            # to the top folder of the project repository.
            # repo_path = '/'.join(os.path.dirname(training.__file__).split('/')[:-1])
            repo_path = os.path.dirname(training.__path__[0])
            expt_script_name = os.path.relpath(expt_script_path, repo_path)
        else:
            launch_script_path = os.path.join(file_dir, 'launch_remote_expt.sh')
            launch_tensorboard_path = None
        if use_gcp:
            launch_script_path = os.path.join(file_dir, 'launch_remote_expt_gcp.sh')
            launch_tensorboard_path = os.path.join(file_dir, 'launch_tensorboard_gcp.sh')

        if use_gcp:
            out = subprocess.Popen(["sh", launch_tensorboard_path,
                                    remote_hostname,
                                    gcp_zone,
                                    expt_id,
                                    port],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        print(f'Launching {expt_script_name}')
        out = subprocess.Popen(["sh",  launch_script_path,
                                remote_hostname,
                                expt_script_name,
                                expt_id,
                                git_branch,
                                python_path,
                                gcp_zone,
                                user],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        try:
            print(out.stderr.readlines())
        except:
            pass

        result = out.stdout.readlines()

        if not result:
            error = out.stderr.readlines()
            raise ConnectionRefusedError(error, '---> Is VPN connected? ')
        else:
            print(result)

    else:
        # Run experiment locally
        print('---> Launching local experiment.')
        if expt_id is None:
            expt_id = exu.get_next_expt_id(neptune_project)
        # logdir = os.path.join(os.getenv('HOME'), 'logs')
        logdir = os.path.join(os.getenv('HOME'), 'logs', str(expt_id))
        os.makedirs(logdir, exist_ok=True)
        expt_fn(logdir=logdir, do_neptune=True)


if __name__=="__main__":
    from training.scripts import dreamer_launch, sb3_launch
    user = getpass.getuser()
    if user == 'cd':
        remote = 'cd-gendreamer-1'
        #remote = 'cd-gendreamer-exp-2'
        #remote = 'cd-gendreamer-n16-1'
        if remote[14:17] == 'n16' or remote[14:17] == 'exp':
            gcp_zone = 'us-west1-b'
        else:
            gcp_zone = 'us-central1-a'
    elif user == 'saal2':
        user = 'ikauvar'
        remote = 't4-tf-2' #t4-tf-4', # 'gse73'
        gcp_zone = 'us-central1-a'
        expt_fn = dreamer_launch.main
        # expt_fn = sb3_launch.main

    project = f'Autonomous-Agents/dra' # dre, GEN, sandbox
    launch_expt(neptune_project=neptune.init(project),
                expt_fn=expt_fn,
                do_use_remote=True,
                which_remote=remote,
                gcp_zone=gcp_zone,
                # gcp_zone="us-west1-b",
                user=user,
                # expt_id='SMS-822',
                )

