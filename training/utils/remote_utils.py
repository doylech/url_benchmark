"""Utilities for connecting to and sharing code with remote machines."""
import subprocess
import git


def send_remote_command(host):
    """
    Ssh into a remote server and execute commands.
    This function is not currently used and
    remains here for reference.
    A bash script is currently used instead of this.
    """
    ssh = subprocess.Popen(["ssh", host],
                           stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           universal_newlines=True,
                           bufsize=0)

    # Send ssh commands to stdin
    ssh.stdin.write("uname -a\n")
    ssh.stdin.write("cd src/curio_sorb/\n")
    ssh.stdin.write("pwd\n")
    ssh.stdin.write("git pull\n")
    ssh.stdin.write("/bin/sh $HOME/.bashrc\n")
    ssh.stdin.close()

    # Fetch output
    for line in ssh.stdout:
        print(line.strip())


def git_commit(expt_id=None):
    """
    Commit to the current branch with a message
    containing the experiment id.
    """
    try:
        # repo = git.Repo(os.getcwd())
        repo = git.Repo(search_parent_directories=True)

        message = f'Running expt: {expt_id}'
        repo.git.commit('--allow-empty', '-am', message)
        commit_id = repo.head.object.hexsha
        branch = repo.active_branch
        branch = branch.name
        origin = repo.remote(name='origin')
        origin.pull()
        origin.push()
    except Exception as e:
        print('Error occurred while pushing the repo: ', e)

    print(f'---> Git commit to branch {branch}: {message}')
    return commit_id, branch


def test_remote_connection(remote_hostname):
    """
    Establish ssh connection with remote server.
    If cannot do, raises an error reminding the user
    to check that the VPN is activated.
    Args:
        remote_hostname: i.e. username@remote.server.com
    """

    print('---> Testing remote connection.')
    try:
        ssh = subprocess.run(["ssh", remote_hostname, "uname -a"],
                             timeout=2, stdout=subprocess.PIPE)
        print("---> Connection to remote established:")
        print(ssh.stdout.decode("utf-8"))
    except subprocess.TimeoutExpired as e:
        raise Exception('Error connecting to remote server, is VPN activated?  ', e)
    
def test_remote_connection_gcp(remote_hostname, gcp_zone, user):
    """
    Establish ssh connection with remote server.
    If cannot do, raises an error reminding the user
    to check that the VPN is activated.
    Args:
        remote_hostname: i.e. username@remote.server.com
    """

    print('---> Testing remote connection.')
    try:
        ssh = subprocess.run(["gcloud", "beta", "compute", "ssh", f"{user}@{remote_hostname}",
                              "--zone", gcp_zone, "--project", "hai-gcp-artificial",
                              "--command", "uname -a"],
                             timeout=12, stdout=subprocess.PIPE)
        print("---> Connection to remote established:")
        print(ssh.stdout.decode("utf-8"))
    except subprocess.TimeoutExpired as e:
        raise Exception('Is GCP instance running? ', e)