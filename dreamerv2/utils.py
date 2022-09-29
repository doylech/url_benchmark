import ruamel.yaml as yaml
import pathlib
import elements
import dreamerv2
import numpy as np

# import neptune.new as neptune
import neptune
import elements

def parse_flags(flags):
    """
    Load configs and update parameters according to flags.
    :param flags: List of strings, i.e. ['--name', 'value']
    :return: config object, logdir
    """
    configs = pathlib.Path(__file__).parent / 'configs.yaml'
    configs = yaml.safe_load(configs.read_text())
    config = elements.Config(configs['defaults'])
    parsed, remaining = elements.FlagParser(configs=['defaults']).parse_known(
      argv=flags, exit_on_help=False)
    for name in parsed.configs:
      config = config.update(configs[name])
    config = elements.FlagParser(config).parse(argv=remaining)
    logdir = pathlib.Path(config.logdir).expanduser()
    return config, logdir