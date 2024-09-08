import numpy as np
import os
import collections
from os.path import dirname, abspath, join
from copy import deepcopy
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import REGISTRY as run_REGISTRY

import wandb

logger = get_logger()

results_path = join(dirname(dirname(abspath(__file__))), "results")


# @ex.main
# def my_main(_run, _config, _log):
#     wandb.init(project='MARL', config=_config, name='QMIX_test')
#     # Setting the random seed throughout the modules
#     config = config_copy(_config)
#     np.random.seed(config["seed"])
#     th.manual_seed(config["seed"])
#     config['env_args']['seed'] = config["seed"]
    
#     # run
#     if "use_per" in _config and _config["use_per"]:
#         run_REGISTRY['per_run'](_run, config, _log)
#     else:
#         run_REGISTRY[_config['run']](_run, config, _log)

#     wandb.finish()
class MockRun:
    """ A mock class to simulate Sacred _run object """
    def __init__(self):
        self.info = {}


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=')+1:].strip()
            break
    return result


if __name__ == '__main__':
    
    params = deepcopy(sys.argv)

    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)


    if "seed" not in config_dict:
        config_dict["seed"] = 42  
    if "env_args" not in config_dict:
        config_dict["env_args"] = {}
    if "map_name" not in config_dict["env_args"]:
        config_dict["env_args"]["map_name"] = "default_map"  


    wandb.init(project='MARL', config=config_dict, name='QMIX_test')


    config = wandb.config

    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # Save to disk by default for sacred
    map_name = parse_command(params, "env_args.map_name", config["env_args"]["map_name"])
    algo_name = parse_command(params, "name", config["name"])
    file_obs_path = join(results_path, "wandb", map_name, algo_name)
    
    logger.info("Saving to wandb in {}.".format(file_obs_path))

    run_config = config_copy(config_dict)
    mock_run = MockRun()

    if "use_per" in config_dict and config_dict["use_per"]:
        run_REGISTRY['per_run'](mock_run, run_config, logger)
    else:
        run_REGISTRY[config_dict['run']](mock_run, run_config, logger)

    wandb.finish()

    # flush
    sys.stdout.flush()
