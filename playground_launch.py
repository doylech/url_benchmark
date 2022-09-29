import common

config = ['--configs', 'defaults', 'dmc', '--seed', '1']
env_name = 'admc_sphero_multiagent_dense_goal'
env = common.Playground(env_name, config)
eval_env = common.Playground(env_name, config)

obs = env.reset()
for i in range(500):
    obs, reward, done, info = env.step(env.action_space.sample())
    print(obs.mean())