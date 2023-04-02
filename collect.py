# collect the trained agent-env rollouts
########################################

from notification import notify
from env.env_config import Config
import numpy as np
from env import cartpole
from stable_baselines3 import PPO
import pickle
from pathlib import Path


agent_name = 'ppo_baseline_0331_5cost'
note = 'data'  # for data saving name
nsample = 1000


# get all params and conditions ###############
# TODO, save the config with the model into one file.
conf = Config()

confdict = {}
for attr, value in conf.__dict__.items():
    confdict[attr] = value

# find the params we need to vary, get the combinations

if conf.sample_method == 'log':
    forwardfn = np.log
    backwardfn = np.exp
else:
    def forwardfn(x): return x
    def backwardfn(x): return x

reso = confdict['reso']

vary_param_names_phi = {}
vary_param_names_theta = {}
for param in confdict.keys():
    if not param.endswith('range') and getattr(conf, param) is None:
        if param.endswith('theta') or param.endswith('phi'):
            continue
        else:
            this_range = confdict['{}_range'.format(param)]
            vary_param_names_phi[param] = backwardfn(np.linspace(
                this_range[0], forwardfn(this_range[1]), reso))
# remove the fixed param in conf inverse override section
for param in confdict.keys():
    if param.endswith('phi') and getattr(conf, param) is not None:
        del vary_param_names_phi[param.replace('_phi', '')]

for param in confdict.keys():
    if not param.endswith('range') and getattr(conf, param) is None:
        if param.endswith('theta') or param.endswith('phi'):
            continue
        else:
            this_range = confdict['{}_range'.format(param)]
            vary_param_names_theta[param] = backwardfn(np.linspace(
                this_range[0], forwardfn(this_range[1]), reso))
# remove the fixed param in conf inverse override section
for param in confdict.keys():
    if param.endswith('theta') and getattr(conf, param) is not None:
        del vary_param_names_theta[param.replace('_theta', '')]


'''
we call task config param vector as phi, agent assumed param vector as theta.
there are x params, and each has r resolution (samples).
so, we have r^x for phi, and same for theta.
we have (r^x)^2 for all theta x phi combination.
in this case, we have (15^3)^2 combinations, each has about 400 ts, each ts has about 15 dim, in total this is maxmimumly 2GB data. should be good.

0328, decrease to 2 vary param just for quicker test
'''

param_order = {'gravity': 0, 'masscart': 1,
               'masspole': 2, 'length': 3, 'force_mag': 4, 'cost': 5}
baselineparam = np.zeros((len(param_order),), dtype=np.float32)
for param in param_order.keys():
    if param not in vary_param_names_phi:
        value = confdict[param]
        if value is None:
            value = confdict[param+'_phi']
        baselineparam[param_order[param]] = value

vary_param_list_phi = [baselineparam.copy()]
for param, samples in vary_param_names_phi.items():
    n = len(vary_param_list_phi)
    for i in range(n):
        thisparam = vary_param_list_phi[i].copy()
        thisparam = np.array([thisparam]*reso)
        thisparam[:, param_order[param]] += samples
        thisparamlist = [thisparam[i] for i in range(thisparam.shape[0])]
        vary_param_list_phi += thisparamlist
    vary_param_list_phi = vary_param_list_phi[n:]

vary_param_list_theta = [baselineparam.copy()]
for param, samples in vary_param_names_theta.items():
    n = len(vary_param_list_theta)
    for i in range(n):
        thisparam = vary_param_list_theta[i].copy()
        thisparam = np.array([thisparam]*reso)
        thisparam[:, param_order[param]] += samples
        thisparamlist = [thisparam[i] for i in range(thisparam.shape[0])]
        vary_param_list_theta += thisparamlist
    vary_param_list_theta = vary_param_list_theta[n:]

assert len(vary_param_list_theta) == reso**len(vary_param_names_theta)

# now the vary_param_list is all the parameter in either phi and theta. we want phi x theta
# the (phi, theta)s make Y, which is the label.

Y = []
for phi in vary_param_list_phi:
    for theta in vary_param_list_theta:
        Y+=[[phi, theta]]*nsample


# Y = []
# for i in range(len(vary_param_list)):
#     for p in vary_param_list:
#         Y += [[vary_param_list[i], p]]*nsample
assert len(Y) == (reso**len(vary_param_names_phi))*(reso**len(vary_param_names_theta))*nsample


# let agent interact with env ##################
env = cartpole.CartPoleEnv()

agent = PPO.load(Path('trained_agent')/agent_name)

# the time series data corresponding to Y is the input, X
X = []  # sample per phi x theta combination
for phi, theta in Y:
    # data of this phi x theta combination,
    # in format of (ts, obs+action), where obs first action last
    this_data = []
    for _ in range(1):
        obs = env.reset(phi=phi, theta=theta)
        ep_obs = []
        ep_action = []
        while True:
            action, _states = agent.predict(obs)
            ep_action.append(action)
            ep_obs.append(obs)  # the obs before action
            obs, _reward, done, _info = env.step(action)
            if done:
                break
        ep_obs = np.array(ep_obs)
        ep_action = np.array(ep_action)
        ep_action.shape
        ep_obs.shape
        this_data.append(
            np.hstack([ep_obs[:, :4].astype(np.float32), ep_action.reshape(-1, 1)]))
    # X.append(np.vstack(this_data).astype(np.float32))
    X += this_data

    # progress
    checklen = len(X)
    if checklen % 100 == 0:
        print(checklen)

env.close()

assert len(Y) == len(X)


# saving
# TODO, integrate config into agent? or something similar

with open('data/{}_{}'.format(agent_name, note), 'wb+') as f:
    pickle.dump((X, Y), f)

notify()
