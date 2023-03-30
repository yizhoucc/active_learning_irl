# collect the trained agent-env rollouts
########################################

from env.env_config import Config
import numpy as np
from env import cartpole
from stable_baselines3 import PPO
import pickle
from pathlib import Path

# get all params and conditions ###############
# TODO, save the config with the model into one file.
conf = Config()

confdict = {}
for attr, value in conf.__dict__.items():
    confdict[attr] = value

# find the params we need to vary, get the combinations
vary_param_names = {}
if conf.sample_method == 'log':
    forwardfn = np.log
    backwardfn = np.exp
else:
    def forwardfn(x): return x
    def backwardfn(x): return x

for param in confdict.keys():
    if not param.endswith('range') and getattr(conf, param) is None:
        this_range = confdict['{}_range'.format(param)]
        reso = confdict['reso']
        vary_param_names[param] = backwardfn(np.linspace(
            this_range[0], forwardfn(this_range[1]), reso))

'''
we call task config param vector as phi, agent assumed param vector as theta.
there are x params, and each has r resolution (samples).
so, we have r^x for phi, and same for theta.
we have (r^x)^2 for all theta x phi combination.
in this case, we have (15^3)^2 combinations, each has about 400 ts, each ts has about 15 dim, in total this is maxmimumly 2GB data. should be good.

0328, decrease to 2 vary param just for quicker test
'''

param_order={'gravity':0, 'masscart':1, 'masspole':2, 'length':3, 'force_mag':4}
baselineparam=np.zeros((5,), dtype=np.float32)
for param in confdict.keys():
    if param in param_order and param not in vary_param_names:
        baselineparam[param_order[param]]=confdict[param]

vary_param_list=[baselineparam.copy()]
for param, samples in vary_param_names.items():
    n=len(vary_param_list)
    for i in range(n):
            thisparam=vary_param_list[i].copy()
            thisparam=np.array([thisparam]*reso)
            thisparam[:,param_order[param]]+=samples
            thisparamlist = [thisparam[i] for i in range(thisparam.shape[0])]
            vary_param_list+=thisparamlist
    vary_param_list=vary_param_list[n:]

assert len(vary_param_list)==reso**len(vary_param_names)
      
# now the vary_param_list is all the parameter in either phi and theta. we want phi x theta
# the (phi, theta)s make Y, which is the label.
Y=[]
for i in range(len(vary_param_list)):
    for p in vary_param_list:
        Y.append([vary_param_list[i], p])
assert len(Y)==(reso**len(vary_param_names))**2



# let agent interact with env ##################
env = cartpole.CartPoleEnv()
agent_name='ppo_baseline0328'
agent = PPO.load(Path('trained_agent')/agent_name)

# the time series data corresponding to Y is the input, X
X=[]
nsample=1 # sample per phi x theta combination
for phi, theta in Y:
    # data of this phi x theta combination, 
    # in format of (ts, obs+action), where obs first action last
    this_data=[] 
    for _ in range(nsample):
        obs = env.reset(phi=phi,theta=theta)
        ep_obs=[]
        ep_action=[]
        while True:
            action, _states = agent.predict(obs)
            ep_action.append(action)
            ep_obs.append(obs) # the obs before action
            obs, _reward, done, _info = env.step(action)
            if done:
                break
        ep_obs=np.array(ep_obs)
        ep_action=np.array(ep_action)
        ep_action.shape
        ep_obs.shape
        this_data.append(np.hstack([ep_obs[:,:4],ep_action.reshape(-1,1)]))
    # for more samples, it adds to the ts, by vstack
    X.append(np.vstack(this_data))
    
    # progress
    checklen=len(X)
    if checklen%100==0:
        print(checklen)

env.close()

assert len(Y)==len(X)



# saving
# TODO, integrate config into agent? or something similar
note='0329testv2'
with open('data/{}_{}'.format(agent_name, note), 'wb+') as f:
    pickle.dump((X, Y), f)

from notification import notify
notify()


    

