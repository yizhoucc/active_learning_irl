# todo
- potential new task if this is not working.

# current res
rrn baseline: good. cost doesnt matter but cost makes theta length matter

rnn prob: not good.

# proposal

## intro
background
based on the idea of inverse rational control.
inverse rational control is a subset of inverse reinforcement learning, where we assume the subject or agnet to be suboptimal but still rational.
generally, the suboptimality comes from wrong assumptions of the actual task dynamics and control preferrences.
it can also comes from biological limitations such as represetnation and computation cost, so far we havent model that in the current version of irc.
comparing to the other irl methods, irc has 3 benifits in neuroscience/cognition field.
1, we do not assume subjects to be near optimal. instead, we assume they are rational and have individual differences.
2, irc infers latent assumption, instead of a hard to inteprate reward function and policy.
3, irc handels complex problems, whereas some inverse optimal control methods achieves 1 and 2 but can only be applied to simple problems.

goal that we try to achieve.
however, limitations exist.
first, irc relies on solving the pomdp in the reverse order, which often involves sampling and marginalization.
the inverse model is not pre exposed to the samples, and we have to do this for each piece of data to evaulate the likelihood.
this brings up efficiency concerns.
second, the irc needs quite a lot of data to startin running.
for one trial data, we wont be able update our theta estimation in a meaningful way.
this is near random exploration.

proposed way.
here i propose a potential upgrade to solve these 2 problems.
i believe the fundamental problems is the sampling.
we use the likelihood function and the current actual data to check if the current theta estimation reproduces samples that are similar to the actual data.
in other words, sharing similar pattern.
if we have a model that learns the pattern of this theta estimation ahead of time, we can direclty evaluate the likelihood without doing more samples.
importantly, we will be able to select the best task to better infer the theta.
if there exists such a model that takes the trajectory time series input and output to inferred theta, the model should twist the hidden representation of the trajectory manifold into a uniform grid theta and phi, and there exists a recurrent submodel that holds a represetntaion of the times series.
using this submodel, we can embed the trial data, and calculate the information content of theta offered by a task phi.
selecting a max information task phi could potentially accerate the irc.
if we think the inverse problem as a learning problem.
we will find the subject is the enviorment and experimenters are the agent, trying to build a subject model (like the world model in dreamer rl) from the interactions.
most previous methods assume the data is precollected, in the replay buffer, so its like the agent is passively learning not actively explore in this setup.
however, we know that from rl, once the agent develops some kind of inaccurate imaginary world model, active learning is almost a must to learn efficiently.
here, i believe for this inverse problem, we are not start from scratch, and active learning can greatly accerate things.
in summary of this new approach, we use the network inference as inverse (likelihood funciton) and update the estimation of theta (baysien optimization), and solve a smaller optimization problem to select the best phi in terms of information. we to thsese two steps iteratively till converge.



# methods

## notations

- $*$ stands for actual data/ground truth data
- $\hat{}$ stands for estimations
- $\theta$  latent assumptions of the task
- $\phi$ task configurations
- $T$ time series trajectories data

## part 1, inference as inverse
Let $F$ be a neural network, reverse function of POMDP, such that 

$$ F(T_{\theta,\phi}, \phi) = Decoder(Encoder(T_{\theta,\phi},\phi)) = p(\theta) $$

the output of the function $F$ is in

$$ \theta \sim N(\mu_{\theta}, \Sigma_{\theta}) $$




## task

demo task, cartpole

## related work
structed state space model. for long time series embedding
https://srush.github.io/annotated-s4/#discrete-time-ssm-the-recurrent-representation
https://www.youtube.com/watch?v=ugaT1uU89TA

https://mp.weixin.qq.com/s/sTlUBXB-PVKI2l0HqRStSA


deep active learning
https://arxiv.org/pdf/2009.00236.pdf

active irl
https://arxiv.org/abs/2207.08645

vae math
https://davidstutz.de/the-mathematics-of-variational-auto-encoders/

transdreamer
https://arxiv.org/pdf/2202.09481.pdf



## data format

### collected simulation data
we call task config param vector as phi, agent assumed param vector as theta.
there are x params, and each has r resolution (samples).
so, we have r^x for phi, and same for theta.
we have (r^x)^2 for all theta x phi combination.
in this case, we have (15^3)^2 combinations, each has about 400 ts, each ts has about 15 dim, in total this is maxmimumly 2GB data. should be good.

## complexity analysis 
```
        x, storage: reso*(2n) n is number of params
            this is the major limitation. large storage, complex increase in power order with nparameter increase.
            the previous pomdp likelihood appraoch, when nparam increase, problem complexity linearly.
        backward function training: when reso is large, have to use linear readout
        inference for theta. best way is softamx or some other way to model the probability.
        marginalize for phi: for all theta for all phi, find phi that a delta theta [000100] x abs(dervitative of tau | phi) is max. 
            delta theta [000100], index one theta among all theta, to be weighted by current p(theta)
            abs(dervitative of tau | phi). given a theta, now only theta affects tau. we can find the derivative of dtheta/dtau, meaning how much change in theta affact change in tau. here tau can be later layers of the network. in short, we find a phi that makes tau is most sensitive to theta change at a particular theta_estimation.
            computatio, in the worst case, we evaluate all theta, all phi, and calculate the gradient. but in practice, we can dynamically adjust the resolution of theta and phi. eg, given a theta, do something like a binsec, and got longer trials are better phi. from there, we can either random with the current knowledge, or continue binsec.
            to acc this, we can 1, process the data storage before hand, 2 keep some binsec point in menonry and other in storage, 
        
```

# other notes
```
[Notification]

token = xxxbarktokenxxx


[Datafolder]
data = /data, some mapped dir for the dev container
```


