# todo
- potential new task if this is not working. see task section.
- implement part 2

# current res
rrn baseline: good. cost doesnt matter but cost makes theta length matter

rnn prob: good, with gradient clip. 

inferrence as inverse: good. need about 30 len trajectory and does one shot inverse.
need some noise to make this slightly harder, but not too much harder for a demo task.


# proposal

## intro
### background
based on the idea of inverse rational control.
inverse rational control is a subset of inverse reinforcement learning, where we assume the subject or agnet to be suboptimal but still rational.
generally, the suboptimality comes from wrong assumptions of the actual task dynamics and control preferrences.
it can also comes from biological limitations such as represetnation and computation cost, so far we havent model that in the current version of irc.
comparing to the other irl methods, irc has 3 benifits in neuroscience/cognition field.
1, we do not assume subjects to be near optimal. instead, we assume they are rational and have individual differences.
2, irc infers latent assumption, instead of a hard to inteprate reward function and policy.
3, irc handels complex problems, whereas some inverse optimal control methods achieves 1 and 2 but can only be applied to simple problems.

### goal that we try to achieve.
however, limitations exist.
first, irc relies on solving the pomdp in the reverse order, which often involves sampling and marginalization.
the inverse model is not pre exposed to the samples, and we have to do this for each piece of data to evaulate the likelihood.
this brings up efficiency concerns.
second, the irc needs quite a lot of data to startin running.
for one trial data, we wont be able update our theta estimation in a meaningful way.
this is near random exploration.

### proposed way.
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

- $^*$ stands for actual data/ground truth data
- $\hat{}$ stands for estimations
- $\theta$  latent assumptions of the task
- $\phi$ task configurations
- $T$ time series trajectories data

## part 1, inference as inverse
Let $F$ be a neural network, reverse function of POMDP, such that 

$$ F(T_{\theta,\phi}, \phi) = Decoder(Encoder(T_{\theta,\phi},\phi)) = p(\theta) $$

the output of the function $F$ is probobalistic:

$$\theta \sim N(\mu_{\theta}, \Sigma_{\theta})$$

the objective function is given by:


$$ \arg\min_{\phi} \;\; \left \langle {\text{KL}\left(\; p(\theta \mid \theta^*) \;\middle\|\; q(\theta \mid T_{\theta^*,\phi}, \phi) \;\right)} \right \rangle_{T,\theta^*} $$



usually we do not model $p(\theta\mid \theta^*)$. 
instead, $p(\theta\mid \theta^*)$ naturally exist because due to stochasity there are some other $\theta$ (usually nearby) besides $\theta^*$ can produce the same trajectory $T$.

the objective function is to minimize the KL divergence of such $p(\theta \mid \theta^*)$ distribution (targets) and the network's predicted $p(\theta)$ distribution given the trajectories (input data). 
the objective function averages over two things.
first, it averages over the different samples of $T \mid \theta^*,\phi$.
second, it averages over the entrire possible assumptions range, $\theta^* \sim \Theta$.

most of the time we just model the $p(\theta\mid \theta^*)$ to be a delta distribution.
instead of using KL divergence we just maximize the probabiliy of $\theta^*$ in $p(\theta) = F(T_{\theta,\phi}, \phi)$ which is a gaussian distribution $N({\mu}_{\theta}, {\Sigma}_{\theta})$.

$$ p(\theta = \theta^*) =$$

$$ p(\theta^* \mid  N(\mu_{\theta}, {\Sigma}_{\theta})) = \frac{1}{\sqrt{(2\pi)^n \det({\Sigma}_{\theta})}} \exp\left(-\frac{1}{2}(\theta^* - {\mu}_{\theta})^T {\Sigma}_{\theta}^{-1} (\theta^*-{\mu}_{\theta})\right)
 $$

minimizing the negative log probabiliy over all $p(\theta^*)$ yields the trained function $F$.

explain:
the idea is really like the variational autoencoder.
the input is the trajectories $T$, and the latent variable is the $\theta$.
in this case, we do not have to reconstruct to the oringal input because we known the latent space already.
we know exactly $p(\theta)$ and $p(T\mid \theta)$
so, instead of training for reconstruction, we train the "encoder" part directly.
here we do have both encoder and decoding by naming.
the encoder here refers to the recurrent hidden states embedding of the time series data, and the decoder here refers to the read out layers from the trajectory representation vector to the output.

## part 2, active IRC

the goal of the active IRC is to select $\phi_i \sim \Phi$ such that $I(\theta;T_{\theta,\phi_i})$ is maxed.
the mutual information $I(\theta;T_{\theta,\phi_i})$ is usually hard to calcualte except from the discrete case, so we use the information gain instead.
the information gain is the KL divergence between the previous estimated agent hidden assumption given all previous data, $p(\theta\mid T_{\theta,\phi_i\; 0:t})$, and the updated estimation given the new trial data $(\theta\mid T_{\theta,\phi_i\; 0:t+1})$. 
if the $T_t+1$ is independent from previous trial history $T_0:t$, then we have:

$$ IG = \text{KL}(p(\theta) \; \mid \mid \;  p(\theta\mid  T_{\theta,\phi_i\; 0:t}) \cdot p(\theta\mid  T_{\theta,\phi_i\; t+1}) )$$

since update follows Baysien, the likelihood $p(T_{\theta,\phi_i}\mid \theta)$ is the new information.
the fisher information can be calculated by

$$ J(\theta) = -\nabla \nabla \ell(T_{\theta,\phi_i}\mid  \theta) $$

$$  = - \langle {\frac{d^2}{d^2\theta} \ell(T_{\theta,\phi_i}\mid  \theta)} \rangle_{T|\theta} $$

$$  = - \langle{ \frac{d^2}{d^2\theta} \ell(T_{\phi_i}(\theta)) \rangle_{T|\theta}} $$

where the $T_{\theta,\phi_i}\mid  \theta$ are the previously sampled trajectories, average over the $\theta$ range of interest.
to calculate the derivative of log of trajectories, we use the embedding part of the trained function $F$ in part 1. 
however, the $p(T_{\phi_i}(\theta))$ is hard to calculate, because the trajectory generation depends on POMDP, resulting in a unknown trajectory probabiliy.
we would need to do a lot of samples and clustering for each case to determine the likelihood precisely.

option 1.
instead of treating the observation to be $T_{\phi_i}(\theta)$, we use observation $=p(\hat{\theta} \mid (T_{\phi_i}(\theta))=\text{F}(T_{\phi_i}(\theta)))$
here we replace the trajectory samples with 'processed' trajectory samples.
they are 'processed' by our part 1 neural network to output a gaussian distribution of estimated $\theta$.
this process shapes the unknown distribution of trajectories into a known gaussian distribution.
however, there could be some information loss during this process.

$$ J(\theta) \ge \hat{J}(\theta) = - \langle{ \frac{d^2}{d^2\theta} \ell[p(\hat{\theta} \mid (T_{\phi_i}(\theta))) \mid \theta] \rangle_{T|\theta}} $$

where the $\hat{J}(\theta)$ is the estimated fisher information.
the 'processed' samples are arbitrary estimators of the latent $\theta$. 
the lower bound of variance of an arbitrary estimator is defined by Cramer-Rao lower bound

$$ n \cdot \text{Var}_{\theta}(\hat{\theta}) \ge \frac{1}{I(\theta)} $$

$$  I(\theta) \ge \frac{1}{n \cdot \text{Var}_{\theta}(\hat{\theta})} $$

where

$$ \text{Var}_{\theta}(\hat{\theta}) = \text{Var}_{\theta}( p(\hat{\theta} \mid (T_{\phi_i}(\theta))) ) $$

finally, we have a upper bound for estimated fisher information.there are 2 concerns. 
first, does max the upper bound of fisher information achieves the same optimization goal in the end for max the fisher inforamtion directly?
the intuitive answer is, maybe not. 
one thing to argue is under gaussian distribution, the sampled upper bound has a fix relationship to the actual fisher info.
some 1/n+1 1/n relationship.
second, does maximizing the estimated fisher information the same as maximizing the actual fisher information?



option 2.
instead of evaluating the $p(T_{\phi_i}(\theta))$, we can use the distance of each trajectory to other trajectories as a indicator for the likelihood.
the idea is, if the trajectory is very different from the others, its probability is low.
if the trajectory is very similar to the others, its probability is high.
let $\hat{\ell}$ be such function that approximate the probability using distance.

$$ J(\theta) = - \langle{ \frac{d^2}{d^2\theta} \hat{\ell}(T_{\phi_i}(\theta))} \rangle_{T|\theta} $$

option 3.
we use baysien rule to change the likelihood.

$$ J(\theta) = - \langle{ \frac{d^2}{d^2\theta} \ell(T_{\theta,\phi_i}\mid  \theta)} \rangle_{T|\theta} $$

$$ = - \langle{ \frac{d^2}{d^2\theta}  \frac{p(\theta \mid T_{\theta,\phi_i}) \cdot p(T_{\theta,\phi_i}) }{p(\theta)}   } \rangle_{T|\theta} $$

because $\theta$ inside the expectation is uniform distributed, we have

$$ = - \langle{ \frac{d^2}{d^2\theta}  p(\theta \mid T_{\theta,\phi_i}) \cdot p(T_{\theta,\phi_i}) } \rangle_{T|\theta} $$

here, this comes back to part1.
the architecture is like an inverted (from traditional) variational auto encoder.
we have input as a uniform distriubtion of $\theta^*$ that we are averaging over.
we have latent being sampled trajectories $T_{\theta,\phi_i}$ (except we dont model the trajectory distrubiton).
the target is $\theta^*$ again, but the network output is a probabilty distribution $p(\theta)$.
$p(\theta)$ will not exactly match the point value $\theta^*$, because sampled trajectories $T_{\theta,\phi_i}$ introduce noise, we lose information in this step, making $p(\theta)$ not a delta distribution when we average this over the full $\Theta$ range.

back to the part 2 option 2. 
we replace the $p(T_{\theta,\phi_i})$ term by using sampled trajectories.
and we get $p(\theta \mid T_{\theta,\phi_i})$ by passing these sample trajectories throught the network in part 1.
due to we only have limited samples of trajectories given $\theta$ and unable to trace the gradient, we choose to approximate the fisher information.
based on Cramer-Rao Lower Bound, the inverse of the variance (covariance) is the lower bound of fisher information.



option 3.
instead of the information gain directly, we try to minimize the entropy $H(\theta)$


lastly.
with whichever approach, we can also reweight the importance of $\theta_j$ based on the current estimation of $p(\theta)$ to ignore the information gain for very unlikely $\theta_j$.

efficiency concerns. this step requries solving an optimization problem that loops over the entire $\Theta, \Phi$ space.
however, the trajectory embeddings can be pre calculated.
maybe the fisher information can also be precalculated.
we can also use something like binary search, for example, first decide between $\phi \sim \Phi_+$ and $\phi \sim \Phi_-$, then split either $\Phi_+$ or $\Phi_-$ into half to find a finer $\phi$.
besides, the selected task will only be used for a small protion of the data to avoid subject adaptation, so it allows time for 'real time' best task selection.

alternatives:

discertize everything. F output to a softmax categorical theta. in finding phi step, with discrete we can calculate information directly. 
the major con is too many output dim when we have many parameters.
my feeling is when the model gets larger, we need a lot more data to train it. could be a problem.

part2, use pure math, need to calcuclate 2nd order dev of p(theta estimated | theta input), this is an autoencoder like thing, input is theta input, latent is T, output is theta estimation, a prob.
still useful, can be used with old irc to get best task.

so faster phi selection needs approximation or assumption. not pure math.

use entropy instead of gi. input theta, T, use network we have p(theata), calcualte entropy. argmin entropy(theta)
use gi, likelihood cannot calcualte, we use distance in embedding, to approxmiate prob distribution. slower, n2.



## task

we need these features to demostrate our part1 and part2.
a task should be:

- (able to inverse $\theta$) give some task configuration $\phi_i$, different agent assumtpion $\theta$ should produce different trajectory $T$.

- (able to differentiate good vs bad task $\phi$) some task configuration $\phi_i$ provide much better information of $\theta$ than $\phi_j$. for example, $p(T_{\phi_i, \theta}\mid \theta)$ is much wider (trajectories are different) than $p(T_{\phi_j, \theta}\mid \theta)$ (trajectories are very similar, hard to tell apart or even identical). 
in other words, we want: at least on some condition such as  $\phi_i$ we can infer $\theta$ (with low uncertainty), and with some other $\phi_j$ we cannot infer $\theta$ (or infer with high uncertainty).

the current cart pole vary length version is acceptable but not an ideal task.
it is acceptable because it satisfied 1, we can infer $\theta$ pretty well.
however, for most $\phi$ we can infer $\theta$ with relatively small uncertainty. that means we cannot differentiate different good/bad $\phi$.






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


# supp

## data format

### collected simulation data
we call task config param vector as phi, agent assumed param vector as theta.
there are x params, and each has r resolution (samples).
so, we have r^x for phi, and same for theta.
we have (r^x)^2 for all theta x phi combination.
in this case, we have (15^3)^2 combinations, each has about 400 ts, each ts has about 15 dim, in total this is maxmimumly 2GB data. should be good.

## complexity analysis 

    x, storage: reso*(2n) n is number of params
        this is the major limitation. large storage, complex increase in power order with nparameter increase.
        the previous pomdp likelihood appraoch, when nparam increase, problem complexity linearly.
    backward function training: when reso is large, have to use linear readout
    inference for theta. best way is softamx or some other way to model the probability.
    marginalize for phi: for all theta for all phi, find phi that a delta theta [000100] x abs(dervitative of tau \mid  phi) is max. 
        delta theta [000100], index one theta among all theta, to be weighted by current p(theta)
        abs(dervitative of tau \mid  phi). given a theta, now only theta affects tau. we can find the derivative of dtheta/dtau, meaning how much change in theta affact change in tau. here tau can be later layers of the network. in short, we find a phi that makes tau is most sensitive to theta change at a particular theta_estimation.
        computatio, in the worst case, we evaluate all theta, all phi, and calculate the gradient. but in practice, we can dynamically adjust the resolution of theta and phi. eg, given a theta, do something like a binsec, and got longer trials are better phi. from there, we can either random with the current knowledge, or continue binsec.
        to acc this, we can 1, process the data storage before hand, 2 keep some binsec point in menonry and other in storage, 



# other notes
```
[Notification]

token = xxxbarktokenxxx


[Datafolder]
data = /data, some mapped dir for the dev container
```


