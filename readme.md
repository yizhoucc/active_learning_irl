# todo

- potential new task if this is not working.

# current res
rrn baseline: good. cost doesnt matter but cost makes theta length matter

rnn prob: not good.


# data format

## collected simulation data
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


