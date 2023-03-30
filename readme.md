# todo

- randomize the parameters for agent family training
- set a timeout
- make it harder? train and see it first.

# data format

## collected simulation data
we call task config param vector as phi, agent assumed param vector as theta.
there are x params, and each has r resolution (samples).
so, we have r^x for phi, and same for theta.
we have (r^x)^2 for all theta x phi combination.
in this case, we have (15^3)^2 combinations, each has about 400 ts, each ts has about 15 dim, in total this is maxmimumly 2GB data. should be good.

# other notes
```
[Notification]

token = xxxbarktokenxxx


[Datafolder]
data = /data, some mapped dir for the dev container
```


