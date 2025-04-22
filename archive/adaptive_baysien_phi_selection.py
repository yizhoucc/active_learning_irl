
from plot_ult import *

# the psi function. P(x) = γ + (1 - γ - λ) * exp[-(x/α)^β]
class Psychometric:
    def __init__(self, gamma, lamda, threshold, slope) -> None:
        self.gamma=gamma # false alarm
        self.lamda=lamda # lapse
        self.threshold=threshold # threshold
        self.slope=slope # slope
    def __call__(self, x):
        return 1- (self.gamma + (1 - self.gamma - self.lamda) * np.exp(-(x/self.threshold)**self.slope))
    def plot(self, therange=[0,10],n=100):
        xs=np.linspace(therange[0], therange[1], n)
        with initiate_plot(4,3,200) as f:
            ax=f.add_subplot(111)
            plt.xscale('log')
            plt.plot(xs, policy(xs))
            plt.xlabel('stimulus intensity')
            plt.ylabel('probability of detection')
            plt.ylim(-0.1,1.1)
            plt.xlim(therange[0], therange[1])
            plt.plot(xs, np.zeros_like(xs), '--', color='grey')
            plt.plot(xs, np.ones_like(xs), '--', color='grey')
            plt.plot([self.threshold,self.threshold],[0,1], '-', color='grey')
            plt.text(self.threshold-0.1,0,'threshold')
            quickspine(ax)

        plt.show()

param={
    'gamma':.1, 
    'lamda':.1, 
    'threshold':1, 
    'slope':3
}
policy=Psychometric(**param)
policy.plot(therange=[0,10])


class PsychometricF2C:
    def __init__(self, gamma, lamda, threshold, slope) -> None:
        self.gamma=gamma # false alarm
        self.lamda=lamda # lapse
        self.threshold=threshold # threshold
        self.slope=slope # slope
    def __call__(self, x,e=0):
        return 1-(1-self.gamma)*np.exp(-10**(self.slope/20*(x-self.threshold+e)))
    def plot(self, therange=[0,10],n=100):
        xs=np.linspace(therange[0], therange[1], n)
        with initiate_plot(4,3,200) as f:
            ax=f.add_subplot(111)
            plt.xscale('log')
            plt.plot(xs, policy(xs))
            plt.xlabel('stimulus intensity')
            plt.ylabel('probability of detection')
            plt.ylim(-0.1,1.1)
            plt.xlim(therange[0], therange[1])
            plt.plot(xs, np.zeros_like(xs), '--', color='grey')
            plt.plot(xs, np.ones_like(xs), '--', color='grey')
            plt.plot([self.threshold,self.threshold],[0,1], '-', color='grey')
            plt.text(self.threshold-0.1,0,'threshold')
            quickspine(ax)
        plt.show()

param={
    'gamma':0.5,  # success at 0 intensity
    'lamda':.1, 
    'threshold':1, 
    'slope':2.5
}
policy=PsychometricF2C(**param)
policy.plot(therange=[10**-4,10])




# placement rule
xs=np.linspace(0,1, 100)
def fun(x):
    return x*(1-x)
plt.plot(xs,fun(xs))

# for one trial (stimulus), calculate the prob of success
phi=0.5
p=policy(phi)
p # success
1-p # failure

# for n trials, calculate the likelihood p(data|threshold)
data=[0,0,1,0,1]
phi_ls=[0.1,0.3,0.5,0.7,0.9]
Q=0
for r,phi in zip(data,phi_ls):
    p=policy(phi)
    if r: # success
        Q+=np.log(p)
    else:
        Q+=np.log(1-p)
np.exp(Q)



import numpy as np
from scipy.stats import norm

def quest(probe_function, start_intensity, prior_std, beta, gamma, lapse_rate, num_trials):
    """
    Implements the QUEST procedure for a given probe function and starting intensity.

    Parameters:
    probe_function (function): The probe function that maps stimulus intensities to probabilities of correct responses.
    start_intensity (float): The starting intensity of the probe function.
    prior_std (float): The standard deviation of the prior distribution on the threshold parameter.
    beta (float): The slope of the psychometric function.
    gamma (float): The guessing rate.
    lapse_rate (float): The lapse rate.
    num_trials (int): The number of trials to run.

    Returns:
    thresholds (ndarray): An array of estimated threshold values at each trial.
    intensities (ndarray): An array of the stimulus intensities used at each trial.
    responses (ndarray): An array of the participant's responses at each trial.
    """
    # Initialize variables
    thresholds = np.zeros(num_trials)
    intensities = np.zeros(num_trials)
    responses = np.zeros(num_trials)

    # Set up the prior distribution on the threshold parameter
    prior_mean = start_intensity
    prior_var = prior_std**2
    posterior_mean = prior_mean
    posterior_var = prior_var

    # Run the QUEST procedure
    for i in range(num_trials):
        # Choose the intensity for the current trial based on the current posterior distribution
        intensity = norm(posterior_mean, np.sqrt(posterior_var)).rvs()

        # Compute the probability of a correct response at the current intensity
        p_correct = probe_function(intensity)

        # Sample a response from a binomial distribution based on the probability of a correct response
        response = np.random.binomial(1, p_correct)

        # Update the posterior distribution on the threshold parameter based on the response
        if response == 1:
            posterior_mean = (posterior_var*prior_mean + beta**2*intensity) / (posterior_var + beta**2)
            posterior_var = posterior_var*prior_var / (posterior_var + beta**2)
        else:
            posterior_mean = (posterior_var*prior_mean - beta**2*intensity) / (posterior_var + beta**2)
            posterior_var = posterior_var*prior_var / (posterior_var + beta**2)

        # Apply the guessing and lapse rates to the response
        if response == 0:
            response = gamma
        elif response == 1:
            response = 1 - lapse_rate
        else:
            response = lapse_rate

        # Save the threshold, intensity, and response for the current trial
        thresholds[i] = posterior_mean
        intensities[i] = intensity
        responses[i] = response
        print(thresholds,intensities,responses)

    return thresholds, intensities, responses

def generate_data(intensities, true_threshold, beta, gamma, lapse_rate):
    """
    Generates simulated data for testing the QUEST procedure.

    Parameters:
    intensities (ndarray): An array of stimulus intensities to use.
    true_threshold (float): The true threshold value.
    beta (float): The slope of the psychometric function.
    gamma (float): The guessing rate.
    lapse_rate (float): The lapse rate.

    Returns:
    simulated_responses (ndarray): An array of simulated participant responses at each intensity.
    """
    # Compute the probability of a correct response at each intensity
    p_correct = 1   - (gamma + lapse_rate    + (1 - gamma - lapse_rate)*np.exp(-(intensities/true_threshold)**beta))
    # Sample a response from a binomial distribution based on the probability of a correct response
    simulated_responses = np.random.binomial(1, p_correct)

    return simulated_responses

# Set up the parameters for the QUEST procedure
start_intensity = 0.5
prior_std = 0.1
beta = 3
gamma = 0.1
lapse_rate = 0.05
num_trials = 5

# Generate some random stimulus intensities
intensities = np.linspace(0.1, 1, num_trials)

# Generate some random simulated data
true_threshold = 0.4
simulated_responses = generate_data(intensities, true_threshold, beta, gamma, lapse_rate)

# Run the QUEST procedure on the simulated data
def probe_function(intensities):
    return  1   - (gamma + lapse_rate    + (1 - gamma - lapse_rate)*np.exp(-(intensities/true_threshold)**beta))
thresholds, intensities, responses = quest(probe_function, start_intensity, prior_std, beta, gamma, lapse_rate, num_trials)


xs=np.linspace(0.,1,50)
ys=xs*(1-xs)

# weibull
xs=np.linspace(-0,5,50)
threshold=3;slope=3.5
weibull=slope/threshold*(xs/threshold)**(slope-1)*np.exp(-((xs)/threshold)**slope)
plt.plot(xs,weibull)
# plt.plot(xs, np.cumsum(weibull)/10)

# derivative of weibull
t=threshold;k=slope;x=xs
# d=-(np.exp(-(x/t)**k) *k* (x/t)**k *(-1 + k*(-1 + (x/t)**k)))/t**2


# cdf, psychometric function
xs=np.linspace(-0,5,50)
threshold=2;slope=3.5;chancelevel=0.5
weibullcdf=1-(1-chancelevel)*np.exp(-((xs)/threshold)**slope)
plt.plot(xs,weibullcdf,label='psychometric function')


phi=1
t=np.linspace(-2,2,401)
weibullcdf=1-(1-chancelevel)*np.exp(-((phi)/t)**slope)
plt.plot(t,weibullcdf,label='psychometric function')

# derivative of psychometric function
d=-(np.exp(-(phi/t)**k)*k*(phi/t)**k)/t
plt.plot(t,d, label='d psychometric function/d theta')

# double derivative of psychometric function
dd=-(np.exp(-(phi/t)**k)*k*(phi/t)**k *(-1 + k *(-1 + (phi/t)**k)))/t**2
# dd=(t**k*np.exp(phi**k/t**k) - k*phi**k)/(t**(k+1)*np.exp(phi**k/t**k))
plt.plot(t,dd, label='d2 psychometric function/d2 theta')
plt.legend()


# log scale
phi= 1
t=np.linspace(-2,2,401)
weibullcdf=np.log(1-(1-chancelevel)*np.exp(-((phi)/t)**slope))
plt.plot(t,weibullcdf,label='psychometric function')

# derivative 
d=-(k* (phi/t)**k)/((-1 + np.exp((phi/t)**k))* t)
plt.plot(t,d, label='d psychometric function/d theta')

# double derivative 
dd=-(k *(phi/t)**k *(1 + k + np.exp((phi/t)**k) *(-1 + k* (-1 + (phi/t)**k))))/((-1 + np.exp((phi/t)**k))**2 *t**2)
plt.plot(t,dd, label='d2 psychometric function/d2 theta')
plt.plot([phi,phi],plt.ylim())
# plt.legend()



# cdf, psychometric function, from quest
delta = 0.01
gamma=0.5
beta=3.5
phi=0
t=np.linspace(-1,1,201)
logweibullcdf=np.log(delta*gamma+(1-delta)*(1-(1-gamma)*np.exp(-10**(beta*(t)))))
plt.plot(t,logweibullcdf,label='psychometric function')

d=10**(beta *t) *(-1 + delta) *beta* np.exp(-10**(beta* t)) *(-1 + gamma) *np.log(10)
logd=(10**(beta* t) *(-1 + beta) *beta* (-1 + gamma)* np.log(10))/(-1 + np.e**(10**(beta *t)) + delta *(-1 + np.e**(10*(beta* t)))* (-1 + gamma) + gamma)
plt.plot(t,logd, label='d psychometric function/d theta')

dd=-10**(beta* t) *(-1 + 10**(beta* t))* (-1 + delta)* beta**2 * np.exp(-10**(beta* t))* (-1 + gamma)* np.log(10)**2
logdd=((delta - 1) *beta**2* (gamma - 1)* np.log(10)**2* 10**(beta* t))/(delta* (gamma - 1) *(np.exp(10**(beta* t)) - 1) + np.exp(10**(beta* t)) + gamma - 1) - ((delta - 1) *beta *(gamma - 1)* np.log(10)* 10**(beta* t) *(delta* beta* (gamma - 1)* np.log(10)* 10**(beta* t)* np.exp(10**(beta* t)) + beta* np.log(10)* 10**(beta* t)* np.exp(10**(beta* t))))/(delta* (gamma - 1)* (np.exp(10**(beta* t)) - 1) +np.exp(10**(beta* t)) + gamma - 1)**2
plt.plot(t,logdd, label='dd psychometric function/d theta')

plt.plot([0,0],plt.ylim())

-1*sum(dd)

h=-t*np.log(1-t)
plt.plot(t,h)

plt.plot(t,np.log2(1/t))
plt.plot(t,np.log2(1/t)+np.log2(1/(1-t)))

plt.plot(t,1/(t*(1-t)))


t=np.linspace(-3,3,222)
weibull=1 - 0.5*np.exp(-10**(t-0.5)*3.5/20)
plt.plot(t,(1 - weibull)*weibull)
plt.plot(t, weibull*weibull)




# here the t is theta. input to psy function is phi-theta, stimulus-threshold
delta = 0.01
gamma=0.5
beta=3.5
phi=0
theta=0.0
t=np.linspace(-2.5,2.5,22)

# treating phi as free variable.
weibull=delta*gamma+(1-delta)*(1-(1-gamma)*np.exp(-10**(beta*(t-theta))))
plt.plot(t,weibull)
# success and failure function
plt.plot(t,((1 - weibull)*(weibull))) # failure rate * choice
plt.plot(t,(weibull*weibull)) # scucess rate * choice
# log version success and failure function
plt.plot(t,np.log((1 - weibull)*weibull))
plt.plot(t,np.log( weibull*weibull))
# log likelihood log L(α) = ∑[yi * log P(xi;α) + (1-yi) * log(1 - P(xi;α))]
logll=weibull*np.log(1 - weibull) +(1-weibull)*np.log(1 - weibull)
plt.plot(t,logll)
plt.plot(t[1:],np.diff(logll))
plt.plot(t[2:],np.diff(np.diff(logll)))
plt.plot([theta,theta],plt.ylim())

q=QuestObject(tGuess,tGuessSd,pThreshold,beta,delta,gamma)
plt.plot(q.x, q.pdf)
plt.plot(q.x,np.convolve(q.pdf,[0,1,0], mode='same'))
plt.plot(q.x,np.convolve(q.pdf,np.diff(np.diff(logll)), mode='same'))




# 0503
t=np.linspace(-1.3,1.2,99)
gamma=0.5
phi=0
theta=-0.0
weibull=delta*gamma+(1-delta)*(1-(1-gamma)*np.exp(-10**(beta*(t-theta))))
plt.plot(t,weibull)

plt.plot(t,weibull*(1-weibull))

plt.plot(t[1:],np.diff(weibull))
plt.plot(t[1:],np.diff(weibull)**2)
plt.plot(t[1:],(weibull*(1-gamma-weibull))[1:])
plt.plot(t[1:],(weibull*(1-gamma-weibull))[1:]/(np.diff(weibull)**2))
plt.plot([t[np.argmin((weibull*(1-weibull))[1:]/(np.diff(weibull)**2))]
,t[np.argmin((weibull*(1-weibull))[1:]/(np.diff(weibull)**2))]
],plt.ylim())

plt.plot(t,weibull)
# plt.plot(t[1:],np.diff(weibull)**2)
plt.plot([t[np.argmin((weibull*(1-gamma-weibull))[1:]/(np.diff(weibull)**2))]
,t[np.argmin((weibull*(1-gamma-weibull))[1:]/(np.diff(weibull)**2))]
],plt.ylim())



plt.plot(t[1:],np.diff(weibull)**2/weibull[1:])
t[np.argmax(np.diff(weibull)**2/weibull[1:])]

plt.plot(t[1:],1/weibull[1:])

plt.plot(t,(1/(weibull*(1-weibull))))




plt.plot(t[1:],(np.diff(weibull)**2)/(weibull*(1-weibull))[1:])
weibull[[np.argmax((np.diff(weibull)**2)/(weibull*(1-weibull))[1:])]]
kernel=(np.diff(weibull)**2)/(weibull*(1-weibull))[1:]
conv=np.convolve(q.pdf,kernel, mode='valid')
plt.plot(conv)
maxi=np.argmax(conv)+50
q.x[maxi]
len(q.pdf)
len(conv)


plt.plot(t[2:],abs(np.diff(np.diff(np.log(weibull)))))
