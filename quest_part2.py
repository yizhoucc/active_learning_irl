# try to use part 2 to do same thing as quest.

from plot_ult import *
import math
def getinf(x):
    return np.nonzero( np.isinf( np.atleast_1d(x) ) )


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
    
'''
notations:
    phi, different intensities, as stimuli
    theta, different thresholds
    agent(theta), the psychometric function depends on threshold theta, output to a probability distribution of actions (0,1) and we randomly take one.

procedure:
    start from a gaussian prior.
    for all phi
        calculate I ~= weighted sum (var(behavior under phi and theta) for all theta). weight is the theta prior.
    use the max I phi as next stimuli.
    
'''


class QuestObject():

    def __init__(self,tGuess,tGuessSd,pThreshold,beta,delta,gamma,grain=0.01,range=None):
        super(QuestObject, self).__init__()
        grain = float(grain) # make sure grain is a float
        if range is None:
            dim = 500
        else:
            if range <= 0:
                raise ValueError('argument "range" must be greater than zero.')
            dim=range/grain
            dim=2*math.ceil(dim/2.0) # round up to even integer
        self.updatePdf = True
        self.warnPdf = True
        self.normalizePdf = False
        self.tGuess = tGuess
        self.tGuessSd = tGuessSd
        self.pThreshold = pThreshold
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.grain = grain
        self.dim = dim
        self.recompute()

    def mean(self):
        """Mean of Quest posterior pdf.

        Get the mean threshold estimate.

        This was converted from the Psychtoolbox's QuestMean function.
        """
        return self.tGuess + np.sum(self.pdf*self.x)/np.sum(self.pdf)

    def mode(self):
        """Mode of Quest posterior pdf.

        t,p=q.mode()
        't' is the mode threshold estimate
        'p' is the value of the (unnormalized) pdf at t.

        This was converted from the Psychtoolbox's QuestMode function.
        """
        iMode = np.argsort(self.pdf)[-1]
        p=self.pdf[iMode]
        t=self.x[iMode]+self.tGuess
        return t

    def p(self,x):
        """probability of correct response at intensity x.

        p=q.p(x)

        The probability of a correct (or yes) response at intensity x,
        assuming threshold is at x=0.

        This was converted from the Psychtoolbox's QuestP function.
        """
        if x < self.x2[0]:
            return self.x2[0]
        if x > self.x2[-1]:
            return self.x2[-1]
        return np.interp(x,self.x2,self.p2)

    def pdf_at(self,t):
        """The (unnormalized) probability density of candidate threshold 't'.

        This was converted from the Psychtoolbox's QuestPdf function.
        """
        i=int(round((t-self.tGuess)/self.grain))+1+self.dim/2
        i=min(len(self.pdf),max(1,i))-1
        p=self.pdf[int(i)]
        return p

    def quantile(self,quantileOrder=None):
        """Get Quest recommendation for next trial level.

        intensity=q.quantile([quantileOrder])

        Gets a quantile of the pdf in the struct q.  You may specify
        the desired quantileOrder, e.g. 0.5 for median, or, making two
        calls, 0.05 and 0.95 for a 90confidence interval.  If the
        'quantileOrder' argument is not supplied, then it's taken from
        the QuestObject instance. __init__() uses recompute() to
        compute the optimal quantileOrder and saves that in the
        QuestObject instance; this quantileOrder yields a quantile
        that is the most informative intensity for the next trial.

        This was converted from the Psychtoolbox's QuestQuantile function.
        """
        if quantileOrder is None:
            quantileOrder = self.quantileOrder
        p = np.cumsum(self.pdf)
        if len(getinf(p[-1])[0]):
            raise RuntimeError('pdf is not finite')
        if p[-1]==0:
            raise RuntimeError('pdf is all zero')
        m1p = np.concatenate(([-1],p))
        index = np.nonzero( m1p[1:]-m1p[:-1] )[0]
        if len(index) < 2:
            raise RuntimeError('pdf has only %g nonzero point(s)'%len(index))
        ires = np.interp([quantileOrder*p[-1]],p[index],self.x[index])[0]
        return self.tGuess+ires

    def sd(self):
        """Standard deviation of Quest posterior pdf.

        Get the sd of the threshold distribution.

        This was converted from the Psychtoolbox's QuestSd function."""
        p=np.sum(self.pdf)
        sd=math.sqrt(np.sum(self.pdf*self.x**2)/p-(np.sum(self.pdf*self.x)/p)**2)
        return sd

    def simulate(self,tTest,tActual):
        """Simulate an observer with given Quest parameters.

        response=QuestSimulate(q,intensity,tActual)

        Simulate the response of an observer with threshold tActual.

        This was converted from the Psychtoolbox's QuestSimulate function."""
        t = min( max(tTest-tActual, self.x2[0]), self.x2[-1] )
        response= np.interp([t],self.x2,self.p2)[0] > random.random()
        return response

    def simulate_p(self,tTest,tActual):
        t = min( max(tTest-tActual, self.x2[0]), self.x2[-1] ) # clip
        return np.interp([t],self.x2,self.p2)[0] # take an inteprotated value
        
    def recompute(self):
        """Recompute the psychometric function & pdf.

        Call this immediately after changing a parameter of the
        psychometric function. recompute() uses the specified
        parameters in 'self' to recompute the psychometric
        function. It then uses the newly computed psychometric
        function and the history in self.intensity and self.response
        to recompute the pdf. (recompute() does nothing if q.updatePdf
        is False.)

        This was converted from the Psychtoolbox's QuestRecompute function."""
        if not self.updatePdf:
            return
        if self.gamma > self.pThreshold:
            warnings.warn( 'reducing gamma from %.2f to 0.5'%self.gamma)
            self.gamma = 0.5
        self.i = np.arange(-self.dim/2, self.dim/2+1)
        self.x = self.i * self.grain
        self.pdf = np.exp(-0.5*(self.x/self.tGuessSd)**2)
        self.pdf = self.pdf/np.sum(self.pdf)
        i2 = np.arange(-self.dim,self.dim+1)
        self.x2 = i2*self.grain
        self.p2 = self.delta*self.gamma+(1-self.delta)*(1-(1-self.gamma)*np.exp(-10**(self.beta*self.x2)))
        if self.p2[0] >= self.pThreshold or self.p2[-1] <= self.pThreshold:
            raise RuntimeError('psychometric function range [%.2f %.2f] omits %.2f threshold'%(self.p2[0],self.p2[-1],self.pThreshold)) # XXX
        if len(getinf(self.p2)[0]):
            raise RuntimeError('psychometric function p2 is not finite')
        index = np.nonzero( self.p2[1:]-self.p2[:-1] )[0] # strictly monotonic subset
        if len(index) < 2:
            raise RuntimeError('psychometric function has only %g strictly monotonic points'%len(index))
        self.xThreshold = np.interp([self.pThreshold],self.p2[index],self.x2[index])[0]
        self.p2 = self.delta*self.gamma+(1-self.delta)*(1-(1-self.gamma)*np.exp(-10**(self.beta*(self.x2+self.xThreshold))))
        if len(getinf(self.p2)[0]):
            raise RuntimeError('psychometric function p2 is not finite')
        self.s2 = np.array( ((1-self.p2)[::-1], self.p2[::-1]) ) # wrong, right
        if not hasattr(self,'intensity') or not hasattr(self,'response'):
            self.intensity = []
            self.response = []
        if len(getinf(self.s2)[0]):
            raise RuntimeError('psychometric function s2 is not finite')

        eps = 1e-14

        pL = self.p2[0]
        pH = self.p2[-1]
        pE = pH*math.log(pH+eps)-pL*math.log(pL+eps)+(1-pH+eps)*math.log(1-pH+eps)-(1-pL+eps)*math.log(1-pL+eps)
        pE = 1/(1+math.exp(pE/(pL-pH)))
        self.quantileOrder=(pE-pL)/(pH-pL)

        if len(getinf(self.pdf)[0]):
            raise RuntimeError('prior pdf is not finite')

        # recompute the pdf from the historical record of trials
        for intensity, response in zip(self.intensity, self.response):
            inten = max(-1e10,min(1e10, intensity)) # make intensity finite
            ii = len(self.pdf) + self.i-round((inten-self.tGuess)/self.grain)-1
            if ii[0]<0:
                ii = ii-ii[0]
            if ii[-1]>=self.s2.shape[1]:
                ii = ii+self.s2.shape[1]-ii[-1]-1
            iii = ii.astype(np.int_)
            if not np.allclose(ii,iii):
                raise ValueError('truncation error')
            self.pdf = self.pdf*self.s2[response,iii]
            if self.normalizePdf and ii % 100 == 0:
                self.pdf = self.pdf/np.sum(self.pdf) # avoid underflow; keep the pdf normalized
        if self.normalizePdf:
            self.pdf = self.pdf/np.sum(self.pdf) # avoid underflow; keep the pdf normalized
        if len(getinf(self.pdf)[0]):
            raise RuntimeError('prior pdf is not finite')

    def update(self,intensity,response):
        """Update Quest posterior pdf.

        Update self to reflect the results of this trial. The
        historical records self.intensity and self.response are always
        updated, but self.pdf is only updated if self.updatePdf is
        true. You can always call QuestRecompute to recreate q.pdf
        from scratch from the historical record.

        This was converted from the Psychtoolbox's QuestUpdate function."""

        if response < 0 or response > self.s2.shape[0]:
            raise RuntimeError('response %g out of range 0 to %d'%(response,self.s2.shape[0]))
        if self.updatePdf:
            inten = max(-1e10,min(1e10,intensity)) # make intensity finite
            ii = len(self.pdf) + self.i-round((inten-self.tGuess)/self.grain)-1
            if ii[0]<0 or ii[-1] > self.s2.shape[1]:
                if self.warnPdf:
                    low=(1-len(self.pdf)-self.i[0])*self.grain+self.tGuess
                    high=(self.s2.shape[1]-len(self.pdf)-self.i[-1])*self.grain+self.tGuess
                    warnings.warn( 'intensity %.2f out of range %.2f to %.2f. Pdf will be inexact.'%(intensity,low,high),
                                   RuntimeWarning,stacklevel=2)
                if ii[0]<0:
                    ii = ii-ii[0]
                else:
                    ii = ii+self.s2.shape[1]-ii[-1]-1
            iii = ii.astype(np.int_)
            if not np.allclose(ii,iii):
                raise ValueError('truncation error')
            self.pdf = self.pdf*self.s2[response,iii]
            # plt.plot(self.x,self.s2[response,iii],label='update function')
            if self.normalizePdf:
                self.pdf=self.pdf/np.sum(self.pdf)
        # keep a historical record of the trials
        self.intensity.append(intensity)
        self.response.append(response)

    def pdf_theta(self, theta):
        self.i = np.arange(-self.dim/2, self.dim/2+1)
        self.x = self.i * self.grain
        self.pdf = np.exp(-0.5*(self.x/self.tGuessSd)**2)
        self.pdf = self.pdf/np.sum(self.pdf)

tActual = 1.5 # ground truth
tGuess = 0.1 # chance level

tGuessSd = 2.0 # sd of Gaussian before clipping to specified range
pThreshold = 0.82
beta = 3.5
delta = 0.01
gamma = 0.5
q=QuestObject(tGuess,tGuessSd,pThreshold,beta,delta,gamma)

# Simulate a series of trials.
trialsDesired=5
wrongRight = 'wrong', 'right'

for k in range(trialsDesired):
    # Get recommended level.  Choose your favorite algorithm.
    # tTest=q.quantile()
    tTest=q.mean()
    # tTest=q.mode()
    plt.plot(q.x,q.pdf/max(q.pdf), color='k', alpha=0.5)
    plt.plot([tTest,tTest],[0,max(q.pdf/max(q.pdf))], color='red', label=r'chosen $phi$')

    # tTest=tTest+random.choice([-0.1,0,0.1])

    # Simulate a trial
    response=int(q.simulate(tTest,tActual))
    print('Trial %3d at %4.1f is %s'%(k+1,tTest,wrongRight[int(response)]))

    # Update the pdf
    q.update(tTest,response)
    plt.plot(q.x,q.pdf/max(q.pdf), color='k', alpha=1)
    # plt.title(response)
    plt.xlabel(r'stimulus intensity ($\phi$)')
    plt.ylabel(r'$P(\theta)$')
    quickspine(plt.gca())
    quickleg(plt.gca(), bbox_to_anchor=(0.5,1))
    



# scucess and failure function plot --------
with initiate_plot(4,3,200) as f:
    ax=f.add_subplot(111)
    plt.plot(q.x2,np.log(q.s2[0,:]),label='failure function')
    plt.plot(q.x2,np.log(q.s2[1,:]),label='success function')
    plt.plot([0,0],[-2,2],'--', color='grey')
    plt.xlim(-5,5)
    # plt.ylim(-0.1,1.1)
    quickleg(ax, bbox_to_anchor=[1,1])
    quickspine(ax)
    plt.xlabel(r'stimulus intensity, $\pm$ threshold [dB]')
    plt.ylabel('log probability')


# psychometric function plot -------
with initiate_plot(4,3,200) as f:
    ax=f.add_subplot(111)
    plt.plot(q.x,[q.simulate_p(x, tActual) for x in q.x],label='psychometric function')
    plt.xlabel(r'stimulus intensity ($\phi$)')
    plt.ylabel('probability')
    plt.ylim(-0.1,1.1)
    plt.plot(q.x, np.zeros_like(q.x), '--', color='grey')
    plt.plot(q.x, np.ones_like(q.x), '--', color='grey')
    plt.plot([tActual,tActual],[0,1], '-', color='grey')
    plt.text(tActual-0.1,0,r'threshold ($\theta$)')
    quickspine(ax)



q=QuestObject(tGuess,tGuessSd,pThreshold,beta,delta,gamma)

# tTest=q.mean()
tTest=q.mode()
plt.plot(q.x,q.pdf/max(q.pdf), color='k', alpha=0.5)
plt.plot([tTest,tTest],[0,max(q.pdf/max(q.pdf))], color='red', label=r'chosen $phi$')
response=int(q.simulate(tTest,tActual))
q.update(tTest,response)
plt.plot(q.x,q.pdf/max(q.pdf), color='k', alpha=1)
# plt.title(response)
plt.xlabel(r'stimulus intensity ($\phi$)')
plt.ylabel(r'$P(\theta)$')
quickspine(plt.gca())
quickleg(plt.gca(), bbox_to_anchor=(0.5,1))
plt.show()

def dd(theta, phi):
    # double derivative dpsi/d2theta
    t=theta
    dd=-(k *(phi/t)**k *(1 + k + np.exp((phi/t)**k) *(-1 + k* (-1 + (phi/t)**k))))/((-1 + np.exp((phi/t)**k))**2 *t**2)    
    return dd


q=q1
Is=[]
for phi in q.x:
    # I=sum([(0.75-q.simulate_p(phi,theta))**2*priorp for priorp,theta in zip(q.pdf/sum(q.pdf),q.x)])
    # I=np.nanmean([dd(theta,phi)*priorp for priorp,theta in zip(q.pdf/sum(q.pdf),q.x)])
    # I=np.sum([(theta,phi)*priorp for priorp,theta in zip(q.pdf/sum(q.pdf),q.x)])
    psy=delta*gamma+(1-delta)*(1-(1-gamma)*np.exp(-10**(beta*(q.x))))
    I0=(np.diff(psy)**2)/((psy*(1-psy))[1:]) # information of delta phi-theta, the kernel
    priortheta=q.pdf/sum(q.pdf)
    conv=np.convolve(priortheta,I0)[len(q.x)//2:-(len(q.x)//2)]
    I=np.sum(conv
             )
    # plt.plot(conv)
    # len(conv)
    # len(priortheta)
    # len(I0)
    # psy=delta*gamma+(1-delta)*(1-(1-gamma)*np.exp(-10**(beta*(q.x))))
    # plt.plot(psy)
    # plt.plot(I0)

    # phi=-1
    # # plt.plot(q.x, dd(q.x,phi))
    # plt.plot(q.x, q.pdf/sum(q.pdf))
    # plt.plot(q.x, dd(q.x,phi)*q.pdf/sum(q.pdf))
    # # plt.plot(q.x,[dd(theta,phi) for priorp,theta in zip(q.pdf/sum(q.pdf),q.x)])
    # plt.plot(q.x,[priorp for priorp,theta in zip(q.pdf/sum(q.pdf),q.x)])
    # plt.plot(q.x,[dd(theta,phi)*priorp for priorp,theta in zip(q.pdf/sum(q.pdf),q.x)])

    Is.append(I)
psy=delta*gamma+(1-delta)*(1-(1-gamma)*np.exp(-10**(beta*(q.x))))
I0=(np.diff(psy)**2)/((psy*(1-psy))[1:]) # information of delta phi-theta, the kernel
priortheta=q.pdf/sum(q.pdf)
conv=np.convolve(priortheta,I0)[len(q.x)//2:-(len(q.x)//2)]
bestphi=q.x[np.argmax(conv)]



# current theta estimation
plt.plot(q.x,q.pdf/max(q.pdf))
plt.xlabel(r'threshold $\theta$')
plt.ylabel(r'current $P(\theta|data)$')
plt.title(r'QUEST: mode $\theta$={:.2f}'.format(q.mode()))
plt.plot([q.mode(),q.mode()],[0,1], '--', color='grey')
quickspine(plt.gca())
plt.show()

# best phi--------
psy=delta*gamma+(1-delta)*(1-(1-gamma)*np.exp(-10**(beta*(q.x))))
I0=(np.diff(psy)**2)/((psy*(1-psy))[1:])
priortheta=q.pdf/sum(q.pdf)
conv=np.convolve(priortheta,I0)[len(q.x)//2:-(len(q.x)//2-1)]
bestphi=q.x[np.argmax(conv)]
plt.plot(q.x,conv)
plt.plot([bestphi,bestphi],[min(conv), max(conv)], '--', color='grey')
plt.xlabel(r'stimulus $\phi$')
plt.ylabel(r'optimization criterion')
plt.title(r'our: best $\phi$={:.2f}'.format(bestphi))
quickspine(plt.gca())





# compare
tActual = 1 # ground truth
tGuess = 0.0 # chance level
tGuessSd = 2.0 # sd of Gaussian before clipping to specified range
pThreshold = 0.82
beta = 3.5
delta = 0.01
gamma = 0.5

# compare how many steps need to converge
q1=QuestObject(tGuess,tGuessSd,pThreshold,beta,delta,gamma)
q2=QuestObject(tGuess,tGuessSd,pThreshold,beta,delta,gamma)
phis1=[]
phis2=[]
noise=0.05
for i in range(55):
    # q1 update
    tTest=q1.mean()
    # tTest=q1.quantile()
    tTest=tTest+(np.random.normal()*noise)
    response=int(q1.simulate(tTest,tActual))
    q1.update(tTest,response)
    phis1.append(tTest)

    # # q2 update
    # Is=[]
    # for i,phi in enumerate(q2.x):
    #     # I=sum([(0.796-q2.simulate_p(phi,theta))**2*priorp for priorp,theta in zip(q2.pdf/sum(q2.pdf),q2.x)])
    #     I=np.nansum([dd(theta,phi)*priorp for priorp,theta in zip(q.pdf/sum(q.pdf),q.x)])
    #     Is.append(I)
    # tTest=q2.x[np.argmin(Is)]
    # tTest=tTest+(np.random.normal()*noise)
    # response=int(q2.simulate(tTest,tActual))
    # q2.update(tTest,response)
    # phis2.append(tTest)

    # q2 update 0504
    psy0=delta*gamma+(1-delta)*(1-(1-gamma)*np.exp(-10**(beta*(q2.x))))
    I0=(np.diff(psy0)**2)/((psy0*(1-psy0))[1:])
    priortheta=q2.pdf/sum(q2.pdf)
    conv=np.convolve(priortheta,I0)[len(q2.x)//2:-(len(q2.x)//2-1)]
    # plt.plot(q.x,psy)
    # plt.plot(q.x[1:],I0)
    # plt.plot(q.x,conv)
    tTest=q2.x[np.argmax(conv)] # selected best phi
    tTest=tTest+(np.random.normal()*noise)
    response=int(q2.simulate(tTest,tActual))
    q2.update(tTest,response)
    phis2.append(tTest)
    
with initiate_plot(9,3,200) as f:
    ax1=f.add_subplot(131)
    ax2=f.add_subplot(132, sharey=ax1)
    ax1.set_ylabel('phi')
    ax1.set_xlabel('steps')
    ax2.set_xlabel('steps')
    ax1.set_title('quest')
    ax2.set_title('our')
    quickspine(ax1)
    quickspine(ax2)
    ax1.plot(list(range(len(phis1))), phis1)
    ax2.plot(list(range(len(phis2))), phis2, color='orange')

    ax3=f.add_subplot(133,sharey=ax1)
    quickspine(ax3)
    ax3.plot(list(range(len(phis2))), [tActual]*len(phis1), color='black')
    ax3.plot(list(range(len(phis1))), phis1)
    ax3.plot(list(range(len(phis2))), phis2)


# compare the best phi given same history
phis1=[]
phis2=[]
noise=0.05
for tActual in np.linspace(-2,2,100): # number of tasks
    q1=QuestObject(tGuess,tGuessSd,pThreshold,beta,delta,gamma)
    for i in range(55): # number of trials per task
        # quest selection
        tTest=q1.mean()
        # tTest=q1.quantile()
        # tTest=q1.mode()
        phis1.append(tTest)

        # our selection
        psy0=delta*gamma+(1-delta)*(1-(1-gamma)*np.exp(-10**(beta*(q1.x))))
        I0=(np.diff(psy0)**2)/((psy0*(1-psy0))[1:])
        priortheta=q1.pdf/sum(q1.pdf)
        conv=np.convolve(priortheta,I0)[len(q1.x)//2:-(len(q1.x)//2-1)]
        tTest2=q1.x[np.argmax(conv)] # selected best phi
        phis2.append(tTest2)
    
        tTest=tTest+(np.random.normal()*noise)
        response=int(q1.simulate(tTest,tActual))
        q1.update(tTest,response)
        
with initiate_plot(3,3,200) as f:
    ax1=f.add_subplot(111)
    ax1.set_ylabel('quest phi')
    ax1.set_xlabel('our phi')
    ax1.set_title('quest(mean) vs our phi')
    ax1.scatter(phis2,phis1, s=0.1)
    plt.axis('equal')
    plt.xlim([-2,2])
    ax1.plot([-2.5,2.5],[-2.5,2.5], color='black')



q1=QuestObject(tGuess,tGuessSd,pThreshold,beta,delta,gamma)
q2=QuestObject(tGuess,tGuessSd,pThreshold,beta,delta,gamma)
with initiate_plot(7,3,200) as f:
    ax1=f.add_subplot(1,2,1)
    ax2=f.add_subplot(1,2,2)

    ax1.plot(q1.x,q1.pdf/max(q1.pdf), color='k', alpha=0.5)

    # q1 update
    tTest=q1.quantile()
    tTest=tTest+(np.random.normal()*noise)
    response=int(q1.simulate(tTest,tActual))
    print('Trial %3d at %4.1f is %s'%(k+1,tTest,wrongRight[int(response)]))
    q1.update(tTest,response)

    ax1.plot([tTest,tTest],[0,max(q1.pdf/max(q1.pdf))], color='red', label=r'chosen $phi$')
    ax1.plot(q1.x,q1.pdf/max(q1.pdf), color='k', alpha=1)
    ax1.set_title('resp={},QUEST phi={:.2f}'.format(response,tTest))
    ax1.set_xlabel(r'stimulus intensity ($\phi$)')
    ax1.set_ylabel(r'$P(\theta)$')
    quickspine(ax1)


    ax2.plot(q2.x,q2.pdf/max(q2.pdf), color='k', alpha=0.5)
    # # q2 update
    # Is=[]
    # for phi in q2.x:
    #     I=sum([(0.75-q2.simulate_p(phi,theta))**2*priorp for priorp,theta in zip(q2.pdf/sum(q2.pdf),q2.x)])
    #     Is.append(I)
    # tTest=q2.x[np.argmin(Is)]
    # tTest=tTest+(np.random.normal()*noise)
    # response=int(q2.simulate(tTest,tActual))
    # q2.update(tTest,response)

    # q2 update 0504
    psy0=delta*gamma+(1-delta)*(1-(1-gamma)*np.exp(-10**(beta*(q2.x))))
    I0=(np.diff(psy0)**2)/((psy0*(1-psy0))[1:])
    priortheta=q2.pdf/sum(q2.pdf)
    conv=np.convolve(priortheta,I0)[len(q2.x)//2:-(len(q2.x)//2-1)]
    # plt.plot(q.x,psy)
    # plt.plot(q.x[1:],I0)
    # plt.plot(q.x,conv)
    tTest=q2.x[np.argmax(conv)] # selected best phi
    tTest=tTest+(np.random.normal()*noise)
    response=int(q2.simulate(tTest,tActual))
    q2.update(tTest,response)



    ax2.plot([tTest,tTest],[0,max(q2.pdf/max(q2.pdf))], color='red', label=r'chosen $phi$')
    ax2.plot(q2.x,q2.pdf/max(q2.pdf), color='k', alpha=1)
    ax2.set_title('resp={},our phi={:.2f}'.format(response,tTest))
    ax2.set_xlabel(r'stimulus intensity ($\phi$)')
    ax2.set_ylabel(r'$P(\theta)$')
    quickspine(ax2)
    quickleg(ax2, bbox_to_anchor=(1,1.2))


p=np.linspace(0,1,22)
plt.plot(p*(1-p))






plt.plot(q.x,[abs(0.75-q.simulate_p(0,theta))*priorp for priorp,theta in zip(q.pdf/sum(q.pdf),q.x)])

sum([(0.75-q.simulate_p(0,theta))**2*priorp for priorp,theta in zip(q.pdf/sum(q.pdf),q.x)])
sum([(0.75-q.simulate_p(1.5,theta))**2*priorp for priorp,theta in zip(q.pdf/sum(q.pdf),q.x)])
