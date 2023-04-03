

from stable_baselines3 import SAC, PPO
import matplotlib.ticker as mticker
from matplotlib import font_manager
from copy import copy
import matplotlib.pylab as pl
from collections import OrderedDict
from colorspacious import cspace_converter
from matplotlib import cm
import matplotlib as mpl
from contextlib import redirect_stdout, redirect_stderr, contextmanager, ExitStack
from astropy.convolution import convolve
import random
import warnings
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from torch.nn import parameter
from scipy.ndimage.measurements import label
from numpy.core.numeric import NaN
from matplotlib.collections import LineCollection
import multiprocessing
from pickle import HIGHEST_PROTOCOL
from inspect import EndOfBlock
from importlib import reload
import enum
import imp
from pathlib import Path
from tkinter import PhotoImage
from turtle import color
import numpy as np
from matplotlib.ticker import MaxNLocator
import scipy.stats
from numpy import pi
from torch import nn, threshold
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import axes_grid1
from matplotlib.patches import Ellipse, Circle
import heapq
from collections import namedtuple
import torch
import os
import scipy.io as spio
# os.chdir('C:\\Users\\24455\\iCloudDrive\\misc\\ffsb')
import numpy as np
from scipy.stats import tstd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import random_projection
import sys


# arg = Config()
warnings.filterwarnings('ignore')
seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(int(seed))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

######## comment out if not using ###########
font_dirs = [Path(os.getenv('workspace')) /
             'firefly_plots/fonts/computer-modern', ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
#############################################
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['font.size'] = '14'
plt.rcParams['axes.unicode_minus'] = False

cmaps = OrderedDict()
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
global color_settings
color_settings = {
    'v': 'tab:blue',
    'w': 'orange',
    'cov': '#8200FF',  # purple, save as belief, just use low alpha
    'b': '#8200FF',  # purple
    's': '#0000FF',  # blue
    'o': '#FF0000',  # red
    'a': '#008100',  # green, slightly dark
    'goal': '#4D4D4D',  # dark grey
    'hidden': '#636363',
    'model': '#FFBC00',  # orange shifting to green, earth
    '': '',
    'pair': [[1, 0, 0], [0, 0, 1]],

}


def hex2rgb(hexstr):
    if hexstr[0] == '#':
        hexstr = hexstr.lstrip('#')
    return [int(hexstr[i:i+2], 16)/255 for i in (0, 2, 4)]


def colorlinspace(color1, color2, n=10):
    # return n by color matrix
    if type(color1) == str:
        color1 = hex2rgb(color1)
    if type(color2) == str:
        color2 = hex2rgb(color2)
    color1 = np.array(color1)
    color2 = np.array(color2)
    return np.linspace(color1, color2, n)


def colorshift(mu, direction, amount, n):
    mu = np.array(mu)
    direction = np.array(direction)
    deltas = np.linspace(np.clip(mu-direction*amount, 0, 1),
                         np.clip(mu+direction*amount, 0, 1), n)
    res = [d.tolist() for d in deltas]
    # for d in deltas:
    #     res.append((mu+d).tolist())
    return res


def mean_confidence_interval(data, confidence=0.95):
    # ci of the mean
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos=[0, 0], nstd=2, color=None, ax=None, alpha=1, edgecolor=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        figure = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(-1.5, 1.5)

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, edgecolor='none',
                    height=height, angle=theta, **kwargs)
    if color is not None:
        ellip.set_color(color)
    if edgecolor:
        ellip.set_color('none')
        ellip.set_edgecolor(edgecolor)
    ellip.set_alpha(alpha)
    ax.add_artist(ellip)
    return ellip


def plot_circle(cov, pos, color=None, ax=None, alpha=1, edgecolor=None, **kwargs):
    'plot a circle'
    if ax is None:
        figure = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(-1.5, 1.5)
    assert cov[0, 0] == cov[1, 1]
    r = cov[0, 0]
    c = Circle(pos, r)
    if color is not None:
        c.set_color(color)
    if edgecolor:
        c.set_color('none')
        c.set_edgecolor(edgecolor)
    c.set_alpha(alpha)
    ax.add_artist(c)
    return c


def overlap_mc(r, cov, mu, nsamples=1000):
    'return the overlaping of a circle and ellipse using mc'
    # xrange=[-cov[0,0],cov[0,0]]
    # yrange=[-cov[1,1],cov[1,1]]
    # xrange=[mu[0]-r*1.1,mu[0]+r*1.1]
    # yrange=[mu[1]-r*1.1,mu[1]+r*1.1]

    check = []
    xs, ys = np.random.multivariate_normal(-mu, cov, nsamples).T
    # plot_overlap(r,cov,mu,title=None)
    for i in range(nsamples):
        # plt.plot(xs[i],ys[i],'.')
        if (xs[i])**2+(ys[i])**2 <= r**2:
            check.append(1)
        else:
            check.append(0)
    P = np.mean(check)
    return P


def plot_overlap(r, cov, mu, title=None):
    'plot the overlap between a circle and ellipse with cov'
    f1 = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-1.5, 1.5)
    if title is not None:
        ax.title.set_text(str(title))
    plot_cov_ellipse(cov, [0, 0], nstd=1, ax=ax)
    plot_circle(np.eye(2)*r, mu, ax=ax, color='r')
    return f1


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def inverseCholesky(vecL):
    """
    Performs the inverse operation to lower cholesky decomposition
    and converts vectorized lower cholesky to matrix P
    P = L L.t()
    """
    size = int(np.sqrt(2 * len(vecL)))
    L = np.zeros((size, size))
    mask = np.tril(np.ones((size, size)))
    L[mask == 1] = vecL
    P = L@(L.transpose())
    return P


def policy_surface(belief, torch_model):
    # manipulate distance, reltive angle in belief and plot
    r_range = [0.2, 1.0]
    r_ticks = 0.01
    r_labels = [r_range[0]]
    a_range = [-pi/4, pi/4]
    a_ticks = 0.05
    a_labels = [a_range[0]]
    while r_labels[-1]+r_ticks <= r_range[-1]:
        r_labels.append(r_labels[-1]+r_ticks)
    while a_labels[-1]+a_ticks <= a_range[-1]:
        a_labels.append(a_labels[-1]+a_ticks)
    policy_data_v = np.zeros((len(r_labels), len(a_labels)))
    policy_data_w = np.zeros((len(r_labels), len(a_labels)))
    for ri in range(len(r_labels)):
        for ai in range(len(a_labels)):
            belief[0] = r_labels[ri]
            belief[1] = a_labels[ai]
            policy_data_v[ri, ai] = baselines_mlp_model.predict(
                belief.view(1, -1).tolist())[0][0].item()
            policy_data_w[ri, ai] = baselines_mlp_model.predict(
                belief.view(1, -1).tolist())[0][1].item()

    plt.figure(0, figsize=(8, 8))
    plt.suptitle('Policy surface of velocity', fontsize=24)
    plt.ylabel('relative distance', fontsize=15)
    plt.xlabel('relative angle', fontsize=15)
    plt.imshow(policy_data_v, origin='lower', extent=[
               a_labels[0], a_labels[-1], r_labels[0], r_labels[-1]])
    plt.savefig('policy surface v {}.png'.format(name_index))

    plt.figure(1, figsize=(8, 8))
    plt.suptitle('Policy surface of angular velocity')
    plt.figure(1, figsize=(8, 8))
    plt.suptitle('Policy surface of velocity', fontsize=24)
    plt.ylabel('relative distance', fontsize=15)
    plt.xlabel('relative angle', fontsize=15)
    plt.imshow(policy_data_w, origin='lower', extent=[
               r_labels[0], r_labels[-1], a_labels[0], a_labels[-1]])

    plt.savefig('policy surface w {}.png'.format(name_index))


def policy_range(r_range=None, r_ticks=None, a_range=None, a_ticks=None):

    r_range = [0.2, 1.0] if r_range is None else r_range
    r_ticks = 0.01 if r_ticks is None else r_ticks
    r_labels = [r_range[0]]

    a_range = [-pi/4, pi/4] if a_range is None else a_range
    a_ticks = 0.05 if a_ticks is None else a_ticks
    a_labels = [a_range[0]]

    while r_labels[-1]+r_ticks <= r_range[-1]:
        r_labels.append(r_labels[-1]+r_ticks)
    while a_labels[-1]+a_ticks <= a_range[-1]:
        a_labels.append(a_labels[-1]+a_ticks)
    policy_data_v = np.zeros((len(r_labels), len(a_labels)))
    policy_data_w = np.zeros((len(r_labels), len(a_labels)))

    return r_labels, a_labels, policy_data_v, policy_data_w


def plot_path_ddpg(modelname, env, num_episode=None):

    from stable_baselines import DDPG

    num_episode = 20 if num_episode is None else num_episode

    agent = DDPG.load(modelname, env=env)

    # create saving vars
    all_ep = []
    # for ecah episode,
    for i in range(num_episode):
        ep_data = {}
        ep_statex = []
        ep_statey = []
        ep_belifx = []
        ep_belify = []
        # get goal position at start
        decisioninfo = env.reset()
        goalx = env.goalx
        goaly = env.goaly
        ep_data['goalx'] = goalx
        ep_data['goaly'] = goaly
        # log the actions raw, v and w
        while not env.stop:
            action, _ = agent.predict(decisioninfo)
            decisioninfo, _, _, _ = env.step(action)
            ep_statex.append(env.s[0, 0])
            ep_statey.append(env.s[0, 1])
            ep_belifx.append(env.b[0, 0])
            ep_belify.append(env.b[0, 1])
        ep_data['x'] = ep_statex
        ep_data['y'] = ep_statey
        ep_data['bx'] = ep_belifx
        ep_data['by'] = ep_belify
        ep_data['goalx'] = env.goalx
        ep_data['goaly'] = env.goaly
        ep_data['theta'] = env.theta.tolist()
        # save episode data dict to all data
        all_ep.append(ep_data)

    for i in range(num_episode):
        plt.figure
        ep_xt = all_ep[i]['x']
        ep_yt = all_ep[i]['y']
        plt.title(str(['{:.2f}'.format(x) for x in all_ep[i]['theta']]))
        plt.plot(ep_xt, ep_yt, 'r-')
        plt.plot(all_ep[i]['bx'], all_ep[i]['by'], 'b-')
        # plt.scatter(all_ep[i]['goalx'],all_ep[i]['goaly'])

        circle = np.linspace(0, 2*np.pi, 100)
        r = all_ep[i]['theta'][-1]
        x = r*np.cos(circle)+all_ep[i]['goalx'].item()
        y = r*np.sin(circle)+all_ep[i]['goaly'].item()
        plt.plot(x, y)

        plt.savefig('path.png')


def sort_evals_descending(evals, evectors):
    """
    Sorts eigenvalues and eigenvectors in decreasing order. Also aligns first two
    eigenvectors to be in first two quadrants (if 2D).

    Args:
      evals (numpy array of floats)    : Vector of eigenvalues
      evectors (numpy array of floats) : Corresponding matrix of eigenvectors
                                          each column corresponds to a different
                                          eigenvalue

    Returns:
      (numpy array of floats)          : Vector of eigenvalues after sorting
      (numpy array of floats)          : Matrix of eigenvectors after sorting
    """

    index = np.flip(np.argsort(evals))
    evals = evals[index]
    evectors = evectors[:, index]
    if evals.shape[0] == 2:
        if np.arccos(np.matmul(evectors[:, 0],
                               1 / np.sqrt(2) * np.array([1, 1]))) > np.pi / 2:
            evectors[:, 0] = -evectors[:, 0]
        if np.arccos(np.matmul(evectors[:, 1],
                               1 / np.sqrt(2) * np.array([-1, 1]))) > np.pi / 2:
            evectors[:, 1] = -evectors[:, 1]
    return evals, evectors


def get_sample_cov_matrix(X):
    """
      Returns the sample covariance matrix of data X

      Args:
        X (numpy array of floats) : Data matrix each column corresponds to a
                                    different random variable

      Returns:
        (numpy array of floats)   : Covariance matrix
    """

    # Subtract the mean of X
    X = X - np.mean(X, 0)
    # Calculate the covariance matrix (hint: use np.matmul)
    cov_matrix = cov_matrix = 1 / X.shape[0] * np.matmul(X.T, X)

    return cov_matrix


def plot_data_new_basis(Y):
    """
    Plots bivariate data after transformation to new bases. Similar to plot_data
    but with colors corresponding to projections onto basis 1 (red) and
    basis 2 (blue).
    The title indicates the sample correlation calculated from the data.

    Note that samples are re-sorted in ascending order for the first random
    variable.

    Args:
      Y (numpy array of floats) : Data matrix in new basis each column
                                  corresponds to a different random variable

    Returns:
      Nothing.
    """

    fig = plt.figure(figsize=[8, 4])
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(Y[:, 0], 'r')
    plt.ylabel('Projection \n basis vector 1')
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(Y[:, 1], 'b')
    plt.xlabel('Projection \n basis vector 1')
    plt.ylabel('Projection \n basis vector 2')
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.plot(Y[:, 0], Y[:, 1], '.', color=[.5, .5, .5])
    ax3.axis('equal')
    plt.xlabel('Projection basis vector 1')
    plt.ylabel('Projection basis vector 2')
    plt.title('Sample corr: {:.1f}'.format(
        np.corrcoef(Y[:, 0], Y[:, 1])[0, 1]))
    plt.show()


def pca(X):
    """
      Performs PCA on multivariate data.

      Args:
        X (numpy array of floats) : Data matrix each column corresponds to a
                                    different random variable

      Returns:
        (numpy array of floats)   : Data projected onto the new basis
        (numpy array of floats)   : Vector of eigenvalues
        (numpy array of floats)   : Corresponding matrix of eigenvectors

    """

    # Subtract the mean of X
    X = X - np.mean(X, axis=0)
    # Calculate the sample covariance matrix
    cov_matrix = get_sample_cov_matrix(X)
    # Calculate the eigenvalues and eigenvectors
    evals, evectors = np.linalg.eigh(cov_matrix)
    # Sort the eigenvalues in descending order
    evals, evectors = sort_evals_descending(evals, evectors)
    # Project the data onto the new eigenvector basis
    score = np.matmul(X, evectors)

    return score, evectors, evals


def column_feature_data(list_data):
    '''
    input: list of data, index as feature
    output: np array of data, column as feature, row as entry
    '''
    total_rows = len(list_data)
    total_cols = len(list_data[0])
    data_matrix = np.zeros((total_rows, total_cols))
    for i, data in enumerate(list_data):
        data_matrix[i, :] = [d[0] for d in data]
    return data_matrix


def plot_inverse_trajectory(theta_trajectory, true_theta, env, agent,
                            phi=None, method='PCA', background_data=None,
                            background_contour=False, number_pixels=10,
                            background_look='contour',
                            ax=None, loss_sample_size=100, H=False, loss_function=None):
    '''    
    plot the inverse trajectory in 2d pc space
    -----------------------------
    input:
    theta trajectory: list of list(theta)
    method: PCA or other projections
    -----------------------------
    output:
    figure
    '''
    # plot trajectory
    if ax is None:
        fig = plt.figure(figsize=[8, 8])
        ax = fig.add_subplot()

    # reshape the data into colume as features
    data_matrix = column_feature_data(theta_trajectory)
    if method == 'PCA':
        try:
            score, evectors, evals = pca(data_matrix)
        except np.linalg.LinAlgError:
            score, evectors, evals = pca(data_matrix)
        # note, the evectors are arranged in cols. one col, one evector.
        plt.xlabel('Projection basis vector 1')
        plt.ylabel('Projection basis vector 2')
        plt.title('Inverse for theta estimation in a PCs space. Sample corr: {:.1f}'.format(
            np.corrcoef(score[:, 0], score[:, 1])[0, 1]))

        # plot true theta
        mu = np.mean(data_matrix, 0)
        if type(true_theta) == list:
            true_theta = np.array(true_theta).reshape(-1)
            true_theta_pc = (true_theta-mu)@evectors
        elif type(true_theta) == torch.nn.parameter.Parameter:
            true_theta_pc = (
                true_theta.detach().numpy().reshape(-1)-mu)@evectors
        else:
            true_theta_pc = (true_theta-mu)@evectors

        ax.scatter(true_theta_pc[0], true_theta_pc[1],
                   marker='o', c='', edgecolors='r', zorder=10)
        # plot latest theta
        ax.scatter(score[-1, 0], score[-1, 1], marker='o',
                   c='', edgecolors='b', zorder=9)

        # plot theta inverse trajectory
        row_cursor = 0
        while row_cursor < score.shape[0]-1:
            row_cursor += 1
            # plot point
            ax.plot(score[row_cursor, 0], score[row_cursor, 1],
                    '.', color=[.5, .5, .5])
            # plot arrow
            # fig = plt.figure(figsize=[8, 8])
            # ax1 = fig.add_subplot()
            # ax1.set_xlim([-1,1])
            # ax1.set_ylim([-1,1])
            ax.quiver(score[row_cursor-1, 0], score[row_cursor-1, 1],
                      score[row_cursor, 0]-score[row_cursor-1,
                                                 0], score[row_cursor, 1]-score[row_cursor-1, 1],
                      angles='xy', color='g', scale=1, scale_units='xy')
        ax.axis('equal')

        # plot hessian
        if H:
            H = compute_H(env, agent, theta_trajectory[-1], true_theta.reshape(-1, 1), phi,
                          trajectory_data=None, H_dim=len(true_theta), num_episodes=loss_sample_size)
            cov = theta_cov(H)
            cov_pc = evectors[:, :2].transpose()@np.array(cov)@evectors[:, :2]
            plot_cov_ellipse(
                cov_pc, pos=score[-1, :2], alpha_factor=0.5, ax=ax)
            stderr = np.sqrt(np.diag(cov)).tolist()
            ax.title.set_text('stderr: {}'.format(
                str(['{:.2f}'.format(x) for x in stderr])))

        # plot log likelihood contour
        loss_function = compute_loss_wrapped(env, agent, true_theta.reshape(-1, 1), np.array(
            phi).reshape(-1, 1), trajectory_data=None, num_episodes=1000) if loss_function is None else loss_function
        current_xrange = list(ax.get_xlim())
        current_xrange[0] -= 0.5
        current_xrange[1] += 0.5
        current_yrange = list(ax.get_ylim())
        current_yrange[0] -= 0.5
        current_yrange[1] += 0.5
        xyrange = [current_xrange, current_yrange]
        # ax1.contourf(X,Y,background_data)
        if background_contour:
            background_data = plot_background(ax, xyrange, mu, evectors, loss_function, number_pixels=number_pixels,
                                              look=background_look) if background_data is None else background_data

    return background_data


def inverse_trajectory_monkey(theta_trajectory,
                              env=None,
                              agent=None,
                              phi=None,
                              background_data=None,
                              background_contour=False,
                              number_pixels=10,
                              background_look='contour',
                              ax=None,
                              loss_sample_size=100,
                              H=None,
                              loss_function=None,
                              **kwargs):
    '''    
        plot the inverse trajectory in 2d pc space
        -----------------------------
        input:
        theta trajectory: list of list(theta)
        method: PCA or other projections
        -----------------------------
        output:
        background contour array and figure
    '''
    # plot trajectory
    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot()
    data_matrix = column_feature_data(theta_trajectory)

    try:
        score, evectors, evals = pca(data_matrix)
    except np.linalg.LinAlgError:
        score, evectors, evals = pca(data_matrix)
        # note, the evectors are arranged in cols. one col, one evector.
        plt.xlabel('Projection basis vector 1')
        plt.ylabel('Projection basis vector 2')
        plt.title('Inverse for theta estimation in a PCs space. Sample corr: {:.1f}'.format(
            np.corrcoef(score[:, 0], score[:, 1])[0, 1]))

    # plot theta inverse trajectory
    row_cursor = 0
    while row_cursor < score.shape[0]-1:
        row_cursor += 1
        # plot point
        # ax.plot(score[row_cursor, 0], score[row_cursor, 1], '.', color=[.5, .5, .5])
        # plot arrow
        # fig = plt.figure(figsize=[8, 8])
        # ax1 = fig.add_subplot()
        # ax1.set_xlim([-1,1])
        # ax1.set_ylim([-1,1])
        ax.plot(score[row_cursor-1:row_cursor+1, 0],
                score[row_cursor-1:row_cursor+1, 1],
                '-',
                linewidth=0.1,
                color='g')
        if row_cursor % 20 == 0 or row_cursor == 1:
            ax.quiver(score[row_cursor-1, 0], score[row_cursor-1, 1],
                      score[row_cursor, 0]-score[row_cursor-1,
                                                 0], score[row_cursor, 1]-score[row_cursor-1, 1],
                      angles='xy', color='g', scale=0.2, width=1e-2, scale_units='xy')
    ax.scatter(score[row_cursor, 0], score[row_cursor, 1],
               marker=(5, 1), s=200, color=[1, .5, .5])
    # ax.axis('equal')

    # plot hessian
    if H is not None:
        cov = theta_cov(H)
        cov_pc = evectors[:, :2].transpose()@np.array(cov)@evectors[:, :2]
        plot_cov_ellipse(cov_pc, pos=score[-1, :2], alpha_factor=0.5, ax=ax)
        stderr = np.sqrt(np.diag(cov)).tolist()
        ax.title.set_text('stderr: {}'.format(
            str(['{:.2f}'.format(x) for x in stderr])))

    # plot log likelihood contour
    loss_function = compute_loss_wrapped(env, agent, true_theta.reshape(-1, 1), np.array(
        phi).reshape(-1, 1), trajectory_data=None, num_episodes=1000) if loss_function is None else loss_function
    current_xrange = list(ax.get_xlim())
    current_xrange[0] -= 0.5
    current_xrange[1] += 0.5
    current_yrange = list(ax.get_ylim())
    current_yrange[0] -= 0.5
    current_yrange[1] += 0.5
    xyrange = [current_xrange, current_yrange]
    # ax1.contourf(X,Y,background_data)
    if background_contour:
        background_data = plot_background(ax, xyrange, mu, evectors, loss_function, number_pixels=number_pixels,
                                          look=background_look) if background_data is None else background_data

    return background_data


def load_inverse_data(filename):
    'load the data pickle file, return the data dict'
    sys.path.insert(0, './inverse_data/')
    if filename[-4:] == '.pkl':
        data = torch.load('inverse_data/{}'.format(filename))
    else:
        data = torch.load('inverse_data/{}.pkl'.format(filename))
    return data


def run_inverse(data=None, theta=None, filename=None):
    import os
    import warnings
    warnings.filterwarnings('ignore')
    from copy import copy
    import time
    import random
    seed = time.time().as_integer_ratio()[0]
    seed = 0
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(int(seed))
    from numpy import pi
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # -----------invser functions-------------
    from InverseFuncs import trajectory, getLoss, reset_theta, theta_range, reset_theta_log, single_inverse
    # ---------loading env and agent----------
    from stable_baselines import DDPG, TD3
    from firefly_task import ffenv_new_cord
    from Config import Config
    arg = Config()
    DISCOUNT_FACTOR = 0.99
    arg.NUM_SAMPLES = 2
    arg.NUM_EP = 1000
    arg.NUM_IT = 2  # number of iteration for gradient descent
    arg.NUM_thetas = 1
    arg.ADAM_LR = 0.007
    arg.LR_STEP = 2
    arg.LR_STOP = 50
    arg.lr_gamma = 0.95
    arg.PI_STD = 1
    arg.goal_radius_range = [0.05, 0.2]

    # agent convert to torch model
    import policy_torch
    baselines_mlp_model = TD3.load(
        'trained_agent//TD_95gamma_mc_smallgoal_500000_9_24_1_6.zip')
    agent = policy_torch.copy_mlp_weights(
        baselines_mlp_model, layers=[128, 128])

    # loading enviorment, same as training
    env = ffenv_new_cord.FireflyAgentCenter(arg)
    env.agent_knows_phi = False

    true_theta_log = []
    true_loss_log = []
    true_loss_act_log = []
    true_loss_obs_log = []
    final_theta_log = []
    stderr_log = []
    result_log = []
    number_update = 100
    if data is None:
        save_dict = {'theta_estimations': []}
    else:
        save_dict = data

    # use serval theta to inverse
    for num_thetas in range(arg.NUM_thetas):

        # make sure phi and true theta stay the same
        true_theta = torch.Tensor(data['true_theta'])
        env.presist_phi = True
        # here we first testing teacher truetheta=phi case
        env.reset(phi=true_theta, theta=true_theta)
        theta = torch.Tensor(data['theta_estimations'][0])
        phi = torch.Tensor(data['phi'])

        save_dict['true_theta'] = true_theta.data.clone().tolist()
        save_dict['phi'] = true_theta.data.clone().tolist()
        save_dict['inital_theta'] = theta.data.clone().tolist()

        for num_update in range(number_update):
            states, actions, tasks = trajectory(
                agent, phi, true_theta, env, arg.NUM_EP)

            result = single_theta_inverse(
                true_theta, phi, arg, env, agent, states, actions, tasks, filename, num_thetas, initial_theta=theta)

            save_dict['theta_estimations'].append(result.tolist())
            if filename is None:
                savename = ('inverse_data/' + filename + "EP" + str(arg.NUM_EP) + "updates" + str(
                    number_update)+"sample"+str(arg.NUM_SAMPLES) + "IT" + str(arg.NUM_IT) + '.pkl')
                torch.save(save_dict, savename)
            elif filename[:-4] == '.pkl':
                torch.save(save_dict, filename)
            else:
                torch.save(save_dict, (filename+'.pkf'))

            print(result)

    print('done')


def continue_inverse(filename):
    data = load_inverse_data(filename)
    theta = data['theta_estimations'][0]
    run_inverse(data=data, filename=filename, theta=theta)


def _jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(
            flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def _hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)


def stderr(cov):
    return np.sqrt(np.diag(cov)).tolist()


def mytick(x, nticks=5, roundto=0):
    # x, input. nticks, number of ticks. round, >0 for decimal, <0 for //10 power
    if roundto != 0:
        scale = 10**(-roundto)
        return np.linspace(np.floor(np.min(x)/scale)*scale, np.ceil(np.max(x)/scale)*scale, nticks)
    else:
        return np.linspace(np.floor(np.min(x)), np.ceil(np.max(x)), nticks)


def input_formatter(**kwargs):
    result = {}
    for key, value in kwargs.items():
        # print("{0} = {1}".format(key, value))
        result[key] = value
    return result


@contextmanager
def suppress(out=True, err=False):
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield


def cart2pol(*args):
    if type(args[0]) == list:
        x = args[0][0]
        y = args[0][1]
    else:
        x = args[0]
        y = args[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def xy2pol(*args, rotation=True):
    # return distance and angle. default rotated left 90 degree for the task
    x = args[0][0]
    y = args[0][1]
    d = np.sqrt(x**2 + y**2)
    a = np.arctan2(y, x)+pi/2 if rotation else np.arctan2(y, x)
    return d, a


def similar_trials(ind, tasks, actions=None, ntrial=10):
    indls = []  # max heap of dist and ind
    for i in range(len(tasks)):
        dx = abs(tasks[i][0]-tasks[ind][0])
        dy = abs(tasks[i][1]-tasks[ind][1])
        if actions is not None:
            dv = abs(actions[i][0][0]-actions[ind][0][0])
            dw = abs(actions[i][0][1]-actions[ind][0][1])
            d = -1*(dx**2+dy**2+0.5*dv**2+0.5*dw**2).item()
        else:
            d = -1*(dx**2+dy**2).item()
        if len(indls) >= ntrial:
            heapq.heappushpop(indls, (d, i))  # push and pop
        else:
            heapq.heappush(indls, (d, i))  # only push
    result = ([i[1] for i in heapq.nlargest(10, indls)])
    return result


def similar_trials2this(tasks, thistask, thisaction=None, actions=None, ntrial=10):
    indls = []  # max heap of dist and ind
    for i in range(len(tasks)):
        dx = abs(tasks[i][0]-thistask[0])
        dy = abs(tasks[i][1]-thistask[1])
        if actions is not None:
            dv = abs(actions[i][0][0]-thisaction[0])
            dw = abs(actions[i][0][1]-thisaction[1])
            d = -1*(dx**2+dy**2+0.5*dv**2+0.5*dw**2).item()
        else:
            d = -1*(dx**2+dy**2).item()
        if len(indls) >= ntrial:
            heapq.heappushpop(indls, (d, i))  # push and pop
        else:
            heapq.heappush(indls, (d, i))  # only push
    result = ([i[1] for i in heapq.nlargest(ntrial, indls)])
    return result


def similar_trials2thispert(tasks, thistask, thispertmeta, ntrial=10, pertmeta=None):
    indls = []  # max heap of dist and ind
    for i in range(len(tasks)):
        # target position
        dx = abs(tasks[i][0]-thistask[0])
        dy = abs(tasks[i][1]-thistask[1])
        # pert strength
        dps = abs(pertmeta[i][1]-thispertmeta[1])*0.1
        # pert direction
        dpd = abs(pertmeta[i][0]-thispertmeta[0])/pi/10
        # pert timing
        dpt = abs(pertmeta[i][2]-thispertmeta[2])*0.005
        d = -1*(dx**2+dy**2).item()-dpt-dpd-dps
        if len(indls) >= ntrial:
            heapq.heappushpop(indls, (d, i))  # push and pop
        else:
            heapq.heappush(indls, (d, i))  # only push
    result = ([i[1] for i in heapq.nlargest(ntrial, indls)])

    return result


def normalizematrix(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


def get_relative_r_ang(px, py, heading_angle, target_x, target_y):
    heading_angle = np.deg2rad(heading_angle)
    distance_vector = np.vstack([px - target_x, py - target_y])
    relative_r = np.linalg.norm(distance_vector, axis=0)

    relative_ang = heading_angle - np.arctan2(distance_vector[1],
                                              distance_vector[0])
    # make the relative angle range [-pi, pi]
    relative_ang = np.remainder(relative_ang, 2 * np.pi)
    relative_ang[relative_ang >= np.pi] -= 2 * np.pi
    return relative_r, relative_ang


def quickspine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def quickallspine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


@contextmanager
def initiate_plot(dimx=24, dimy=9, dpi=100, fontweight='normal'):
    plt.rcParams['figure.figsize'] = (dimx, dimy)
    plt.rcParams['font.weight'] = fontweight
    plt.rcParams['mathtext.default'] = 'regular'
    # plt.rcParams["font.family"] = 'Arial'
    plt.rcParams['font.family'] = 'CMU Serif'
    plt.rcParams['font.size'] = '14'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    fig = plt.figure(dpi=dpi)
    yield fig
    plt.show()


def quicksave(name, fig=None):
    if not fig:
        plt.savefig('/data/figures/{}.svg'.format(name),
                    dpi='figure', format='svg', bbox_inches="tight")
    else:
        fig.savefig('/data/figures/{}.svg'.format(name),
                    dpi='figure', format='svg', bbox_inches="tight")


def set_violin_plot(bp, facecolor, edgecolor, linewidth=1, alpha=1, ls='-', hatch=r''):
    plt.setp(bp['bodies'], facecolor=facecolor, edgecolor=edgecolor,
             linewidth=linewidth, alpha=alpha, ls=ls, hatch=hatch)
    plt.setp(bp['cmins'], facecolor=facecolor, edgecolor=edgecolor,
             linewidth=linewidth, alpha=alpha, ls=ls)
    plt.setp(bp['cmaxes'], facecolor=facecolor, edgecolor=edgecolor,
             linewidth=linewidth, alpha=alpha, ls=ls)
    plt.setp(bp['cmedians'], facecolor='k', edgecolor='k',
             linewidth=linewidth, alpha=alpha, ls=ls)
    plt.setp(bp['cbars'], facecolor='None', edgecolor='None',
             linewidth=linewidth, alpha=alpha, ls=ls)


def set_box_plot(bp, color, linewidth=1, alpha=0.9, ls='-', unfilled=False):
    if unfilled is True:
        plt.setp(bp['boxes'], facecolor='None', edgecolor=color,
                 linewidth=linewidth, alpha=1, ls=ls)
    else:
        plt.setp(bp['boxes'], facecolor=color, edgecolor=color,
                 linewidth=linewidth, alpha=alpha, ls=ls)
    plt.setp(bp['whiskers'], color='k',
             linewidth=linewidth, alpha=alpha, ls=ls)
    plt.setp(bp['caps'], color='k', linewidth=linewidth, alpha=alpha, ls=ls)
    plt.setp(bp['medians'], color='k', linewidth=linewidth, alpha=alpha, ls=ls)


def filter_fliers(data, whis=1.5, return_idx=False):
    filtered_data = []
    fliers_ides = []
    for value in data:
        Q1, Q2, Q3 = np.percentile(value, [25, 50, 75])
        lb = Q1 - whis * (Q3 - Q1)
        ub = Q3 + whis * (Q3 - Q1)
        filtered_data.append(value[(value > lb) & (value < ub)])
        fliers_ides.append(np.where((value > lb) & (value < ub))[0])
    if return_idx:
        return filtered_data, fliers_ides
    else:
        return filtered_data


def my_ceil(a, precision=0):
    return np.round(a + 0.5 * 10**(-precision), precision)


def my_floor(a, precision=0):
    return np.round(a - 0.5 * 10**(-precision), precision)


def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def match_targets(df, reference):
    df.reset_index(drop=True, inplace=True)
    reference.reset_index(drop=True, inplace=True)
    df_targets = df.loc[:, ['target_x', 'target_y']].copy()
    reference_targets = reference.loc[:, ['target_x', 'target_y']].copy()

    closest_df_indices = []
    for _, reference_target in reference_targets.iterrows():
        distance = np.linalg.norm(df_targets - reference_target, axis=1)
        closest_df_target = df_targets.iloc[distance.argmin()]
        closest_df_indices.append(closest_df_target.name)
        df_targets.drop(closest_df_target.name, inplace=True)

    matched_df = df.loc[closest_df_indices]
    matched_df.reset_index(drop=True, inplace=True)

    return matched_df


def config_colors():
    colors = {'LSTM_c': 'olive', 'EKF_c': 'darkorange', 'monkV_c': 'indianred', 'monkB_c': 'blue',
              'sensory_c': '#29AbE2', 'belief_c': '#C1272D', 'motor_c': '#FF00FF',
              'reward_c': 'C0', 'unreward_c': 'salmon',
              'gain_colors': ['k', 'C2', 'C3', 'C5', 'C9']}
    return colors


def pol2xy(a, d):
    x = d*np.cos(a)
    y = d*np.sin(a)
    return [x, y]


def colorgrad_line(x, y, z, ax=None, linewidth=2, cmap='viridis'):
    x, y, z = np.array(x), np.array(y), np.array(z)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = np.array(z)
    if ax:
        lc = LineCollection(segments, cmap=cmap)
        lc.set_array(colors)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)
        plt.colorbar(line, ax=ax)
    else:
        fig, ax = plt.subplots()
        lc = LineCollection(segments, cmap='viridis')
        lc.set_array(colors)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax)
    return ax

    with initiate_plot(4, 2, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111) if not ax else ax
        for given_state in statelike:
            ax.plot(given_state[:, 0], given_state[:, 1],
                    color=color, linewidth=1)
        if goalcircle:
            for eachtask in taskslike:
                ax.add_patch(plt.Circle(
                    (eachtask[0], eachtask[1]), 0.13, color=color_settings['goal'], alpha=0.5, edgecolor='none'))
        else:
            ax.scatter(taskslike[:, 0], taskslike[:, 1],
                       s=2, color=color_settings['goal'])
        ax.axis('equal')
        ax.set_xlabel('world x')
        ax.set_ylabel('world y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # legend and label
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc=2, prop={'size': 6})
        return ax


def percentile_err2d(data, axis=0, percentile=95):
    def fun(x): return percentile_err1d(x, percentile=percentile)
    return np.apply_along_axis(fun, axis, data)


def percentile_err1d(data, percentile=95):
    # return asysmetry error for data of categority x samples
    low = 100-percentile
    high = percentile
    lower_ci = np.percentile(data, low)
    upper_ci = np.percentile(data, high)
    asymmetric_error = np.array([lower_ci, upper_ci]).T
    res = np.array([np.abs(np.mean(data)-asymmetric_error[0]),
                   np.abs(asymmetric_error[1]-np.mean(data))])
    return res


def percentile_err(data, percentile=95):
    # return asysmetry error for data of categority x samples
    low = 100-percentile
    high = percentile
    lower_ci = [np.percentile(i, low) for i in data]
    upper_ci = [np.percentile(i, high) for i in data]
    # without adjusting for the mean
    asymmetric_error = np.array(list(zip(lower_ci, upper_ci))).T
    mean = np.array([np.mean(i) for i in data])
    res = np.array([np.abs(mean.T-asymmetric_error[0, :]),
                   np.abs(asymmetric_error[1, :]-mean.T)])
    return res


def thetaconfhist(finalcov):
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar([i for i in range(finalcov.shape[0])], torch.diag(
            finalcov)**0.5*1/torch.tensor(theta_mean), color='tab:blue')
        # title and axis names
        ax.set_ylabel('inferred parameter uncertainty (std/mean)')
        ax.set_xticks([i for i in range(finalcov.shape[0])])
        ax.set_xticklabels(theta_names, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


sf = mticker.ScalarFormatter(useOffset=False, useMathText=True)
def g(x, pos): return "${}$".format(sf._formatSciNotation('%1e' % x))


fmt = mticker.FuncFormatter(g)


def d_process(logs, densities):
    means = [log[-1][0]._mean if log else None for log in logs]
    stds = [np.diag(log[-1][0]._C)**0.5 if log else None for log in logs]
    xs, ys, errs = [], [], []
    for i in range(len(logs)):
        if means[i] is not None and stds[i] is not None and densities[i] is not None:
            x, y, err = densities[i], means[i], stds[i]
            xs.append(x)
            ys.append(y)
            errs.append(err)
    xs, ys, errs = torch.tensor(xs).float(), torch.tensor(
        ys).float(), torch.tensor(errs).float()
    return xs, ys, errs


def d_noise(xs, ys, errs):
    lables = ['pro v', 'pro w', 'obs v', 'obs w']
    with initiate_plot(2.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.set_xticks([i for i in range(len(xs))])
        ax.set_xticklabels([fmt(i.item()) for i in xs])
        ax.set_ylabel('noise std', fontsize=12)
        ax.set_xlabel('density', fontsize=12)
        ax.set_title('inferred noise level vs ground density', fontsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for i in range(4):
            ax.errorbar([i for i in range(len(xs))], ys[:, i],
                        errs[:, i], label=lables[i], alpha=0.7)
        ax.legend(fontsize=8)


def d_kalman_gain(xs, ys, errs, conf=20, alpha=0.9):
    with initiate_plot(3, 3, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)

        # angular w
        pn_sample = np.random.normal(
            ys[:, 3], errs[:, 3], size=(100, ys.shape[0])).clip(1e-3,)
        on_sample = np.random.normal(
            ys[:, 5], errs[:, 5], size=(100, ys.shape[0])).clip(1e-3,)
        k_sample = pn_sample**2/(pn_sample**2+on_sample**2)
        mu = np.array([np.mean(k_sample[:, i])
                      for i in range(k_sample.shape[1])])
        lb = [np.percentile(k_sample[:, i], conf)
              for i in range(k_sample.shape[1])]
        hb = [np.percentile(k_sample[:, i], 100-conf)
              for i in range(k_sample.shape[1])]
        cibound = np.array([lb, hb])
        err = np.array([np.abs(mu.T-cibound[0, :]),
                       np.abs(cibound[1, :]-mu.T)])
        ax.errorbar(np.arange(len(xs))-0.1, y=mu, yerr=err,
                    label='angular w', alpha=alpha)

        # forward v
        pn_sample = np.random.normal(
            ys[:, 2], errs[:, 2], size=(100, ys.shape[0])).clip(1e-3,)
        on_sample = np.random.normal(
            ys[:, 4], errs[:, 4], size=(100, ys.shape[0])).clip(1e-3,)
        k_sample = pn_sample**2/(pn_sample**2+on_sample**2)
        mu = np.array([np.mean(k_sample[:, i])
                      for i in range(k_sample.shape[1])])
        lb = [np.percentile(k_sample[:, i], conf)
              for i in range(k_sample.shape[1])]
        hb = [np.percentile(k_sample[:, i], 100-conf)
              for i in range(k_sample.shape[1])]
        cibound = np.array([lb, hb])
        err = np.array([np.abs(mu.T-cibound[0, :]),
                       np.abs(cibound[1, :]-mu.T)])
        ax.errorbar(np.arange(len(xs))+0.1, y=mu, yerr=err,
                    label='forward v', alpha=alpha)

        # ax.hlines(y=0.5, xmin=0, xmax=len(xs)-1,linestyle='--', linewidth=2, color='black',alpha=0.8)
        # ax.plot(ys[:,2]/ys[:,4],label='forward v')
        # ax.plot(ys[:,2]**2/(ys[:,2]**2+ys[:,4]**2),label='forward v')
        # ax.plot(ys[:,3]**2/(ys[:,3]**2+ys[:,5]**2),label='angular w')
        # ax.plot(ys[:,3]/ys[:,5],label='angular w')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.set_yscale('log')
        ax.set_yticks([0., 0.5, 1.0])
        ax.set_xticks(list(range(len(xs))))
        ax.set_xticklabels([fmt(i.item()) for i in xs])
        ax.legend(fontsize=12)
        ax.set_ylabel('Kalman gain', fontsize=12)
        ax.set_xlabel('density', fontsize=12)
        # ax.set_title('observation reliable degree',fontsize=16)


def barpertacc(accs, trialtype, ax=None, label=None, shift=0, width=0.4):
    with initiate_plot(3, 3, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if ax is None:
            ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar([i+shift for i in range(len(accs))], accs, width, label=label)
        # title and axis names
        ax.set_ylabel('trial reward rate')
        ax.set_xticks([i for i in range(len(accs))])
        ax.set_xticklabels(trialtype, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
    return ax


def barpertacc(accs, trialtype, ax=None, label=None, shift=0, width=0.4):
    with initiate_plot(3, 3, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if ax is None:
            ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar([i+shift for i in range(len(accs))], accs, width, label=label)
        # title and axis names
        ax.set_ylabel('radial error')
        ax.set_xticks([i for i in range(len(accs))])
        ax.set_xticklabels(trialtype, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
    return ax


def mybar(data, xlabels, ax=None, label=None, shift=0, width=0.4):
    with initiate_plot(3, 3, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if ax is None:
            ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar([i+shift for i in range(len(data))], data, width, label=label)
        # title and axis names
        ax.set_ylabel('radial error')
        ax.set_xticks([i for i in range(len(data))])
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
    return ax


def gauss_div(m1, s1, m2, s2):
    newm = 1/(1/s1-1/s2)*(m1/s1-m2/s2)
    news = 1/(1/s1-1/s2)
    return newm, news


def dfoverhead_single(df, ind=None, alpha=0.5):
    ind = np.random.randint(0, len(df)) if not ind else ind
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.plot(df.iloc[ind].pos_x, df.iloc[ind].pos_y, label='path')
        goal = plt.Circle([df.iloc[ind].target_x, df.iloc[ind].target_y], 65,
                          facecolor=color_settings['goal'], edgecolor='none', alpha=alpha, label='target')
        ax.plot(0., 0., "*", color='black', label='start')
        ax.add_patch(goal)
        ax.axis('equal')
        ax.set_xlabel('world x [cm]')
        ax.set_ylabel('world y [cm]')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # legend and label
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc=2, prop={'size': 6})


def dfctrl_single(df, ind=None, alpha=0.5):
    ind = np.random.randint(0, len(df)) if not ind else ind
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ts = np.arange(len(df.iloc[ind].action_v))/10
        ax.plot(ts, df.iloc[ind].action_v, label='forward v control')
        ax.plot(ts, df.iloc[ind].action_w, label='angular w control')
        # ax.plot(0.,0., "*", color='black',label='start')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('control magnitude')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(left=0.)
        ax.set_ylim(-1., 1)
        ax.set_yticks([-1, 0, 1])
        # print(ax.xaxis.get_ticklabels())
        # for n, label in enumerate(ax.xaxis.get_ticklabels()):
        #     if n % 4 != 0:
        #         label.set_visible(False)
        # ax.axhline(0., dashes=[1,2], color='black', alpha=0.3)
        ax.spines['bottom'].set_position(('data', 0.))
        # legend and label
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  loc='upper right', prop={'size': 6})


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plotmetatrend(df):
    data = list(df.category)
    vocab = {'skip': 0, 'normal': 1, "crazy": 2, 'lazy': 3, 'wrong_target': 4}
    data = [vocab[each] for each in data]
    x = [i for i in range(len(data))]
    # plt.plot(y)
    plt.plot(x, smooth(data, 30), lw=2, label='trial type')
    plt.xlabel('trial number')
    plt.ylabel('reward, reward rate and trial type')
    plt.legend()
    data = list(df.rewarded)
    data = [1 if each else 0 for each in data]
    plt.plot(x, smooth(data, 30), lw=2, label='reward')
    print(vocab)


def ovbystate():
    ind = torch.randint(low=100, high=300, size=(1,))
    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.plot(states[ind][:, 0], states[ind][:, 1], color='r', alpha=0.5)
        goalcircle = plt.Circle(
            [tasks[ind][0], tasks[ind][1]], 0.13, color='y', alpha=0.5)
        ax.add_patch(goalcircle)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, 0.6)


def get_ci(log, low=5, high=95, threshold=2, ind=-1):
    res = [l[2] for l in log[:ind//threshold]]
    mean = log[ind][0]._mean
    allsamples = []
    for r in res:
        for point in r:
            allsamples.append([point[1], point[0]])
    allsamples.sort(key=lambda x: x[0])
    aroundsolution = allsamples[:ind//threshold]
    aroundsolution.sort(key=lambda x: x[0])
    alltheta = np.vstack([x[1] for x in aroundsolution])

    lower_ci = [np.percentile(alltheta[:, i], low)
                for i in range(alltheta.shape[1])]
    upper_ci = [np.percentile(alltheta[:, i], high)
                for i in range(alltheta.shape[1])]
    asymmetric_error = np.array(list(zip(lower_ci, upper_ci))).T
    res = np.array([np.abs(mean.T-asymmetric_error[0, :]),
                   np.abs(asymmetric_error[1, :]-mean.T)])
    # res=asymmetric_error
    return res


def twodatabar(data1, data2, err1=None, err2=None, labels=None, shift=0.4, width=0.5, ylabel=None, xlabel=None, color=['b', 'r'], xname=''):
    xs = list(range(max(len(data1), len(data2))))
    label1 = labels[0] if labels else None
    label2 = labels[1] if labels else None
    with initiate_plot(6, 4, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar(xs, data1, width, yerr=err1, label=label1, color=color[0])
        ax.bar([i+shift for i in range(len(xs))], data2,
               width, yerr=err2, label=label2, color=color[1])
        # title and axis names
        ax.set_ylabel(ylabel)
        ax.set_xticks([i for i in range(max(len(data1), len(data2)))])
        if xlabel:
            ax.set_xticklabels(xlabel, rotation=45, ha='right')
        ax.set_xlabel(xname)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()


def conditional_cov(covyy, covxx, covxy):
    # return the conditional cov of y|x
    # here use covxy==covyx as no causal here
    return covyy-covxy@np.linalg.inv(covxx)@covxy.T


def conditional_cov_block(cov, paramindex):
    cov = np.asfarray(cov)
    covxy = np.array(list(cov[:, paramindex][:paramindex]) +
                     list(cov[:, paramindex][paramindex+1:]))
    covyy = np.diag(cov)[paramindex]
    temp = np.delete(cov, (paramindex), axis=0)
    covxx = np.delete(temp, (paramindex), axis=1)
    return covyy-covxy@np.linalg.inv(covxx)@covxy.T


def scatter_hist(x, y):
    def _scatter_hist(x, y, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y, alpha=0.3)

        # now determine nice limits by hand:
        binwidth = 0.5
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth)*1.1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=22)
        ax_histy.hist(y, bins=22, orientation='horizontal')

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    # use the previously defined function
    _scatter_hist(x, y, ax, ax_histx, ax_histy)
    plt.show()


def overheaddf_tar(df, alpha=1, **kwargs):
    fontsize = 9
    with initiate_plot(1.8, 1.8, 300) as fig:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.set_xlim([-235, 235])
        ax.set_ylim([-2, 430])
        x_temp = np.linspace(-235, 235)
        ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
        ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
        ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)
        fig.tight_layout(pad=0)
        ax.plot(np.linspace(0, 230 + 7),
                np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
        ax.scatter(df[df.rewarded].target_x, df[df.rewarded].target_y, c='k',
                   alpha=alpha, edgecolors='none', marker='.', s=9, lw=1, label='rewarded')
        ax.scatter(df[~df.rewarded].target_x, df[~df.rewarded].target_y, c=[
                   1, 0.5, 0.5], alpha=alpha, edgecolors='none', marker='.', s=9, lw=1, label='unrewarded')
        ax.legend(loc='upper right', bbox_to_anchor=(0, 0))
        quicksave('all tar overhead')


def overheaddf_path(df, indls, alpha=0.5, **kwargs):
    pathcolor = 'gray' if 'color' not in kwargs else kwargs['color']
    fontsize = 9
    with initiate_plot(1.8, 1.8, 300) as fig:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.set_xlim([-235, 235])
        ax.set_ylim([-2, 430])
        x_temp = np.linspace(-235, 235)
        ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
        ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
        ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)
        fig.tight_layout(pad=0)
        ax.plot(np.linspace(0, 230 + 7),
                np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')

        for trial_i in indls:
            ax.plot(df.iloc[trial_i].pos_x, df.iloc[trial_i].pos_y,
                    c=pathcolor, lw=0.1, ls='-', alpha=alpha)
        quicksave('all trial path')


def conditional_cov(covyy, covxx, covxy):
    # return the conditional cov of y|x
    # here use covxy==covyx as no causal here
    return covyy-covxy@np.linalg.inv(covxx)@covxy.T


def conditional_cov_block(cov, paramindex):
    cov = np.asfarray(cov)
    covxy = np.array(list(cov[:, paramindex][:paramindex]) +
                     list(cov[:, paramindex][paramindex+1:]))
    covyy = np.diag(cov)[paramindex]
    temp = np.delete(cov, (paramindex), axis=0)
    covxx = np.delete(temp, (paramindex), axis=1)
    return covyy-covxy@np.linalg.inv(covxx)@covxy.T


def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))  # std
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def getcbarnorm(min, mid, max):
    divnorm = matplotlib.colors.TwoSlopeNorm(vmin=min, vcenter=mid, vmax=max)
    return divnorm


def eval_curvature(agent, env, phi, theta, tasks, vctrl=0., wctrl=0., ntrials=10):
    actions = []
    with suppress():
        for task in tasks:
            for _ in range(ntrials):
                env.reset(phi=phi, theta=theta,
                          pro_traj=None, vctrl=0., wctrl=0.)
                epactions, _, _, _ = run_trial(
                    agent, env, given_action=None, given_state=None, action_noise=0.1)
                if len(epactions) > 5:
                    actions.append(torch.stack(epactions))
    cur = abs(sum([sum(a[:, 1]) for a in actions]))
    return cur


def npsummary(nparray):
    print("n samples ", len(nparray))
    print("mean ", np.mean(nparray))
    print("std ", np.std(nparray))
    print("med ", np.median(nparray))
    print("range ", np.min(nparray), np.max(nparray))


def quickleg(ax, loc='lower right', bbox_to_anchor=(-1, -1)):

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg = ax.legend(by_label.values(), by_label.keys(),
                    loc='lower right', bbox_to_anchor=bbox_to_anchor)
    for lh in leg.legendHandles:
        lh.set_alpha(1)


def f_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.xlabel('parameter coef')
    ax = plt.gca()
    quickspine(ax)


def relu(arr):
    arr[arr < 0] = 0
    return arr



def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])
                

