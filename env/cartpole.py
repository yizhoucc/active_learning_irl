

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

# added 0328, parameter range and sampling. make the task a family of tasks
from env.env_config import Config
default_config=Config()


class CartPoleEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, config=None):

        self.config=default_config if config is None else config
        self.timeout=self.config.timeout
        # self.training=training # whether use the assumed param

        self.tau = self.config.tau # seconds between state updates
        self.kinematics_integrator = "euler"
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high=np.inf
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-high, high=high, shape=(12,), dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self.t+=1

        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        
        if action == 2:
            force = self.force_mag 
        elif action==0:
            force=-self.force_mag
        else:
            force=0
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.t>=self.timeout
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = -1*abs(force)*self.cost

        return np.hstack([self.state, self._gravity, self._masscart, self._masspole, self._total_mass, self._length, self._polemass_length, self._force_mag, self._cost]).astype(np.float32), reward, done, {}

    def random_params(self):
        # resolution and sample method (log or not)
        # if fixed param region in config has value, use that instead of sampling
        reso=self.config.reso
        method=self.config.sample_method
        
        if method=='log':
            forwardfn=np.log
            backwardfn=np.exp
        else:
            forwardfn=lambda x: x
            backwardfn=lambda x: x

        gravitys= backwardfn(np.linspace(self.config.gravity_range[0], forwardfn(self.config.gravity_range[1]), reso))
        masscarts= backwardfn(np.linspace(self.config.masscart_range[0], forwardfn(self.config.masscart_range[1]), reso))
        masspoles= backwardfn(np.linspace(self.config.masspole_range[0], forwardfn(self.config.masspole_range[1]), reso))
        lengths= backwardfn(np.linspace(self.config.length_range[0], forwardfn(self.config.length_range[1]), reso))
        force_mags= backwardfn(np.linspace(self.config.force_mag_range[0], forwardfn(self.config.force_mag_range[1]), reso))
        costs= backwardfn(np.linspace(self.config.cost_range[0], forwardfn(self.config.cost_range[1]), reso))

        gravity= np.random.choice(gravitys) if self.config.gravity is None else self.config.gravity
        masscart= np.random.choice(masscarts) if self.config.masscart is None else self.config.masscart
        masspole= np.random.choice(masspoles) if self.config.masspole is None else self.config.masspole
        length= np.random.choice(lengths) if self.config.length is None else self.config.length
        force_mag= np.random.choice(force_mags) if self.config.force_mag is None else self.config.force_mag     
        cost= np.random.choice(costs) if self.config.cost is None else self.config.cost

        return gravity,masscart,masspole,length,force_mag,cost

    def reset(self,phi=None,theta=None):

        # if not provide task conf, apply that
        if phi is None:
            gravity,masscart,masspole,length,force_mag,cost=self.random_params()
            self.gravity = gravity
            self.masscart = masscart
            self.masspole = masspole
            self.total_mass = self.masspole + self.masscart
            self.length = length  # actually half the pole's length
            self.polemass_length = self.masspole * self.length
            self.force_mag = force_mag
            self.cost=cost
        else:
            self.gravity = phi[0]
            self.masscart = phi[1]
            self.masspole = phi[2]
            self.total_mass = self.masspole + self.masscart
            self.length = phi[3]
            self.polemass_length = self.masspole * self.length
            self.force_mag = phi[4]
            self.cost=phi[5]
        # if not provide assumption, assumption is the true config. eg, when training.
        if theta is None:
            self._gravity = gravity
            self._masscart = masscart
            self._masspole = masspole
            self._total_mass = self._masspole + self._masscart
            self._length = length  # actually half the pole's length
            self._polemass_length = self._masspole * self._length
            self._force_mag = force_mag
            self._cost=cost
        else:
            self._gravity = theta[0]
            self._masscart = theta[1]
            self._masspole = theta[2]
            self._total_mass = self._masspole + self._masscart
            self._length = theta[3]
            self._polemass_length = self._masspole * self._length
            self._force_mag = theta[4]
            self._cost=cost=theta[5]

        self.t=0
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        
        return np.hstack([self.state, self._gravity, self._masscart, self._masspole, self._total_mass, self._length, self._polemass_length, self._force_mag,self._cost]).astype(np.float32)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None










