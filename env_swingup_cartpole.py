"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
import pygame
from pygame import gfxdraw

class SUCartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self, sutton_barto_reward: bool = False, render_mode: Optional[str] = None
    ):
        self._sutton_barto_reward = sutton_barto_reward

        self.gravity = 5
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.3  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "semmi-euler"

        self.t = 0
        self.t_limit = 4000
        self.epi = 0
        self.r = 0
        self.bananas = []

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

        self.banana_image = pygame.image.load('banana.png')
        self.background = pygame.image.load('antarctic_bg.jpg')
        self.tusk_color = (234, 221, 202)  # Ivory color
        self.tusk_width = 3
        self.tusk_length = 20


    def calculate_custom_reward(self, x, x_dot, theta, theta_dot):
        two_pi = 2 * np.pi
        
        # Reward for keeping the pole upright
        reward_theta = (np.e**(np.cos(theta) + 1.0) - 1.0)
        
        # Reward for keeping the cart near the center
        #reward_x = np.cos((x / self.x_threshold) * (np.pi / 2.0))
        reward_x = - (x / self.x_threshold) ** 2
        
        # Penalize high angular velocity to encourage smooth balancing
        reward_theta_dot = np.exp(-0.1 * (theta_dot**2))

        reward_x_dot =  ((np.cos(theta) * (np.e**(np.cos(x_dot)+1.0) - 1) / two_pi) + 1.0)
        
        # Combine the rewards with adjusted weights
        reward = (0.6 * reward_theta + 0.7 * reward_x + 0.3 * reward_theta_dot + 0.2 * reward_x_dot)

        if np.abs(theta) < 6/180 * np.pi:
            reward += 3

        if np.abs(theta) < 3/180 * np.pi:
            reward += 3

        if np.abs(x) >= self.x_threshold * 0.8:
            reward -= 2
        
        self.r = reward
        return reward

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
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
        
        # Ensure theta is within the range [-π, π]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        self.state = (x, x_dot, theta, theta_dot)

        done = False
        terminated = False

        if x < -self.x_threshold or x > self.x_threshold:
            terminated = True
            self.epi += 1

        self.t += 1
        done = self.t >= self.t_limit
    
        if not terminated:
            if self._sutton_barto_reward:
                reward = 0.0
            elif not self._sutton_barto_reward:
                reward = self.calculate_custom_reward(x, x_dot, theta, theta_dot)
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            if self._sutton_barto_reward:
                reward = -1.0
            else:
                reward = self.calculate_custom_reward(x, x_dot, theta, theta_dot)
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            if self._sutton_barto_reward:
                reward = -1.0
            else:
                reward = self.calculate_custom_reward(x, x_dot, theta, theta_dot)

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = np.random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.2, 0.2, 0.2, 0.2]))
        self.steps_beyond_terminated = None
        self.t = 0
        self.bananas = []

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run pip install "gymnasium[classic-control]"'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0
        earwidth = 40.0  # Width of each ear
        inearwidth = 25.0
        earheight = 60.0  # Height of each ear
        inearheight = 45.0
        reward = self.r

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))

        bg_scale_factor = 0.14
        bg_width = int(self.background.get_width() * bg_scale_factor)
        bg_height = int(self.background.get_height() * bg_scale_factor)
        bg_scaled_image = pygame.transform.scale(self.background, (bg_width, bg_height))
        bg_image_flipped = pygame.transform.flip(bg_scaled_image, False, True)
        self.surf.blit(bg_image_flipped, (0,0))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 130  # TOP OF CART

        # Draw the ears first (background layer)
        ear_color = (111, 78, 55) 
        ear_y = carty - (cartheight / 2 + earheight / 8)  # Position ears behind the cart
        pygame.draw.ellipse(self.surf, ear_color, pygame.Rect(cartx - cartwidth / 2 - earwidth / 2, ear_y, earwidth, earheight))
        pygame.draw.ellipse(self.surf, ear_color, pygame.Rect(cartx + cartwidth / 2 - earwidth / 2, ear_y, earwidth, earheight))

        # Draw the inner ears (background layer)
        inear_color = (196, 164, 132) 
        inear_y = carty - (cartheight / 2 + inearheight / 8)  # Position ears behind the cart
        pygame.draw.ellipse(self.surf, inear_color, pygame.Rect(cartx - cartwidth / 2 - inearwidth / 3, inear_y, inearwidth, inearheight))
        pygame.draw.ellipse(self.surf, inear_color, pygame.Rect(cartx + cartwidth / 2 - inearwidth / 1.5, inear_y, inearwidth, inearheight))

        # Draw the rounded corner cart
        cart_rect = pygame.Rect(cartx - cartwidth / 2, carty - cartheight / 2, cartwidth, cartheight)
        pygame.draw.rect(self.surf, (92, 64, 51), cart_rect, border_radius=10)

        # Draw the eyes first
        eye_angle = x[2]
        eye_offset_x = 15
        eye_offset_y = -10

        left_eye_x = cartx - eye_offset_x
        left_eye_y = carty - eye_offset_y
        right_eye_x = cartx + eye_offset_x
        right_eye_y = carty - eye_offset_y

        eye_radius = 6
        pupil_radius = 3

        # Draw the left eye
        gfxdraw.aacircle(self.surf, int(left_eye_x), int(left_eye_y), eye_radius, (255, 255, 255))
        gfxdraw.filled_circle(self.surf, int(left_eye_x), int(left_eye_y), eye_radius, (255, 255, 255))

        # Draw the right eye
        gfxdraw.aacircle(self.surf, int(right_eye_x), int(right_eye_y), eye_radius, (255, 255, 255))
        gfxdraw.filled_circle(self.surf, int(right_eye_x), int(right_eye_y), eye_radius, (255, 255, 255))

        pupil_offset_x = (eye_radius - pupil_radius) * np.sin(eye_angle)
        pupil_offset_y = (eye_radius - pupil_radius) * np.cos(eye_angle)

        # Draw the pupils
        gfxdraw.aacircle(self.surf, int(left_eye_x + pupil_offset_x), int(left_eye_y + pupil_offset_y), pupil_radius, (0, 0, 0))
        gfxdraw.filled_circle(self.surf, int(left_eye_x + pupil_offset_x), int(left_eye_y + pupil_offset_y), pupil_radius, (0, 0, 0))

        gfxdraw.aacircle(self.surf, int(right_eye_x + pupil_offset_x), int(right_eye_y + pupil_offset_y), pupil_radius, (0, 0, 0))
        gfxdraw.filled_circle(self.surf, int(right_eye_x + pupil_offset_x), int(right_eye_y + pupil_offset_y), pupil_radius, (0, 0, 0))

        # Draw the tusks
        tusk_offset_x = 15
        tusk_offset_y = 30

        # Left tusk
        left_tusk_points = [
            (cartx - tusk_offset_x, carty - tusk_offset_y),
            (cartx - tusk_offset_x - self.tusk_width, carty - tusk_offset_y + self.tusk_length),
            (cartx - tusk_offset_x + self.tusk_width, carty - tusk_offset_y + self.tusk_length)
        ]
        # Left tusk
        right_tusk_points = [
            (cartx + tusk_offset_x, carty - tusk_offset_y),
            (cartx + tusk_offset_x - self.tusk_width, carty - tusk_offset_y + self.tusk_length),
            (cartx + tusk_offset_x + self.tusk_width, carty - tusk_offset_y + self.tusk_length)
        ]

        pygame.draw.polygon(self.surf, self.tusk_color, left_tusk_points)
        pygame.draw.polygon(self.surf, self.tusk_color, right_tusk_points)

        # Draw the pole after the eyes
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (210, 125, 45))
        gfxdraw.filled_polygon(self.surf, pole_coords, (210, 125, 45))

        # Draw the axle
        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (152, 133, 88),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (152, 133, 88),
        )
        if self.banana_image is None:
            print("Banana image not loaded. Check the path.")
            return
        
        base_factor = 0.01
        if reward > 0:
            banana_size_factor = base_factor + ( reward / 200)  # Example: scale size based on reward
        else:
            banana_size_factor = 0
        banana_width = int(self.banana_image.get_width() * banana_size_factor)
        banana_height = int(self.banana_image.get_height() * banana_size_factor)
        banana_image_scaled = pygame.transform.scale(self.banana_image, (banana_width, banana_height))
        banana_image_flipped = pygame.transform.flip(banana_image_scaled, False, True)


        # Create a new banana every 10 frames
        if self.t % 80 == 0 and self.t > 50:
            banana_x = np.random.randint(cartx - 0.5 * cartwidth, cartx + 0.5 * cartwidth)
            banana_y = carty  # Start at the top of the screen
            banana_velocity_y = np.random.uniform(3, 10)  # Random downward velocity
            self.bananas.append([banana_x, banana_y, banana_velocity_y, banana_image_flipped])

        # Update and draw all bananas
        for banana in self.bananas:
            banana[1] += banana[2]  # Update banana y position based on velocity
            banana[2] -= self.gravity * 0.06  # Apply gravity to velocity
            self.surf.blit(banana[3], (banana[0], banana[1]))
        
        # Remove bananas that fall off the screen
        self.bananas = [banana for banana in self.bananas if banana[1] <= self.screen_height]

        # Render text for step count and episode number
        try:
            font = pygame.font.Font('freesansbold.ttf', 32)  # Load the font
        except FileNotFoundError:
            font = pygame.font.SysFont(None, 32)  # Fallback to default system font

        white = (255, 255, 255)
        skin = (210, 180, 140)
        text = font.render(f'{self.epi*20}', True, skin)
        text = pygame.transform.flip(text, False, True)
        textRect = text.get_rect()
        textRect.center = (self.screen_width // 6 * 5.0, self.screen_height // 6 * 4.5)
        self.surf.blit(text, textRect)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

class CartPoleVectorEnv(VectorEnv):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        sutton_barto_reward: bool = False,
    ):
        self._sutton_barto_reward = sutton_barto_reward

        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        self.state = None

        self.steps = np.zeros(num_envs, dtype=np.int32)
        self.prev_done = np.zeros(num_envs, dtype=np.bool_)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )

        self.low = -0.05
        self.high = 0.05

        self.single_action_space = spaces.Discrete(2)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.single_observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.screen_width = 600
        self.screen_height = 400
        self.screens = None
        self.surf = None

        self.steps_beyond_terminated = None

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta, theta_dot = self.state
        force = np.sign(action - 0.5) * self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
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

        self.state = np.stack((x, x_dot, theta, theta_dot))

        terminated: np.ndarray = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        self.steps += 1

        truncated = self.steps >= self.max_episode_steps

        if self._sutton_barto_reward is True:
            reward = -np.array(terminated, dtype=np.float32)
        else:
            reward = np.ones_like(terminated, dtype=np.float32)

        # Reset all environments which terminated or were truncated in the last step
        self.state[:, self.prev_done] = self.np_random.uniform(
            low=self.low, high=self.high, size=(4, self.prev_done.sum())
        )
        self.steps[self.prev_done] = 0
        reward[self.prev_done] = 0.0
        terminated[self.prev_done] = False
        truncated[self.prev_done] = False

        self.prev_done = terminated | truncated

        return self.state.T.astype(np.float32), reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        self.low, self.high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(
            low=self.low, high=self.high, size=(4, self.num_envs)
        )
        self.steps_beyond_terminated = None
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_done = np.zeros(self.num_envs, dtype=np.bool_)

        return self.state.T.astype(np.float32), {}

def render(self):
    if self.render_mode is None:
        assert self.spec is not None
        gym.logger.warn(
            "You are calling render method without specifying any render mode. "
            "You can specify the render_mode at initialization, "
            f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
        )
        return

    try:
        import pygame
        from pygame import gfxdraw
    except ImportError as e:
        raise DependencyNotInstalled(
            'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
        ) from e

    if self.screen is None:
        pygame.init()
        if self.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
        else:  # mode == "rgb_array"
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
    if self.clock is None:
        self.clock = pygame.time.Clock()

    world_width = self.x_threshold * 2
    scale = self.screen_width / world_width
    polewidth = 10.0
    polelen = scale * (2 * self.length)
    cartwidth = 50.0
    cartheight = 30.0
    earwidth = 40.0  # Width of each ear
    earheight = 60.0  # Height of each ear

    if self.state is None:
        return None

    x = self.state

    # Load the background image
    background_image_path = "/36641.jpg"  # Change this to your image path
    try:
        background_image = pygame.image.load(background_image_path)
        background_image = pygame.transform.scale(background_image, (self.screen_width, self.screen_height))
    except pygame.error as e:
        raise RuntimeError(f"Unable to load background image: {e}")

    # Create a surface for drawing
    self.surf = pygame.Surface((self.screen_width, self.screen_height))

    # Blit the background image onto the surface
    self.surf.blit(background_image, (0, 0))

    l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
    axleoffset = cartheight / 4.0
    cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
    carty = self.screen_height - 50  # TOP OF CART (above ground and grass)

    # Draw the ears first (background layer)
    ear_color = (105, 105, 105)  # Dark grey
    ear_y = carty - (cartheight / 2 + earheight / 8)  # Position ears behind the cart
    pygame.draw.ellipse(self.surf, ear_color, pygame.Rect(cartx - cartwidth / 2 - earwidth / 2, ear_y, earwidth, earheight))
    pygame.draw.ellipse(self.surf, ear_color, pygame.Rect(cartx + cartwidth / 2 - earwidth / 2, ear_y, earwidth, earheight))

    # Draw the rounded corner cart
    cart_rect = pygame.Rect(cartx - cartwidth / 2, carty - cartheight / 2, cartwidth, cartheight)
    pygame.draw.rect(self.surf, (255, 165, 0), cart_rect, border_radius=10)

    # Draw the eyes first
    eye_angle = x[2]
    eye_offset_x = 15
    eye_offset_y = -10

    left_eye_x = cartx - eye_offset_x
    left_eye_y = carty - eye_offset_y
    right_eye_x = cartx + eye_offset_x
    right_eye_y = carty - eye_offset_y

    eye_radius = 5
    pupil_radius = 2

    # Draw the left eye
    gfxdraw.aacircle(self.surf, int(left_eye_x), int(left_eye_y), eye_radius, (255, 255, 255))
    gfxdraw.filled_circle(self.surf, int(left_eye_x), int(left_eye_y), eye_radius, (255, 255, 255))

    # Draw the right eye
    gfxdraw.aacircle(self.surf, int(right_eye_x), int(right_eye_y), eye_radius, (255, 255, 255))
    gfxdraw.filled_circle(self.surf, int(right_eye_x), int(right_eye_y), eye_radius, (255, 255, 255))

    pupil_offset_x = (eye_radius - pupil_radius) * np.sin(eye_angle)
    pupil_offset_y = (eye_radius - pupil_radius) * np.cos(eye_angle)

    # Draw the pupils
    gfxdraw.aacircle(self.surf, int(left_eye_x + pupil_offset_x), int(left_eye_y + pupil_offset_y), pupil_radius, (0, 0, 0))
    gfxdraw.filled_circle(self.surf, int(left_eye_x + pupil_offset_x), int(left_eye_y + pupil_offset_y), pupil_radius, (0, 0, 0))

    gfxdraw.aacircle(self.surf, int(right_eye_x + pupil_offset_x), int(right_eye_y + pupil_offset_y), pupil_radius, (0, 0, 0))
    gfxdraw.filled_circle(self.surf, int(right_eye_x + pupil_offset_x), int(right_eye_y + pupil_offset_y), pupil_radius, (0, 0, 0))

    # Draw the pole after the eyes
    l, r, t, b = (
        -polewidth / 2,
        polewidth / 2,
        polelen - polewidth / 2,
        -polewidth / 2,
    )

    pole_coords = []
    for coord in [(l, b), (l, t), (r, t), (r, b)]:
        coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
        coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
        pole_coords.append(coord)
    gfxdraw.aapolygon(self.surf, pole_coords, (139, 69, 19))
    gfxdraw.filled_polygon(self.surf, pole_coords, (139, 69, 19))

    # Draw the axle
    gfxdraw.aacircle(
        self.surf,
        int(cartx),
        int(carty + axleoffset),
        int(polewidth / 2),
        (129, 132, 203),
    )
    gfxdraw.filled_circle(
        self.surf,
        int(cartx),
        int(carty + axleoffset),
        int(polewidth / 2),
        (129, 132, 203),
    )

    # Render text for step count and episode number
    try:
        font = pygame.font.Font('freesansbold.ttf', 32)  # Load the font
    except FileNotFoundError:
        font = pygame.font.SysFont(None, 32)  # Fallback to default system font

    white = (255, 255, 255)
    green = (0, 255, 0)
    blue = (0, 0, 128)
    black = (0, 0, 0)
    text = font.render(f'{self.t} | {self.epi*50}', True, black)
    text = pygame.transform.flip(text, False, True)
    textRect = text.get_rect()
    textRect.center = (self.screen_width // 6 * 5, self.screen_height // 6 * 5)
    self.surf.blit(text, textRect)

    self.surf = pygame.transform.flip(self.surf, False, True)
    self.screen.blit(self.surf, (0, 0))
    if self.render_mode == "human":
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()
    elif self.render_mode == "rgb_array":
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screens is not None:
            import pygame

            pygame.quit()