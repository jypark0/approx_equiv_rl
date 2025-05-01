# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs


class NoisyPixelsWrapper(dm_env.Environment):
    def __init__(self, env, noise_level):
        self._env = env
        self._noise_level = noise_level

    def step(self, action):
        time_step = self._env.step(action)
        obs = time_step.observation

        noise = (
            255 * self._env.task.random.normal(0, self._noise_level, obs.shape)
        ).astype(np.uint8)

        obs = np.clip(obs + noise, 0, 255)
        return time_step._replace(observation=obs)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class GravityWrapper(dm_env.Environment):
    def __init__(self, env, gravity):
        env.physics.model.opt.gravity = gravity
        self._env = env

    def step(self, action):
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class TransformPixelsWrapper(dm_env.Environment):
    def __init__(self, env, transform):
        assert transform in ["identity", "c2_vertical", "c2_horizontal", "d2"]
        self.transform = transform
        self._env = env

        self._g_id = None
        self._domain_name = self._env.task.__module__.split(".")[-1]

    def step(self, action):
        if self._g_id != 0:
            # Transform actions if necessary
            if self._domain_name == "acrobot":
                action = -action
            elif self._domain_name == "cartpole":
                action = -action
            elif self._domain_name == "ball_in_cup":
                if self.transform == "c2_vertical":
                    action[0] = -action[0]
                elif self.transform == "c2_horizontal":
                    action[1] = -action[1]

            elif self._domain_name == "reacher":
                if self._g_id in [2, 3]:
                    action = -action

        time_step = self._env.step(action)

        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def _transform_observation(self, time_step):
        if self._g_id == 0:
            return time_step
        else:
            obs = time_step.observation

            # Obs: [C, H, W]
            if self.transform == "c2_vertical":
                obs = np.flip(obs, axis=2)
            elif self.transform == "c2_horizontal":
                obs = np.flip(obs, axis=1)
            elif self.transform == "d2":
                if self._g_id == 1:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                elif self._g_id == 2:
                    obs = np.flip(obs, axis=2)
                elif self._g_id == 3:
                    obs = np.rot90(obs, k=2, axes=(1, 2))
                    obs = np.flip(obs, axis=2)

        return time_step._replace(observation=obs.copy())

    def reset(self):
        time_step = self._env.reset()

        if self.transform in ["c2_vertical", "c2_horizontal"]:
            self._g_id = self._env.task._random.randint(2)
        elif self.transform == "d2":
            self._g_id = self._env.task._random.randint(4)
        else:
            self._g_id = 0

        return self._transform_observation(time_step)

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ConditionalActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats
        self._domain_name = self._env.task.__module__.split(".")[-1]

    def step(self, action):
        reward = 0.0
        discount = 1.0

        if self._domain_name == "acrobot":
            # [joint1_x, joint2_x, joint1_y, joint2_y]
            cur_pos = self._env.physics.orientations()
            # Joint 1 in fourth quadrant
            cond = cur_pos[0] > 0.0 and cur_pos[2] < 0.0
        elif self._domain_name == "cartpole":
            # [cart_x, pole_sin, pole_cos]
            cur_pos = self._env.physics.bounded_position()
            # Pole in first quadrant
            cond = cur_pos[1] > 0.0 and cur_pos[2] > 0.0
        elif self._domain_name == "ball_in_cup":
            # [cup_x, cup_z, ball_x, ball_z]
            cur_pos = self._env.physics.position()
            # Only repeat if ball is in top right of image
            cond = cur_pos[2] > 0.0 and cur_pos[3] > 0.3
        elif self._domain_name == "reacher":
            # [shoulder_theta, wrist_theta]
            cur_pos = self._env.physics.position()
            # Shoulder in first quadrant
            cond = cur_pos[0] > 0.0 and cur_pos[0] < np.pi / 2
        else:
            raise NotImplementedError

        if cond:
            for i in range(2 * self._num_repeats):
                time_step = self._env.step(action)
                reward += (time_step.reward or 0.0) * discount
                discount *= time_step.discount
                if time_step.last():
                    break
            return time_step._replace(reward=reward, discount=discount)
        else:
            for i in range(self._num_repeats):
                time_step = self._env.step(action)
                reward += (time_step.reward or 0.0) * discount
                discount *= time_step.discount
                if time_step.last():
                    break
            return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ReflectActionWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self._domain_name = self._env.task.__module__.split(".")[-1]

    def step(self, action):
        if self._domain_name == "acrobot":
            # [joint1_x, joint2_x, joint1_y, joint2_y]
            cur_pos = self._env.physics.orientations()
            cond = cur_pos[0] > 0.0 and -0.5 < cur_pos[2] < 0.0
        elif self._domain_name == "cartpole":
            # [cart_x, pole_sin, pole_cos]
            cur_pos = self._env.physics.bounded_position()
            cond = 0.0 < cur_pos[1] < 0.5 and cur_pos[2] > 0.0
            pass
        elif self._domain_name == "ball_in_cup":
            # [cup_x, cup_z, ball_x, ball_z]
            cur_pos = self._env.physics.position()
            cond = cur_pos[2] > 0.0 and cur_pos[1] > 0.3
        elif self._domain_name == "reacher":
            # [shoulder_theta, wrist_theta]
            cur_pos = self._env.physics.position()
            cond = np.pi / 2 < cur_pos[0] < np.pi
        else:
            raise NotImplementedError

        if cond:
            action = -action

        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key="pixels"):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(domain, task, variant, frame_stack, action_repeat, seed, transform=False):
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(
            domain, task, task_kwargs={"random": seed}, visualize_reward=False
        )
        pixels_key = "pixels"
    else:
        name = f"{domain}_{task}_vision"
        env = manipulation.load(name, seed=seed)
        pixels_key = "front_close"
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)

    # Variants
    if domain == "acrobot":
        if variant == "gravity":
            env = GravityWrapper(env, gravity=[-2, 2, -9.81])
            env = ActionRepeatWrapper(env, action_repeat)
        elif variant == "repeat_action":
            env = ConditionalActionRepeatWrapper(env, action_repeat)
        else:
            env = ActionRepeatWrapper(env, action_repeat)
    elif domain == "cartpole":
        if variant == "gravity":
            env = GravityWrapper(env, gravity=[0.2, -0.2, -9.81])
            env = ActionRepeatWrapper(env, action_repeat)
        elif variant == "repeat_action":
            env = ConditionalActionRepeatWrapper(env, action_repeat)
        elif variant == "reflect_action":
            env = ReflectActionWrapper(env)
            env = ActionRepeatWrapper(env, action_repeat)
        else:
            env = ActionRepeatWrapper(env, action_repeat)
    elif domain == "ball_in_cup":
        if variant == "gravity":
            env = GravityWrapper(env, gravity=[-2, 2, -9.81])
            env = ActionRepeatWrapper(env, action_repeat)
        elif variant == "repeat_action":
            env = ConditionalActionRepeatWrapper(env, action_repeat)
        elif variant == "reflect_action":
            env = ReflectActionWrapper(env)
            env = ActionRepeatWrapper(env, action_repeat)
        else:
            env = ActionRepeatWrapper(env, action_repeat)
    elif domain == "reacher":
        if variant == "repeat_action":
            env = ConditionalActionRepeatWrapper(env, action_repeat)
        elif variant == "reflect_action":
            env = ReflectActionWrapper(env)
            env = ActionRepeatWrapper(env, action_repeat)
        else:
            env = ActionRepeatWrapper(env, action_repeat)

    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=85, width=85, camera_id=camera_id)
        env = pixels.Wrapper(env, pixels_only=False, render_kwargs=render_kwargs)

    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)

    # Transform image if necessary
    if transform:
        if domain == "acrobot":
            if variant == "horizontal":
                env = TransformPixelsWrapper(env, "c2_horizontal")
            else:
                env = TransformPixelsWrapper(env, "c2_vertical")
        elif domain == "cartpole":
            env = TransformPixelsWrapper(env, "c2_vertical")
        elif domain == "ball_in_cup":
            env = TransformPixelsWrapper(env, "c2_vertical")
        elif domain == "reacher":
            env = TransformPixelsWrapper(env, "d2")

    env = ExtendedTimeStepWrapper(env)
    return env
