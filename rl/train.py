# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

from torch.autograd.grad_mode import enable_grad

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs
from escnn import gspaces
from omegaconf import OmegaConf, open_dict

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(domain, task, variant, cfg):
    if any(cfg.name.startswith(s) for s in ["FullEquiv", "RelaxedEquiv", "RPP"]):
        if domain == "reacher":
            gspace = gspaces.flipRot2dOnR2(N=2)
            cup_catch = False
        elif domain == "acrobot":
            if variant == "horizontal":
                gspace = gspaces.flip2dOnR2(axis=0.0)
                cup_catch = False
            else:
                gspace = gspaces.flip2dOnR2()
                cup_catch = False
        elif domain == "cartpole":
            gspace = gspaces.flip2dOnR2()
            cup_catch = False
        elif domain == "ball_in_cup":
            gspace = gspaces.flip2dOnR2()
            cup_catch = True
        else:
            raise NotImplementedError

        return hydra.utils.instantiate(
            cfg.cls,
            gspace=gspace,
            cup_catch=cup_catch,
        )

    elif cfg.name.startswith("NonEqui"):
        return hydra.utils.instantiate(cfg.cls)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create envs
        self.train_env = dmc.make(
            self.cfg.domain,
            self.cfg.task,
            self.cfg.variant,
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            self.cfg.seed,
        )
        self.eval_env = dmc.make(
            self.cfg.domain,
            self.cfg.task,
            self.cfg.variant,
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            self.cfg.seed,
        )
        # create replay buffer
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        # create agent
        with open_dict(self.cfg):
            self.cfg.agent.cls.obs_shape = data_specs[0].shape
            self.cfg.agent.cls.action_shape = data_specs[1].shape
        self.agent = make_agent(
            self.cfg.domain,
            self.cfg.task,
            self.cfg.variant,
            self.cfg.agent,
        )

        # create logger
        wandb_config = OmegaConf.to_container(
            self.cfg.wandb, resolve=True, throw_on_missing=True
        )

        # If resuming and wandb_id file exists
        if self.cfg.resume and (self.work_dir / "wandb_id.txt").is_file():
            with open(self.work_dir / "wandb_id.txt") as f:
                run_id = f.read().strip()

            wandb_config["id"] = run_id

            with open_dict(self.cfg):
                self.cfg.wandb.id = run_id

        self.logger = Logger(self.work_dir, wandb_kwargs=wandb_config)

        wandb_hyperparams = OmegaConf.to_container(
            self.cfg, resolve=True, throw_on_missing=True
        )
        self.logger.wandb_run.config.update(wandb_hyperparams, allow_val_change=True)

        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / "buffer")

        self.replay_loader = make_replay_loader(
            self.work_dir / "buffer",
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
        )
        self._replay_iter = None

        if self.cfg.save_video:
            self.eval_video_recorder = VideoRecorder(self.work_dir / "eval_video")
            self.test_orig_video_recorder = VideoRecorder(
                self.work_dir / "test_orig_video"
            )
        else:
            self.eval_video_recorder = VideoRecorder(None)
            self.test_orig_video_recorder = VideoRecorder(None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir / "train_video" if self.cfg.save_train_video else None
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.eval_video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.observation, self.global_step, eval_mode=True
                    )
                time_step = self.eval_env.step(action)
                self.eval_video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.eval_video_recorder.save(f"{self.global_frame}.mp4")

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f"{self.global_frame}.mp4")
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "total_time", self.timer.total_time(), self.global_frame, ty="eval"
                )
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(
                    time_step.observation, self.global_step, eval_mode=False
                )

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def test(self):
        # Original env
        steps, rewards, episode = [], [], 0
        test_until_episode = utils.Until(self.cfg.num_test_episodes)

        while test_until_episode(episode):
            step, total_reward = 0, 0

            time_step = self.eval_env.reset()
            self.test_orig_video_recorder.init(
                self.eval_env, enabled=(episode in range(5))
            )
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.observation, self.global_step, eval_mode=True
                    )
                time_step = self.eval_env.step(action)
                self.test_orig_video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.test_orig_video_recorder.save(f"episode_{episode}.mp4")

            with self.logger.dump_ctx(self.global_frame, ty="test_orig") as log:
                log("episode_reward", total_reward)
                log("episode_length", step * self.cfg.action_repeat)
                log("episode", episode)

            steps.append(step)
            rewards.append(total_reward)

        metrics = {
            "episode_reward_mean": np.mean(rewards),
            "episode_reward_std": np.std(rewards),
            "episode_reward_se": np.std(rewards, ddof=1) / np.sqrt(episode),
            "episode_length_mean": np.mean(steps),
            "episode_length_std": np.std(steps),
            "episode_length_se": np.std(steps, ddof=1) / np.sqrt(episode),
        }
        self.logger.log_metrics(metrics, step=self.global_step, ty="test_orig")

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        keys_to_save = ["timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        # Save agent separately
        agent_dict = self.agent.save()
        payload = {**payload, **agent_dict}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        with snapshot.open("rb") as f:
            payload = torch.load(f, map_location="cpu")
        for k, v in payload.items():
            if not k.startswith("agent"):
                self.__dict__[k] = v

        # Load agent separately
        self.agent.load(payload)

        # Get run id from wandb api


@hydra.main(version_base=None, config_path="cfgs", config_name="config")
def main(cfg):
    from train import Workspace as W

    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        if cfg.resume:
            print(f"resuming: {snapshot}")
            workspace.load_snapshot()
        else:
            print(f"not resuming with snapshot {snapshot}")
    print("[Train] Start training")
    workspace.train()

    print("[Test] Start testing")
    workspace.test()


if __name__ == "__main__":
    main()
