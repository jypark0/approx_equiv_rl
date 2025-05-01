import argparse
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

import utils
from dmc import NoisyPixelsWrapper, make
from train import make_agent
from video import VideoRecorder


def main(args):
    root_dir = Path(args.snapshot_dir)

    # Get wandb project and id from experiment_dir
    with hydra.initialize(config_path="cfgs", version_base=None):
        exp_cfg = hydra.compose(
            config_name="config",
            overrides=OmegaConf.load((root_dir) / ".hydra/overrides.yaml"),
        )

    domain = exp_cfg.domain
    task = exp_cfg.task
    variant = exp_cfg.variant

    with open(root_dir / "wandb_id.txt") as f:
        run_id = f.read().strip()

    print("Reading config from wandb...")
    api = wandb.Api(timeout=59)

    run = api.run(f"{domain}_{task}_{variant}/{run_id}")

    config = run.config

    # Create agent
    print("Creating agent and loading snapshot...")
    agent_cfg = OmegaConf.create(config["agent"])
    agent = make_agent(domain, task, variant, agent_cfg)

    snapshot = root_dir / "snapshot.pt"
    with snapshot.open("rb") as f:
        payload = torch.load(f, map_location="cpu")

    agent.load(payload)

    # Create envs
    domain = config["domain"]
    task = config["task"]
    variant = config["variant"]

    test_orig_env = make(
        domain,
        task,
        variant,
        config["frame_stack"],
        config["action_repeat"],
        seed=config["seed"] + 1000,
    )

    test_noisy_env = make(
        domain,
        task,
        variant,
        config["frame_stack"],
        config["action_repeat"],
        seed=config["seed"] + 1000,
    )
    test_noisy_env = NoisyPixelsWrapper(test_noisy_env, args.noise_std)

    # Test
    results = {
        "orig_rewards": [],
        "noisy_rewards": [],
        "orig_lengths": [],
        "noisy_lengths": [],
    }

    global_step = payload["_global_step"]

    for env, name in zip([test_orig_env, test_noisy_env], ["orig", "noisy"]):
        episode = 0
        test_until_episode = utils.Until(args.test_num_episodes)

        with tqdm(total=args.test_num_episodes, desc=f"[{name}]", disable=True) as pbar:
            while test_until_episode(episode):
                step, total_reward = 0, 0

                time_step = env.reset()
                video_path = root_dir / f"test/{name}_video"
                video_path.mkdir(parents=True, exist_ok=True)
                video_recorder = VideoRecorder(video_path)

                video_recorder.init(env, enabled=(episode in range(5)))
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(agent):
                        action = agent.act(
                            time_step.observation, global_step, eval_mode=True
                        )
                    time_step = env.step(action)
                    video_recorder.record(env)
                    total_reward += time_step.reward
                    step += 1

                episode += 1
                pbar.update(1)

                video_recorder.save(f"episode_{episode}.mp4")

                results[f"{name}_rewards"].append(total_reward)
                results[f"{name}_lengths"].append(step * config["action_repeat"])

        print(
            f"[{name}] Reward: {np.mean(results[f'{name}_rewards']):.3f} ± {np.std(results[f'{name}_rewards'], ddof=1) / np.sqrt(args.test_num_episodes):.3f}, Length: {np.mean(results[f'{name}_lengths']):.3f} ± {np.std(results[f'{name}_lengths'], ddof=1) / np.sqrt(args.test_num_episodes):.3f}"
        )

    np.savez(root_dir / "test_results.npz", **results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_dir", type=str, required=True)
    parser.add_argument("--test_num_episodes", type=int, default=50)
    parser.add_argument("--noise_std", type=float, default=0.02)

    args = parser.parse_args()

    main(args)
