from __future__ import annotations

import argparse

from train_reppo_mjx import CONFIGS, evaluate_saved_model


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", required=True)
    parser.add_argument("--env", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-eval", type=int, default=None)
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    overrides = parse_args()
    cfg = dict(CONFIGS[0])
    if overrides["env"] is not None:
        cfg["env_name"] = overrides["env"]
    if overrides["num_envs"] is not None:
        cfg["num_envs"] = overrides["num_envs"]
    if overrides["device"] is not None:
        cfg["device"] = overrides["device"]
    if overrides["num_eval"] is not None:
        cfg["num_eval"] = overrides["num_eval"]

    evaluate_saved_model(overrides["load_path"], cfg)