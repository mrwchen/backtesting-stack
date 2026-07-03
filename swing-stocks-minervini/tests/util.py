import dataclasses

from src.config import Config


def make_cfg(**overrides) -> Config:
    return dataclasses.replace(Config.from_env(), **overrides)
