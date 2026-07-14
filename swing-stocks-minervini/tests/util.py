import dataclasses

from src.config import Config


def make_cfg(**overrides) -> Config:
    values = {"simulation_mode": "independent"}
    values.update(overrides)
    return dataclasses.replace(Config.from_env(), **values)
