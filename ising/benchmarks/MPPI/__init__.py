from .dynamics import BicycleModel


def get_dynamics_model(config):
    if config.dynamics == "Bicycle":
        return BicycleModel(config.delta_t)
    else:
        raise ValueError(f"Unknown dynamics model: {config.dynamics}")
