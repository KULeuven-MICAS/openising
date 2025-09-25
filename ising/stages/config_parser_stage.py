from ising.stages import LOGGER, TOP
from typing import Any
import os
import yaml
from pathlib import Path
from argparse import Namespace
from ising.stages.stage import Stage, StageCallable

class ConfigParserStage(Stage):
    """! Stage to parse the configuration file for the Ising model simulation."""

    def __init__(self,
                 list_of_callables: list[StageCallable],
                 *,
                 problem_type: str,
                 config_path: str,
                 **kwargs: Any):
        super().__init__(list_of_callables, **kwargs)
        assert config_path.split(".")[-1] == "yaml", "Config file must be a YAML file"
        self.config_path = TOP / config_path
        self.problem_type = problem_type

    def run(self) -> Any:
        """! Parse the configuration file and return the parsed configuration."""

        LOGGER.debug(f"Parsing configuration file: {self.config_path}")
        with Path(self.config_path).open(encoding="utf-8") as file:
            config: dict = yaml.safe_load(file)
        os.sched_setaffinity(0, range(config['nb_cores']))
        config["problem_type"] = self.problem_type

        # Validate the configuration structure
        self.validate_config(config)
        LOGGER.debug(f"Configuration parsed successfully: {config}")

        # Temporarily store the config into argparse.Namespace to be compatible with old code
        config = Namespace(**config)

        self.kwargs["config"] = config
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], **self.kwargs)
        yield from sub_stage.run()

    def validate_config(self, config: dict) -> None:
        """! Validate the configuration structure."""

        pass  # Placeholder for actual validation logic
