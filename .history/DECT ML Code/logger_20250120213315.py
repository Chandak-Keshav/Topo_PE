import os
import logging
from typing import Callable, Any
import time
import functools
import wandb


class Logger:
    def __init__(self, dev: bool = True, wandb_logger: bool = True):
        self.dev = dev
        self.wandb = None

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Ensure the logs directory exists
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        log_file_path = os.path.join(log_dir, f"test-{timestr}.logs")
        fh = logging.FileHandler(filename=log_file_path, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        # Debugging output
        self.logger.info(f"Logger initialized. Logs will be saved to: {log_file_path}")

    def wandb_init(self, config):
        self.wandb = wandb.init(
            project=config.project,
            name=config.name,
            tags=config.tags + ["dev"],
        )

    def log(self, msg: str | None = None, params: dict[str, str] | None = None) -> None:
        if msg:
            self.logger.info(msg)

        if params and self.wandb:
            self.wandb.log(params)

    def log_config(self, config):
        if self.wandb:
            self.wandb.config.update(config)
        else:
            raise ValueError()


def timing(logger: Logger, dev: bool = True):
    if dev:

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                logger.log(f"Calling {func.__name__}")
                ts = time.time()
                value = func(*args, **kwargs)
                te = time.time()
                logger.log(f"Finished {func.__name__}")
                if logger:
                    logger.log("func:%r took: %2.4f sec" % (func.__name__, te - ts))
                return value

            return wrapper

        return decorator
    else:

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                value = func(*args, **kwargs)
                return value

            return wrapper

        return decorator
