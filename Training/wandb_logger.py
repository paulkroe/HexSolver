from dotenv import load_dotenv
import os
import wandb

load_dotenv()

class WandBLogger:
    is_logged_in = False

    def __init__(self, enabled=True, run_name=None, model=None):
        self.enabled = enabled

        if self.enabled:
            self.login()
            self.run = wandb.init(entity="tpu",
                                  project="hex-solver",
                                  reinit=True)

            if model is not None:
                self.watch(model)

    def login(self):
        if not WandBLogger.is_logged_in:
            wandb_api_key = os.getenv("WANDB_KEY")
            wandb.login(key=wandb_api_key)
            WandBLogger.is_logged_in = True

    def watch(self, model, log_freq=1):
        wandb.watch(model, log="all", log_freq=log_freq)

    def log(self, log_dict, commit=True, step=None):
        if self.enabled:
            if step:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)

    def finish(self):
        if self.enabled:
            self.run.finish()
