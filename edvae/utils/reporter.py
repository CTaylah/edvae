import shutil
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Reporter(SummaryWriter):
    def __init__(self, log_dir: str, experiment_name: str = "", snapshot_directory=None):
        self.log_dir = log_dir
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.file_name = experiment_name + current_time
        self.file_folder = log_dir + '/' + experiment_name + current_time
        super().__init__(self.file_folder)

        self.losses = {}
        self.zero_losses()

    def accumulate_loss(self, tag, value):
        self.losses[tag] += value


    # Zero losses. Not doing so may cause a memory leak...
    def zero_losses(self):
        for loss in self.losses:
            self.losses[loss] = 0
    
    # Add a list of loss labels to the reporter
    def add_loss_labels(self, names: list[str]):
        for name in names:
            self.losses[name] = 0

    def write_loss(self, key: str, iteration: int,  divisor: float=1):
        self.add_scalar(key, self.losses[key] / divisor, iteration)



# Exapmle usage:
# reporter = Reporter('./logs/vae_gan', './logs/snapshots')
# reporter.add_loss_labels(['MSE Loss', 'KL Loss',]) 

# for epoch in range(num_epochs):
    # for batch in batches:
                #Caluculate error and backprop
                # reporter.accumulate_loss('MSE Loss', mse.item())
                # reporter.accumulate_loss('KL Loss', kl.item())
    
    # reporter.write_losses(epoch, len(batches))
    # reporter.zero_losses()

# reporter.close()