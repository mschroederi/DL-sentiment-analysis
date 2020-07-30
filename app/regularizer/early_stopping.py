import torch
import math

class EarlyStopping():
    def __init__(self, patience: int=5, checkpoint_path: str='best_model.pt', verbose: bool=True):
        super().__init__()
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.best_validation_loss = math.inf
        self.counter = 0

    def track(self, epoch: int, model, validation_loss: int):
        if validation_loss < self.best_validation_loss:
            if self.verbose:
                print('Validation loss decreased from {:.4f} to {:.4f} in epoch {}.  Creating model checkpoint ...\n'.format(self.best_validation_loss, validation_loss, epoch))
            self.best_validation_loss = validation_loss
            self.save_model(model)
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter > self.patience:
                return True
    
    def save_model(self, model):
        torch.save(model, self.checkpoint_path)

    def get_best_version(self, model):
        if self.best_validation_loss is math.inf:
            raise Exception('Cannot bet best model. No model stored yet.')
        return torch.load(self.checkpoint_path)
