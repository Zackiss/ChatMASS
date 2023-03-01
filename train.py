import time
from collections import defaultdict

from torch.utils.data import DataLoader
import torch


class Trainer:
    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.loss = None
        self.train_dataset = train_dataset

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def run(self):
        model, config = self.model, self.config

        # set up the optimizer
        model.optimize_check()
        self.optimizer = model.optimize()
        # set up the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        # set the model module to training mode
        model.train()
        # store the iter times
        self.iter_num = 0
        # record the start time
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            try:
                # fetch the next batch (x, y)
                batch = next(data_iter)
            except StopIteration:
                # re-init iterator if needed
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch
            # forward the model, calculate loss
            logit, self.loss = model(x, y)

            # backward and update the parameters
            model.zero_grad(set_to_none=True)
            # backward pass the loss
            self.loss.backward()
            # solve the gradient explode problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            # update parameters inside layers
            self.optimizer.step()

            self.iter_num += 1
            # calculate the iter time by stored start time
            t_now = time.time()
            self.iter_dt = t_now - self.iter_time
            self.iter_time = t_now

            # termination conditions, control the max loops training will go on
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
