import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn


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
            # sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e8)),
            shuffle=True,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers
            # drop_last=True
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
                print("loss at {0}".format(self.loss.item()))
                # backward and update the parameters
                model.zero_grad(set_to_none=True)
                # apply the gradient accumulation
                if config.gradient_accumulation_steps != 0:
                    self.loss = self.loss / config.gradient_accumulation_steps
                    # backward pass the loss
                    self.loss.backward()
                    # solve the gradient explode problem
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    if self.iter_num % config.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    # backward pass the loss
                    self.loss.backward()
                    # solve the gradient explode problem
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.iter_num += 1
                # calculate the iter time by stored start time
                t_now = time.time()
                self.iter_dt = t_now - self.iter_time
                self.iter_time = t_now
                print("iter {0} reached in {1}m".format(self.iter_num, self.iter_time))
                # termination conditions, control the max loops training will go on
                if (config.max_iters is not None and self.iter_num >= config.max_iters) or self.loss.item() < 0.1:
                    torch.save(model.state_dict(), "trained_model")
                    break
                if self.iter_num % 20 == 0:
                    query = "她正值青春貌美的年龄，"
                    x = torch.tensor([self.train_dataset.sentence_parse_to_ids(query)], dtype=torch.long).to("cuda")
                    y = model.generate(x, 20, temperature=0.6, do_sample=False, top_k=35)[0]
                    completion = "".join(self.train_dataset.idx_to_words(y))
                    print("test result at iter {0}: ".format(self.iter_num) + completion)
                torch.cuda.empty_cache()
            except RuntimeError as e:
                torch.cuda.empty_cache()
                print(e)
                continue



