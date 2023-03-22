import logging
import time
from torch.utils.data import DataLoader
import torch


class Trainer:
    """"""
    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.loss = None
        self.train_dataset = train_dataset

        self.model = self.model.to(self.config.device)
        logging.info("running on device " + self.config.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def run(self):
        """
        epoch: process that fully walk through the whole train set
        batch size: truncate the whole train set into several batches of samples
        iteration: process one batch in training
        """
        model, config = self.model, self.config
        # set up the optimizer
        model.optimize_check()
        self.optimizer = model.optimize()

        # set up the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e8)),
            shuffle=False,
            batch_size=config.batch_size,
            num_workers=config.num_workers
            # drop_last=True
        )
        # set the model module to training mode
        model.train()
        # try:
        max_epoch = config.epoch
        for i in range(max_epoch):
            # store the iter times
            self.iter_num = 0
            # record the start time
            self.iter_time = time.time()
            data_iter = iter(train_loader)
            print(f"training on epoch {i}")
            # In one epoch, we train on all samples iteratively
            while True:
                try:
                    # fetch the next batch with size of batch-size
                    batch = next(data_iter)
                except StopIteration:
                    # re-init iterator if needed
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                # [(x, y), (x, y), (x, y),...]
                batch = [t.to(self.config.device) for t in batch]
                # train per sample, the smallest unit in training procedure
                # set iteration time to 0, if iteration time > max iteration depth, jump to next batch
                self.iter_num = 0
                x, y = batch
                # forward the model, calculate loss
                for _ in range(config.steps_per_sample):
                    logit, self.loss = model(x, y, z)
                    logging.info("loss at {0}".format(self.loss.item()))
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

                    # termination conditions, control the max loops training will go on
                    if self.iter_num % 20 == 0:
                        self.test(model, query="你应该看看昨晚播出的节目，")
                    torch.cuda.empty_cache()
                self.iter_num += 1
                # calculate the iter time of a batch
                t_now = time.time()
                self.iter_dt = t_now - self.iter_time
                self.iter_time = t_now
                print("iter {0} reached in {1}m".format(self.iter_num, self.iter_time))
                # If current batch iteration out of depth, jump to the next batch
                if config.max_iters is not None:
                    if self.iter_num >= config.max_iters:
                        break
                elif self.iter_num >= 1e3:
                    break
            if self.loss.item() < 0.01:
                # if loss reach requirement, stop epoch training, start next epoch
                torch.save(model.state_dict(), "trained_model_finish")
                break
        # except RuntimeError as e:
        #     torch.save(model.state_dict(), "trained_model_protect")
        #     torch.cuda.empty_cache()
        #     print(e)

    def test(self, model, query):
        # test period
        x = torch.tensor([self.train_dataset.sentence_parse_to_ids(query)], dtype=torch.long).to(self.config.device)
        y = model.generate(x, 20, temperature=0.6, do_sample=False, top_k=35)[0]
        completion = "".join(self.train_dataset.idx_to_words(y))
        logging.info("test result at iter {0}: ".format(self.iter_num) + completion)



