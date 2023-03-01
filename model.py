import torch
import torch.nn as nn
from Layers.block import Block
from torch.nn.functional import cross_entropy, softmax


class Model(nn.Module):
    """the main model implementation"""
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        # main structure of model: several blocks, one linear normal and one linear to assign probability
        assigned_blocks = [Block(config)
                           for _ in range(config.layer_num)]
        self.param_dict = {}
        self.train_passport = {
            "decay": [],
            "static": []
        }
        self.pipeline = nn.Sequential(*assigned_blocks)
        self.dropout = nn.Dropout(config.embedding_num)
        self.normalize = nn.LayerNorm(config.embedding_num)
        self.decode_head = nn.Linear(config.embedding_num, config.vocab_size, bias=False)
        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_num)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.embedding_num))
        self.apply(self.weight_assign)

    @staticmethod
    def model_preset(self, module):
        """set initial weight for all trainable or static layers"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def optimize_check(self):
        """separate out all parameters to those that will and won't experience decay and parameter shrink"""
        # we only consider optimizing (params decay to shrink model) the block layer, exclude normalize and embedding
        for module_name, module in self.named_modules():
            for parameter_name, parameter in module.named_parameters():
                # full param name
                fpn = '%s.%s' % (module_name, parameter_name) if module_name else parameter_name
                if parameter_name.endswith('bias'):
                    # all biases will not be decayed
                    self.train_passport["static"].append(fpn)
                elif parameter_name.endswith('weight') and isinstance(module, (torch.nn.LayerNorm, torch.nn.Embedding)):
                    # weights of blacklist modules will NOT be weight decayed
                    self.train_passport["static"].append(fpn)
                elif parameter_name.endswith('weight') and isinstance(module, (torch.nn.Linear,)):
                    # weights of whitelist modules will be weight decayed
                    self.train_passport["decay"].append(fpn)
        self.param_dict = {pn: p for pn, p in self.named_parameters()}

    def optimize(self):
        """generate optimizer of parameters"""
        optim_groups = [
            {"params": [self.param_dict[pn] for pn in sorted(self.train_passport["static"])],
             "weight_decay": self.config.weight_decay},
            {"params": [self.param_dict[pn] for pn in sorted(self.train_passport["decay"])], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=self.config.train_betas)
        return optimizer

    def forward(self, inputs, targets=None):
        """the forward process of model"""
        # embedding of input sequences
        token_embeddings = self.token_embedding(inputs)
        position_embeddings = self.position_embedding[:, :inputs.size(1), :]
        embeddings = token_embeddings + position_embeddings
        # forward procedures of model
        logit = self.decode_head(
            self.normalize(
                self.pipeline(
                    self.dropout(embeddings)
                )
            )
        )
        loss = cross_entropy(logit.view(-1, logit.size(-1), targets.view(-1))) if targets is not None else None
        return logit, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=0.75, do_sample=False, top_k=None):
        """
        From Karpathy, 3/1/2023
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        index = ids
        for _ in range(max_new_tokens):
            # if the sequence context is too long, we must crop it at block_size
            idx_cond = index if index.size(1) <= self.config.block_size else index[:, -self.config.block_size:]
            # forward the model to get the logit for the index in the sequence
            logit, _ = self(idx_cond)
            # scale by desired temperature
            logit = logit[:, -1, :] / temperature
            # we only select top k options
            if top_k is not None:
                # get ordered sequence v, which is top k options of all logit
                v, _ = torch.topk(logit, top_k)
                logit[logit < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logit to (normalized) probabilities
            probs = softmax(logit, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                # sample from indexes, get result by probability
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # we only take top-rated answer as sample
                _, idx_next = torch.topk(probs, k=1)
            # append sampled index to the running sequence and continue
            index = torch.cat((index, idx_next), dim=1)
        return index
