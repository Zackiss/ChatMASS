import torch
import torch.nn as nn
from layers.encode_block import Block as EncoderBlock
from layers.decode_block import Block as DecoderBlock
from torch.nn.functional import cross_entropy, softmax


class Model(nn.Module):
    """the main model implementation"""

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        # main structure of model: several blocks, one linear normal and one linear to assign probability
        self.param_dict = {}
        self.train_passport = {
            "decay": [],
            "static": []
        }
        assigned_encoder_blocks = [EncoderBlock(config)
                                   for _ in range(config.layer_num)]
        self.encoder_output = None
        assigned_decoder_blocks = [DecoderBlock(config, self.encoder_output)
                                   for _ in range(config.layer_num)]
        self.encoder_pipeline = nn.Sequential(*assigned_encoder_blocks)
        self.decoder_pipeline = nn.Sequential(*assigned_decoder_blocks)
        self.encoder_pipeline = nn.Sequential(*assigned_encoder_blocks)
        self.decoder_pipeline = nn.Sequential(*assigned_decoder_blocks)
        self.dropout = nn.Dropout(config.embedding_drop)
        self.normalize = nn.LayerNorm(config.embedding_dim)

        self.decode_head = nn.Linear(config.embedding_dim, config.block_size, bias=False)
        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim)

        self.position_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.embedding_dim))
        self.apply(self.model_preset)

    @staticmethod
    def model_preset(module):
        """set initial weight for all trainable or static layers"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.GroupNorm):
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
                elif parameter_name.endswith('weight') and isinstance(module,
                                                                      (torch.nn.LayerNorm, torch.nn.Embedding,)):
                    # weights of blacklist modules will NOT be weight decayed
                    self.train_passport["static"].append(fpn)
                elif parameter_name.endswith('weight') and isinstance(module, (torch.nn.Linear,)):
                    # weights of whitelist modules will be weight decayed
                    self.train_passport["decay"].append(fpn)
        self.param_dict = {pn: p for pn, p in self.named_parameters()}

    def optimize(self):
        """generate optimizer of parameters"""
        optimizer = torch.optim.AdamW([
            {
                "params": [self.param_dict[pn] for pn in sorted(self.train_passport["static"])],
                "weight_decay": self.config.weight_decay
            },
            {
                "params": [self.param_dict[pn] for pn in sorted(self.train_passport["decay"])],
                "weight_decay": 0.0
            },
        ], lr=self.config.learning_rate, betas=(0.9, 0.95))
        return optimizer

    def forward(self, inputs, targets=None):
        """the forward process of model"""
        # embedding of input sequences
        inputs = unmasked part of inputs
        predict_target = unmasked targets used for cross entropy loss
        input_target = apply random mask on targets

        inputs_token_embeddings = self.token_embedding(inputs)
        inputs_position_embeddings = self.position_embedding[:, :inputs.size(1), :]
        inputs_embeddings = inputs_token_embeddings + inputs_position_embeddings

        if targets is not None:
            targets_token_embeddings = self.token_embedding(input_target)
            targets_position_embeddings = self.position_embedding[:, :targets.size(1), :]
            targets_embeddings = targets_token_embeddings + targets_position_embeddings
        else:
            targets_embeddings = None

        # forward procedures of encoder
        encoder_output = self.normalize(
            self.encoder_pipeline(
                self.dropout(inputs_embeddings)
            )
        )

        # forward procedures of decoder
        self.encoder_output = self.dropout(encoder_output)
        logit = self.decode_head(
            self.normalize(
                self.decoder_pipeline(
                    targets_embeddings
                )
            )
        )

        loss = cross_entropy(logit.view(-1, logit.size(-1)),
                             predict_target.view(-1)) if targets is not None else None
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
