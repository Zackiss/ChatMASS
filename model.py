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
        assigned_encoder_blocks = [EncoderBlock(config)
                                   for _ in range(config.layer_num)]
        assigned_decoder_blocks = [DecoderBlock(config)
                                   for _ in range(config.layer_num)]
        self.param_dict = {}
        self.train_passport = {
            "decay": [],
            "static": []
        }
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

    def init_mask(self, target_matrix: torch.Tensor):
        mask = torch.ones(target_matrix.size())
        # Calculate the number of consecutive columns to select
        num_selected = target_matrix.size(1) // 2
        # Generate a random starting index for the selection
        start_index = torch.randint(0, target_matrix.size(1) - num_selected + 1, size=(1,)).item()
        # Set the selected columns to 0
        mask[:, start_index:start_index + num_selected] = 0
        original_matrix = mask.clone()

        # Identify the consecutive 0 columns
        first_col = torch.min(torch.any(mask, dim=0)).item()
        last_col = torch.max(torch.any(mask, dim=0)).item()
        # Calculate the shift distance
        shift_distance = last_col - first_col + 1
        # Shift the columns
        mask[:, shift_distance:] = mask[:, :-shift_distance]
        # Fill the empty columns
        mask[:, :shift_distance] = 1
        return original_matrix, mask

    def apply_mask(self, target_matrix: torch.Tensor, mask: torch.Tensor, inverse=False):
        if not inverse:
            target_matrix.masked_fill(mask, float("-inf"))
        else:
            target_matrix.masked_fill(~mask, float("-inf"))
        return target_matrix

    def forward(self, inputs, targets=None, mass_mask=False):
        """the forward process of model"""
        # embedding of input sequences
        inputs_token_embeddings = self.token_embedding(inputs)
        inputs_position_embeddings = self.position_embedding[:, :inputs.size(1), :]
        inputs_embeddings = inputs_token_embeddings + inputs_position_embeddings

        mask, shifted_mask = self.init_mask(inputs_embeddings)
        # masked input denotes x2,x3,_,_,x6,x7 which
        inputs_embeddings = self.apply_mask(inputs_embeddings, mask)

        # target denotes x1,x2,x3,x4,x5,x6 which is the whole sentence
        # masked target denotes _,_,_,x4,x5,_ which
        # the generate procedure will fill the gaps in masked input
        targets_embeddings = self.apply_mask(inputs_embeddings, shifted_mask, True)

        # forward procedures of encoder
        encoder_output = self.normalize(
            self.encoder_pipeline(
                self.dropout(inputs_embeddings)
            )
        )

        # forward procedures of decoder
        # logit denotes x1,x2,x3,x4,x5,x6, generated by model
        logit = self.decode_head(
            self.normalize(
                self.decoder_pipeline(
                    targets_embeddings,
                    self.dropout(encoder_output)
                )
            )
        )

        loss = cross_entropy(self.apply_mask(logit.view(-1, logit.size(-1)), shifted_mask, True),
                             self.apply_mask(targets.view(-1), shifted_mask, True)) if targets is not None else None
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
