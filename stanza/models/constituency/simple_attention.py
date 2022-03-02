import logging

import torch.nn as nn

from stanza.models.constituency.positional_encoding import ConcatSinusoidalEncoding

logger = logging.getLogger('stanza')

class SimpleAttentionModule(nn.Module):
    def __init__(self,
                 n_layers,
                 n_heads,
                 d_input,
                 d_model,
                 d_timing,
                 d_feed_forward):
        super().__init__()

        if d_model <= d_timing:
            d_model += d_timing
            logger.warning("d_model <= d_timing.  changing d_model to %d", d_model)

        if d_model % n_heads != 0:
            d_model = d_model + n_heads - d_model % n_heads
            logger.warning("d_model %% n_heads != 0.  changing d_model to %d", d_model)

        self.d_model = d_model
        self.attn_proj = nn.Linear(d_input, d_model - d_timing)
        self.attn_timing = ConcatSinusoidalEncoding(d_model=d_timing)
        self.attn_layers = nn.ModuleList([nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                                          for _ in range(n_layers)])
        self.linear_in = nn.ModuleList([nn.Linear(d_model, d_feed_forward)
                                        for _ in range(n_layers)])
        self.linear_out = nn.ModuleList([nn.Linear(d_feed_forward, d_model)
                                         for _ in range(n_layers)])
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        x = self.attn_proj(x)
        x = self.attn_timing(x)

        for layer, ff_in, ff_out in zip(self.attn_layers, self.linear_in, self.linear_out):
            # TODO: residual dropout if this is working at all
            x_attn = layer(x, x, x)[0]
            x = x + x_attn
            # TODO: layer norms?
            x_ff = self.nonlinearity(ff_out(self.nonlinearity(ff_in(x))))
            x = x + x_ff

        return x

