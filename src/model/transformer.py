# type: ignore

from .attention import MultiHeadSelfAttention, CrossAttention, FeedForward
from .positional_encoding import PositionalEncoding
from torch import nn

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.mha = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, X, src_mask=None):
        X = self.norm1(X + self.mha(X, attn_mask=src_mask))
        X = self.norm2(X + self.ff(X))
        return X

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, mask=True)
        self.cross_attn = CrossAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, X, enc_out, tgt_mask=None, src_mask=None):
        X = self.norm1(X + self.self_attn(X, attn_mask=tgt_mask))
        X = self.norm2(X + self.cross_attn(X, enc_out, context_mask=src_mask))
        X = self.norm3(X + self.ff(X))
        return X

class Transformer(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_len):
        super().__init__()
        self.encoder_embedding = nn.Embedding(encoder_vocab_size, embed_dim, padding_idx=0)
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, embed_dim, padding_idx=0)

        self.encoder_pos = PositionalEncoding(embed_dim, max_len)
        self.decoder_pos = PositionalEncoding(embed_dim, max_len)

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)
        ])

        self.output_projection = nn.Linear(embed_dim, decoder_vocab_size)

    def make_pad_mask(self, seq, pad_idx=0):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, src, tgt):
        src_mask = self.make_pad_mask(src)
        tgt_mask = self.make_pad_mask(tgt)

        src = self.encoder_pos(self.encoder_embedding(src))
        tgt = self.decoder_pos(self.decoder_embedding(tgt))

        enc_out = src
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask=src_mask)

        dec_out = tgt
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)

        logits = self.output_projection(dec_out)
        return logits