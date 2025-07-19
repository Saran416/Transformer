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

    def forward(self, X):
        X = self.norm1(X + self.mha(X))
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

    def forward(self, X, enc_out):
        X = self.norm1(X + self.self_attn(X))
        X = self.norm2(X + self.cross_attn(X, enc_out))
        X = self.norm3(X + self.ff(X))
        return X

class Transformer(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_len):
        super().__init__()
        self.encoder_embedding = nn.Embedding(encoder_vocab_size, embed_dim)
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, embed_dim)

        self.encoder_pos = PositionalEncoding(embed_dim, max_len)
        self.decoder_pos = PositionalEncoding(embed_dim, max_len)

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)
        ])

        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, decoder_vocab_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, src, tgt):
        src = self.encoder_embedding(src)
        src = self.encoder_pos(src)

        tgt = self.decoder_embedding(tgt)
        tgt = self.decoder_pos(tgt)

        enc_out = src
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)

        dec_out = tgt
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out)

        logits = self.output_projection(dec_out)
        return logits

