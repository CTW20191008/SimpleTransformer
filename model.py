import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """A multi-head self-attention module."""

    def __init__(self, embed_size, heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
        N = query.shape[0]  # batch size
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Get Q, K, V matrices
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim)
        keys = self.keys(key).view(N, key_len, self.heads, self.head_dim)
        values = self.values(value).view(N, value_len, self.heads, self.head_dim)

        # Transpose to get dimensions (N, heads, query_len, head_dim)
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        # Calculate the attention scores
        energy = torch.einsum("nhqd,nhkd->nhqk", queries, keys)  # (N, heads, query_len, key_len)
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)

        # Get the weighted value vectors
        attention_out = torch.einsum("nhql,nhld->nhqd", attention, values)  # (N, heads, query_len, head_dim)

        # Concatenate heads and pass through the final linear layer
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous()  # (N, query_len, heads, head_dim)
        attention_out = attention_out.view(N, query_len, self.embed_size)  # (N, query_len, embed_size)

        fc_out = self.fc_out(attention_out)
        return fc_out


class SimpleSelfAttention(nn.Module):
    """A simple self-attention module."""

    def __init__(self, embed_size, heads=1):
        super(SimpleSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
        # Get Q, K, V matrices
        queries = self.queries(query)
        keys = self.keys(key)
        values = self.values(value)

        # Calculate the attention scores
        energy = torch.bmm(queries, keys.transpose(1, 2))
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)

        # Get the weighted value vectors
        attention_out = torch.bmm(attention, values)
        fc_out = self.fc_out(attention_out)
        return fc_out


class SimpleTransformerBlock(nn.Module):
    """A simple transformer block."""

    def __init__(self, embed_size, input_length):
        super(SimpleTransformerBlock, self).__init__()
        # self.attention = SimpleSelfAttention(embed_size)
        self.attention = MultiHeadSelfAttention(embed_size)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm([input_length, embed_size])

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
        )

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)

        # Add skip connection, followed by LayerNorm
        x = self.norm1(attention + query)

        forward = self.feed_forward(x)
        # Add skip connection, followed by LayerNorm
        out = self.norm2(forward + x)
        return out


class PositionalEncoding(nn.Module):
    """Add positional encoding to the input tensor."""

    def __init__(self, embed_size, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                position = torch.tensor([[pos]], dtype=torch.float32)
                div_term = torch.pow(
                    10000, (2 * (i // 2)) / torch.tensor(embed_size).float()
                )
                self.encoding[pos, i] = torch.sin(position / div_term)
                self.encoding[pos, i + 1] = torch.cos(position / div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, : x.size(1), :].to(x.device)


class SimpleTransformer(nn.Module):
    """A simple transformer model."""

    def __init__(self, embed_size, input_length, vocab_size, fusion='sum', output_length=1):
        super(SimpleTransformer, self).__init__()
        self.input_length = input_length
        self.fusion = fusion

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, input_length)
        self.transformer_block = SimpleTransformerBlock(embed_size, input_length)
        self.fc_1 = nn.Linear(input_length, output_length)
        self.fc_2 = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        embedding = self.embed(x)
        # print(f"[TMP]: embedding shape is {embedding.shape}")
        # Add positional encoding
        embedding += self.pos_encoder(embedding)
        # print(f"[TMP]: embedding+position shape is {embedding.shape}")
        transformer_out = self.transformer_block(embedding, embedding, embedding)
        # print(f"[TMP]: transformer_out shape is {transformer_out.shape}")

        ## Do fusion
        transformer_out = transformer_out.permute(0, 2, 1)
        if self.fusion == "fc":
            # Do fc
            out = self.fc_1(transformer_out)
        elif self.fusion == "max_pooling":
            # Do max pooling
            out = F.max_pool1d(transformer_out, kernel_size=self.input_length, stride=self.input_length)
        else:
            # Do sum
            out = torch.sum(transformer_out, dim=2, keepdim=True)
        out = out.permute(0, 2, 1)
        # print(f"[TMP]: attention_out shape is {out.shape}")

        out = self.fc_2(out)
        # print(f"[TMP]: fc_out shape is {out.shape}")

        return out
