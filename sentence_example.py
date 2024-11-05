import torch
import torch.nn as nn


class CrossAttention(nn.Module):

    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))

    def forward(self, x_1, x_2):           # x_2 is new
        queries_1 = x_1 @ self.W_query

        keys_2 = x_2 @ self.W_key          # new
        values_2 = x_2 @ self.W_value      # new

        attn_scores = queries_1 @ keys_2.T # new
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1)

        context_vec = attn_weights @ values_2
        return context_vec


sentence = 'Life is short, eat dessert first'
dc = {s:i for i,s 
      in enumerate(sorted(sentence.replace(',', '').split()))}
print(f"[INFO]: dc is {dc}")

sentence_int = torch.tensor(
    [dc[s] for s in sentence.replace(',', '').split()]
)
print(f"[INFO]: sentence_int is {sentence_int}")

vocab_size = 50_000
torch.manual_seed(123)
embed = torch.nn.Embedding(vocab_size, 3)
embedded_sentence = embed(sentence_int).detach()
print(f"[INFO]: embedded_sentence is {embedded_sentence}")
print(f"[INFO]: embedded_sentence shape is {embedded_sentence.shape}")

d_in, d_out_kq, d_out_v = 3, 2, 4

first_input = embedded_sentence
second_input = torch.rand(8, d_in)
print("[INFO]: First input shape:", first_input.shape)
print("[INFO]: Second input shape:", second_input.shape)

crossattn = CrossAttention(d_in, d_out_kq, d_out_v)
context_vectors = crossattn(first_input, second_input)
print(context_vectors)
print("Output shape:", context_vectors.shape)
