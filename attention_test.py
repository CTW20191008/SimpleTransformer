import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # 将输入特征映射到注意力权重

    def forward(self, x):
        # x 的形状是 (batch_size, 3, feature_dim)
        scores = self.linear(x)  # 计算注意力分数，形状为 (batch_size, 3, 1)
        weights = F.softmax(scores, dim=1)  # 计算注意力权重，形状为 (batch_size, 3, 1)

        # 加权求和，形状为 (batch_size, 1, feature_dim)
        weighted_sum = (weights * x).sum(dim=1, keepdim=True)
        return weighted_sum

# 假设 output 是形状为 (100, 3, 10) 的张量
output = torch.randn(100, 3, 10)

# 创建注意力层并应用
attention_layer = AttentionLayer(input_dim=10)
final_output = attention_layer(output)  # final_output 形状为 (100, 1, 10)

print(final_output.shape)  # 输出: torch.Size([100, 1, 10])
