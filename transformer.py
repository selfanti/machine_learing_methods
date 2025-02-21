import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        位置编码层（可学习式）
        :param d_model: 特征维度
        :param max_len: 最大序列长度
        """
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pos_embedding[:, :x.size(1), :]


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """
    缩放点积注意力
    :param q: 查询矩阵 (..., seq_len_q, d_k)
    :param k: 键矩阵 (..., seq_len_k, d_k)
    :param v: 值矩阵 (..., seq_len_k, d_v)
    :param mask: 掩码矩阵（可选）
    :return: 注意力加权后的值矩阵和注意力权重
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (..., seq_len_q, seq_len_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_weights = F.softmax(scores, dim=-1)  # (..., seq_len_q, seq_len_k)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, v)  # (..., seq_len_q, d_v)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义线性变换层
        self.wq = nn.Linear(d_model, d_model)  # Q矩阵变换
        self.wk = nn.Linear(d_model, d_model)  # K矩阵变换
        self.wv = nn.Linear(d_model, d_model)  # V矩阵变换

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)  # 输出变换

    def split_heads(self, x):
        """
        将输入张量分割为多头
        :param x: (batch_size, seq_len, d_model)
        :return: (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        # 线性变换 + 分割多头
        q = self.split_heads(self.wq(q))  # (batch_size, num_heads, seq_len_q, d_k)
        k = self.split_heads(self.wk(k))  # (batch_size, num_heads, seq_len_k, d_k)
        v = self.split_heads(self.wv(v))  # (batch_size, num_heads, seq_len_v, d_k)

        # 计算注意力
        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, mask, self.dropout
        )  # attn_output: (batch_size, num_heads, seq_len_q, d_k)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            attn_output.size(0), attn_output.size(2), self.d_model
        )  # (batch_size, seq_len_q, d_model)

        # 输出线性变换
        output = self.out(attn_output)
        return output, attn_weights


class FeedForward(nn.Module):
    """位置前馈网络"""

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 前馈网络子层
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    """完整Transformer编码器"""

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ---------------------------- #
#         使用示例              #
# ---------------------------- #
if __name__ == "__main__":
    # 参数配置
    batch_size = 32
    seq_len = 50
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048

    # 生成随机输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 初始化模型
    encoder = TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )

    # 前向传播
    output = encoder(x)
    print("输入形状:", x.shape)  # torch.Size([32, 50, 512])
    print("输出形状:", output.shape)  # torch.Size([32, 50, 512])