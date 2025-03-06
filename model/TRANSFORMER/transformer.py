#Implementation of Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ------------------- 核心组件实现 -------------------
class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    #self-attention,参考self_attention.png

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v: [batch_size, seq_len, d_k]
        d_k = k.size(-1)   #获取一个多头的维度
        #MatMUL和Scale
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [batch, seq_len, seq_len]
        #Mask，Decoder中会使用
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        #softmax
        attn_weights = F.softmax(scores, dim=-1)
        #dropout
        attn_weights = self.dropout(attn_weights)
        #MatMul
        output = torch.matmul(attn_weights, v)  # [batch, seq_len, d_v]
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    #注意，本模块包含了论文中的Multi-head Attention 和Add&Norm模块

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # 线性投影层
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def split_heads(self, x):
        """拆分多头 [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]"""
        batch_size, seq_len = x.size(0), x.size(1)
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """合并多头 [batch, num_heads, seq_len, d_k] -> [batch, seq_len, d_model]"""
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def forward(self, q, k, v, mask=None):
        #注意这里接收的三个输入实际上是相同的，都是经过词嵌入（Embedding）和位置编码（Positional Encoding）后的输入矩阵
        #词嵌入能够捕捉单词的语义信息，语义相近的单词在向量空间中的距离也较近
        #位置编码的作用就是为模型引入单词的位置信息，让模型能够区分不同位置的单词。位置编码通常通过特定的函数计算得到
        #输入矩阵形状为batch_size*d,d为向量维度，论文中为512; batch_size为单词个数
        residual = q  # 残差连接

        # 线性投影 + 拆分多头，线性投影得到了矩阵Q,K,V，再将这几个矩阵分给多个头
        q = self.split_heads(self.wq(q))
        k = self.split_heads(self.wk(k))
        v = self.split_heads(self.wv(v))

        # 计算注意力
        attn_output, attn_weights = self.attention(q, k, v, mask)
        attn_output = self.combine_heads(attn_output)  # 合并多头,将多个自注意力模块输入拼接起来

        # 输出投影
        output = self.wo(attn_output)
        output = self.dropout(output)
        #Add&Norm
        output = self.layer_norm(output + residual)
        return output, attn_weights


class PositionWiseFFN(nn.Module):
    """位置前馈网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        #Add&Norm
        x = self.layer_norm(x + residual)
        return x


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ------------------- Encoder & Decoder 层实现 -------------------
class EncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x, _ = self.self_attn(x, x, x, mask)
        x = self.ffn(x)
        return x


class DecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 自注意力（带目标序列掩码）
        x, _ = self.self_attn(x, x, x, tgt_mask)
        # 交叉注意力（查询来自解码器，键/值来自编码器）
        x, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.ffn(x)
        return x


# ------------------- 完整Transformer实现 -------------------
class Transformer(nn.Module):
    """完整Transformer模型"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 词嵌入 + 位置编码
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 编码器堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 解码器堆叠
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            src_emb = layer(src_emb, src_mask)
        return src_emb

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        tgt_emb = self.pos_encoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, encoder_output, src_mask, tgt_mask)
        return tgt_emb

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_layer(decoder_output)

    def generate_mask(self, src, tgt, pad_idx=0):
        # 创建源序列填充掩码和目标序列填充掩码[batch, 1, 1, src_len]，主要是为了解决源句子和目标句子长度不一致的问题，pad_idx是用来填充的值
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]

        # 创建未来掩码，避免泄露
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool().to(tgt.device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask  # [batch, 1, tgt_len, tgt_len]

        return src_mask, tgt_mask


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    #使用翻译语言作为例子
    # 超参数
    src_vocab_size = 1000  # 源语言词汇表大小
    tgt_vocab_size = 800  # 目标语言词汇表大小
    d_model = 512
    num_heads = 8
    num_layers = 6
    batch_size = 32
    seq_len = 20

    # 初始化模型
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers)

    # 生成模拟数据，即源语言词表和目标语言的词表
    src = torch.randint(0, src_vocab_size, (batch_size, seq_len))  # 源序列
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))  # 目标序列

    # 生成掩码
    src_mask, tgt_mask = model.generate_mask(src, tgt)

    # 前向传播
    output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
    print("Output shape:", output.shape)  # [batch_size, tgt_seq_len-1, tgt_vocab_size]

    # 计算损失（假设目标序列为tgt_input）
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss = criterion(output.view(-1, tgt_vocab_size), tgt[:, 1:].contiguous().view(-1))
    print("Loss:", loss.item())