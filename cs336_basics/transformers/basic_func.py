import torch
from einops import einsum, rearrange, repeat
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
import warnings


class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = self._init_weight()

    def _init_weight(self):
        theta = (2 / (self.in_features+self.out_features))**(1/2)
        min_cutoff = -3 * theta
        max_cutoff = 3 * theta
        empty_weight = torch.empty(self.out_features, self.in_features, dtype=self.dtype, device=self.device)
        weight = nn.init.trunc_normal_(empty_weight, 0, theta, min_cutoff, max_cutoff)
        return nn.Parameter(weight)

    def forward(self, x:torch.Tensor) ->torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings , embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = self._init_weight()

    def _init_weight(self):
        empty_weight = torch.empty(self.num_embeddings, self.embedding_dim, dtype=self.dtype, device=self.device)
        weight = nn.init.trunc_normal_(empty_weight, 0, 1, -3, 3)
        return nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape_token_id = x.shape
        flatten_token_ids = x.flatten()
        flatten_embeds = self.weight[flatten_token_ids]
        embeds_shape = shape_token_id + (self.embedding_dim, )
        embeds = flatten_embeds.view(size=embeds_shape)
        return embeds


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = self._init_weight()

    def _init_weight(self):
        g = torch.ones([self.d_model,], dtype=self.dtype, device=self.device)
        return nn.Parameter(g)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_x = self._rms(x)
        y = einsum(x, 1/rms_x, "... d_model, ... -> ... d_model")
        o = einsum(y, self.weight, "... d_model, d_model -> ... d_model").to(in_dtype)
        return o
    
    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        varience = (1/self.d_model) * torch.sum(torch.square(x), dim=-1) + self.eps
        return torch.sqrt(varience)


def SiLU(x:torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x) # tensor之间用 * 代表逐元素乘法

def ReLU(x:torch.Tensor) -> torch.Tensor:
    return torch.max(x, torch.tensor(0))
        
class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self._check_dimension()
        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w2 = Linear(in_features=d_ff, out_features=d_model)
        self.w3 = Linear(in_features=d_model, out_features=d_ff)

    
    def _check_dimension(self):
        if self.d_ff / self.d_model != 8/3:
            warnings.warn(f"You should set d_ff toapproximately 8/3 × d_model in your implementation, now {self.d_ff/self.d_model:.2f}")
        
        if self.d_ff % 64 != 0:
            warnings.warn(f"You should ensure that the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your hardware, now {self.d_ff}")

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.w2(SiLU(self.w1(x)) * self.w3(x))


class RoPE_llama(nn.Module):
    """llama风格的RoPE实现"""
    def __init__(self, theta: float, d_k:int, max_seq_len:int|None=None, device=None, dtype=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.device = device 
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        if max_seq_len is not None:
            self._init_weight(max_seq_len)

    def _init_weight(self, max_seq_len):
        self.inv_freq = torch.tensor(
                [1 / (self.theta ** (2 * k / self.d_k)) for k in range(self.d_k // 2 )],
                device=self.device, dtype=self.dtype
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        position_ids = torch.arange(self.max_seq_len, device=self.device, dtype=self.dtype)
        #广播，每个位置的token都拿到一个d_k/2长度的高低频旋转角度
        freqs = einsum(position_ids, self.inv_freq, "... seq_len, half_dk -> ... seq_len half_dk")
        emb = torch.cat((freqs,freqs), dim=-1) # 在最后一个维度拼接，[... seq_len, d_k]
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, x: Float[torch.Tensor, " ... seq_len d_k"], token_positions: Float[torch.Tensor, " ... seq_len"]):
        """
        这是llama系列模型（例如qwen、internvl）的使用rope的方法，比起reformer的原文有一点改动。
        原文是相邻的两个隐藏维度构成的平面上旋转
        实际应用上，会把第i个维度和第i+d//2个维度一起旋转，所以拿到freq了以后直接在-1的维度拼接在一起，让x_i和x_{i+d//2}共用一个旋转角度\theta
        所以，使用cos+sin的时候只要执行下面定义的很简单的rotate_half就可以简便的完成了。
        
        可以这么做的理由是，拓展到高维空间的时候，任意两个平面旋转都可以。
        这个做法和rope原文的区别就是对于旋转某个任意位置的token的时候，旋转这个token在R^{d_k}空间下的表征的时候
        把原来[x0,x1,x2,x3,x4,x5]变成
        """
        # 复用/更新cache
        max_seq_len = token_positions.shape[0]
        if self.max_seq_len is None or max_seq_len > self.max_seq_len:
            self._init_weight(max_seq_len)

        # 这里前面的...是由token_positions引入的，d_k前面的维度都遵循token_positions
        cos: Float[torch.Tensor, "... seq_len d_k"] = self.cos_cache[token_positions]
        sin: Float[torch.Tensor, "... seq_len d_k"] = self.sin_cache[token_positions]

        embed_cos = einsum(x, cos, "... seq_len d_k, ... seq_len d_k -> ... seq_len d_k")
        embed_sin = einsum(self.rotate_half(x), sin, "... seq_len d_k, ... seq_len d_k -> ... seq_len d_k")
        return embed_cos + embed_sin

    def rotate_half(x):
        x1 = x[..., : x.shape[0]//2]
        x2 = x[..., x.shape[0]//2 :]
        return torch.cat((-x2, x1), dim=-1)



class RoPE(nn.Module):
    """RoFormer的标准RoPE实现"""
    def __init__(self, theta: float, d_k:int, max_seq_len:int|None=None, device=None, dtype=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.device = device 
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        if max_seq_len is not None:
            self._init_weight(max_seq_len)

    def _init_weight(self, max_seq_len):
        """构建初始的[max_seq_len,d_k//2]的旋转矩阵，后面做cos和sin"""
        inv_freq = torch.tensor(
            [1 / self.theta**(2 * k / self.d_k) for k in range(self.d_k//2)],
            device=self.device, dtype=self.dtype
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        position_ids = torch.arange(max_seq_len)
        freqs = einsum(position_ids, inv_freq, "seq_len , half_dk ->  seq_len half_dk")
        emb = repeat(freqs, "seq_len half_dk -> seq_len (half_dk repeat)", repeat=2)
        cos_values = emb.cos()
        sin_values = emb.sin()
        self.register_buffer("cos_cache", cos_values, persistent=False)
        self.register_buffer("sin_cache", sin_values, persistent=False)

    def forward(self, x: Float[torch.Tensor, " ... seq_len d_k"], token_positions: Float[torch.Tensor, " ... seq_len"]):
        max_seq_len = token_positions.shape[-1]
        if self.max_seq_len is None or max_seq_len > self.max_seq_len:
            self._init_weight(max_seq_len)
        
        # 这里前面的...是由token_positions引入的，d_k前面的维度都遵循token_positions
        cos: Float[torch.Tensor, "... seq_len d_k"] = self.cos_cache[token_positions]
        sin: Float[torch.Tensor, "... seq_len d_k"] = self.sin_cache[token_positions]

        even_x = x[..., 0::2]
        odd_x = x[..., 1::2]
        
        # 逐元素乘法
        embed_cos = einsum(x, cos, "... seq_len d_k, ... seq_len d_k -> ... seq_len d_k")
        embed_odd_sin = einsum(-x[..., 1::2], sin[...,1::2], "... seq_len half_dk, ... seq_len half_dk -> ... seq_len half_dk")
        embed_even_sin = einsum(x[..., 0::2], sin[...,0::2], "... seq_len half_dk, ... seq_len half_dk -> ... seq_len half_dk")
        embed_sin = rearrange(
            torch.stack([embed_odd_sin,embed_even_sin], dim =-1),
            "... d_k two -> ... (d_k two)"
            )
        return embed_cos + embed_sin


def softmax(x:torch.Tensor, dim:int) -> torch.Tensor:
    max_x = x.max(dim=dim).values.unsqueeze(dim).expand(x.shape)
    scaled_x = x - max_x
    exp_x = scaled_x.exp()
    sum_exp_x = exp_x.sum(dim=dim).unsqueeze(dim).expand(x.shape)
    return exp_x/sum_exp_x

def dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None) -> torch.Tensor:
    d_k = Q.shape[-1]
    attention_scores = einsum(Q,K, "... num_queries d_k , ... num_key_values d_k -> ... num_queries num_key_values")
    scaled_attention_scores = attention_scores/ d_k**0.5
    if mask is not None:
        scaled_attention_scores[~mask] = scaled_attention_scores[~mask] - 10e6
    attention_weight = softmax(scaled_attention_scores, -1)
    O = einsum(attention_weight, V, "... num_queries num_key_values , ... num_key_values d_v -> ... num_queries d_v")
    return O

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, theta:int=10000, max_seq_len:int=512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        intermediate_dim = self.num_heads * self.head_dim
        self.q_proj = Linear(d_model, intermediate_dim)
        self.k_proj = Linear(d_model, intermediate_dim)
        self.v_proj = Linear(d_model, intermediate_dim)
        self.output_proj = Linear(intermediate_dim, d_model)
        self.rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_model"], token_positions: Int[Tensor, " ... sequence_length"] | None = None):
        seq_len = x.shape[-2]

        query: Float[torch.Tensor, "... seq_len intermediate_dim"] = self.q_proj(x)
        key: Float[torch.Tensor, "... seq_len intermediate_dim"] = self.k_proj(x)
        value: Float[torch.Tensor, "... seq_len intermediate_dim"] = self.v_proj(x)

        Q = rearrange(query, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", 
            num_heads=self.num_heads, head_dim=self.head_dim)
        K = rearrange(key, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", 
            num_heads=self.num_heads, head_dim=self.head_dim)
        V = rearrange(value, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", 
            num_heads=self.num_heads, head_dim=self.head_dim)

        mask_size = (*Q.shape[:-1], seq_len)
        mask = (1-torch.triu(torch.ones(size=mask_size), diagonal=1)).bool()

        if token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        O = dot_product_attention(Q, K, V, mask)
        O = rearrange(O, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")
        output = self.output_proj(O)
        return output



def test_linear_layer():
    linear_layer = Linear(3,5)
    print(linear_layer.state_dict())
    x = torch.randn(2,3)
    y = linear_layer(x)
    print(y.shape)
    print(y)

def test_embedding_layer():
    x = torch.LongTensor([[1,2,3], [4,5,6]])
    embedding = Embedding(10,10)
    embeded_x = embedding(x)
    print(embeded_x.shape)

def test_rms_norm():
    x = torch.tensor([[1,2,3],[2,3,4],[3,4,5]], dtype=torch.float32)
    norm_layer = RMSNorm(3)
    x_norm = norm_layer(x)
    print(x_norm.shape)
    print(x_norm)

def test_swiglu():
    x = torch.tensor([[1,2,3],[2,3,4],[3,4,5]], dtype=torch.float32)
    model = SwiGLU(3,8)
    output = model(x)
    print(output.shape)
    print(output)
    print(model.state_dict())

def test_mha():
    x = torch.randn(size=[3,5,8])
    model = CausalMutiHeadAttention(d_model=8,num_heads=2,apply_rope=False)
    o = model(x)
    print(o.shape)
    print(model.state_dict())



if __name__ == "__main__":
    # test_linear_layer()
    # test_embedding_layer()
    test_rms_norm()
    # test_swiglu()
    # test_mha()


    
