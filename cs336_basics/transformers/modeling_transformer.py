import torch
from einops import einsum, rearrange, repeat
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
import warnings
from cs336_basics.transformers import basic_func


class MLP(nn.Module):
    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.linear1 = basic_func.Linear(d_model, d_ff)
        self.linear2 = basic_func.Linear(d_ff, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(basic_func.ReLU(self.linear1(x)))

        
class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, theta:int, max_seq_len:int):
        super().__init__()
        self.ffn = basic_func.SwiGLU(d_model=d_model, d_ff=d_ff)
        self.attn = basic_func.CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len)
        self.ln1 = basic_func.RMSNorm(d_model=d_model)
        self.ln2 = basic_func.RMSNorm(d_model=d_model)
    
    def forward(self, 
            x: Float[Tensor, " batch sequence_length d_model"],
            token_positions: Float[Tensor, " batch sequence_length"]
        ) -> torch.Tensor:
        y = x + self.attn(self.ln1(x), token_positions)
        z = y + self.ffn(self.ln2(y))
        return z
        

class TransformerLM(nn.Module):
    def __init__(self, vocab_size:int, context_length:int, num_layers:int, d_model:int, num_heads:int, d_ff:int, rope_theta:int):
        super().__init__()
        self.token_embeddings = basic_func.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model,num_heads=num_heads,d_ff=d_ff,theta=rope_theta,max_seq_len=context_length) \
                for _ in range(num_layers)
                ]
            )
        self.ln_final = basic_func.RMSNorm(d_model=d_model)
        self.lm_head = basic_func.Linear(d_model, vocab_size)
    
    def forward(self, 
                in_indices: Int[Tensor, " batch_size sequence_length"],
                token_positions: Int[Tensor, " batch_size sequence_length"]
            )-> Float[Tensor, " batch_size sequence_length vocab_size"]:
        hidden_state = self.token_embeddings(in_indices)
        for layer in self.layers:
            hidden_state = layer(hidden_state, token_positions)
        logits = self.lm_head(self.ln_final(hidden_state))
        return logits

        
        
def test_ffn():
    d_model = 8
    d_ff = 16
    model = FeedForward(d_model=d_model, d_ff=d_ff)       
    x = torch.randn(size = [2,3,d_model])
    y = model(x)
    print(y.shape)
     
if __name__=="__main__":
    test_ffn()
