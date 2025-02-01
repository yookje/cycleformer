import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy



from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
)

#for fuzzy
class EncoderDecoder2(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, encoder_pe, tgt_embed, decoder_pe, generator):
        super(EncoderDecoder2, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder_pe = encoder_pe
        self.decoder_pe = decoder_pe
        self.generator = generator
        
        
    
    def forward(self, src, src_fuzzy, tgt, tgt_mask):
        "Take in and process masked src and target sequences."
        
        return self.decode(self.encode(src, src_fuzzy), src, tgt, tgt_mask, -1)
    

 
    
    def encode(self, src, src_fuzzy):
        """
        src: [B, node_size, 2], no need for src_mask
        """
        src_embeddings = self.src_embed(src_fuzzy)
        
        self.encoder_lut = src_embeddings
        
        if self.encoder_pe is not None:
            src_embeddings = self.encoder_pe(src_embeddings, src)
        
        #modified for aafm
        #self.adj_mat = torch.cdist(src, src)

        res = self.encoder(src_embeddings)
        #self.enc_attention_map = self.encoder.enc_attention_map


        
        return res
    

    def old_decode(self, memory, src_fuzzy, tgt, tgt_mask):
        
        self.memory = memory

        B, N, E = memory.shape
        B, V = tgt.shape

        valid_indices = (tgt != -1)
        device = tgt.device
        batch_indices = torch.arange(B, device = device).unsqueeze(-1).expand_as(tgt) # [B, V]
        sequence_indices = torch.arange(V, device = device).unsqueeze(0).expand_as(tgt) # [B, V]

        tgt_valid = tgt[valid_indices]
        batch_indices_valid = batch_indices[valid_indices]
        sequence_indices_valid = sequence_indices[valid_indices]

        if self.tgt_embed is not None:
            whole_embeddings = self.tgt_embed(src_fuzzy)

            tgt_embeddings = torch.zeros(B, V, E, device = device).to(dtype = whole_embeddings.dtype)
            tgt_embeddings[batch_indices_valid, sequence_indices_valid, :] = whole_embeddings[batch_indices_valid, tgt_valid, :]
        else:
            tgt_embeddings = torch.zeros(B, V, E, device = device).to(dtype = memory.dtype)
            tgt_embeddings[batch_indices_valid, sequence_indices_valid, :] = memory[batch_indices_valid, tgt_valid, :]
        
        self.decoder_lut = tgt_embeddings
        
        tgt_embeddings, _= self.decoder_pe(tgt_embeddings)

        return self.decoder(tgt_embeddings, memory, tgt_mask)
    
    
    def decode(self, memory, src_fuzzy, tgt, tgt_mask, decoding_idx):
        
        self.memory = memory
        
        B, N, E = memory.shape
        B, V = tgt.shape

        valid_indices = (tgt != -1)
        device = tgt.device
        batch_indices = torch.arange(B, device = device).unsqueeze(-1).expand_as(tgt) # [B, V]
        sequence_indices = torch.arange(V, device = device).unsqueeze(0).expand_as(tgt) # [B, V]

        tgt_valid = tgt[valid_indices]
        batch_indices_valid = batch_indices[valid_indices]
        sequence_indices_valid = sequence_indices[valid_indices]

        if self.tgt_embed is not None:
            whole_embeddings = self.tgt_embed(src_fuzzy)

            tgt_embeddings = torch.zeros(B, V, E, device = device).to(dtype = whole_embeddings.dtype)
            tgt_embeddings[batch_indices_valid, sequence_indices_valid, :] = whole_embeddings[batch_indices_valid, tgt_valid, :]
        else:
            tgt_embeddings = torch.zeros(B, V, E, device = device).to(dtype = memory.dtype)
            tgt_embeddings[batch_indices_valid, sequence_indices_valid, :] = memory[batch_indices_valid, tgt_valid, :]
        
        self.decoder_lut = tgt_embeddings
        
       
        

        #modified
        tgt_embeddings, _= self.decoder_pe(tgt_embeddings)
        __, circular_pe = self.decoder_pe(memory)

        
        #print("\n memory size encoder output ", memory.size())
        
        
        circular_pe = circular_pe[:,:,:].repeat(B, 1, 1)
        
        
        if decoding_idx==-1: #train phase

            #ith_cpe = circular_pe[:,-1,:].unsqueeze(1).repeat(1, N, 1)
            #ith_cpe[batch_indices_valid, tgt_valid, :] = circular_pe[batch_indices_valid, sequence_indices_valid, :]
           
            ith_cpe = torch.zeros_like(circular_pe).to(dtype = memory.dtype).requires_grad_(False)
            #memory_batch_indices = torch.arange(B, device = device).unsqueeze(-1).repeat(1, N).view(-1)  #[B*N]
            #memory_indices = torch.arange(N, device = device).unsqueeze(0).repeat(B, 1).view(-1)  #[B*N]
            #tour_indicies = tsp_tours.view(-1) 
            #ith_cpe[memory_batch_indices,tour_indicies , : ]= circular_pe[memory_batch_indices, memory_indices, :]   
            self.generator.got_cpe(ith_cpe) # using memory for last ffn
            """
            ith_cpe = torch.zeros_like(circular_pe).to(dtype = memory.dtype)
            memory_batch_indices = torch.arange(B, device = device).unsqueeze(-1).repeat(1, N).view(-1)  #[B*N]
            memory_indices = torch.arange(N, device = device).unsqueeze(0).repeat(B, 1).view(-1)  #[B*N]

    
            # 각 행에 대해 누락된 값을 계산
            mask = torch.ones((B, N), dtype=torch.bool)  # 모든 값 포함된 초기 마스크
            mask[torch.arange(B).unsqueeze(1), tgt] = False  # 각 tgt 값은 False로 설정
            
            missing_values = torch.nonzero(mask, as_tuple=True)[1].view(B, -1).to(device=tgt.device)  # [B, 1] 크기
            tgt_full = torch.cat((tgt, missing_values), dim=1) 
            
            tgt_full_indicies = tgt_full.view(-1) 
           
            ith_cpe[memory_batch_indices,tgt_full_indicies , : ]= circular_pe[memory_batch_indices, memory_indices, :]   
                        
            #ith_cpe[memory_batch_indices,memory_indices, : ]= circular_pe[memory_batch_indices, tgt_full_indicies , :]  
         
            print("\n tgt", tgt[1,:5])
            print("\n circular  cpe", circular_pe[1,:3,:3])
            print("\n ith c pe", ith_cpe[0,0,:3], ith_cpe[0,7,:3], ith_cpe[0,15,:3], ith_cpe[])
            self.generator.got_cpe(ith_cpe) # using memory for last ffn
            """

            
        else: #inference
            
            ith_cpe = circular_pe[:,decoding_idx+1,:].unsqueeze(1).repeat(1, N, 1)
            ith_cpe[batch_indices_valid, tgt_valid, :] = circular_pe[batch_indices_valid, sequence_indices_valid, :]
            
            self.generator.got_cpe(ith_cpe) #using memory

            

        

        return self.decoder(tgt_embeddings, memory, tgt_mask)
        
        #modified for aafm
        #return self.decoder(tgt_embeddings, memory, tgt_mask, self.adj_mat)
    
class Generator(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(Generator, self).__init__()
        
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def got_cpe(self, cpe):
        self.cpe = cpe
    
  
    
    def got_dist(self, dist):
        self.dist_mat = dist
        
    
    def forward(self, x, visited_mask, comparison_matrix):
        #comparison_matrix = comparison_matrix + self.cpe

        
        #modified
        self.norm = nn.LayerNorm(x.size()[-1]).cuda() 
        #print("\n comp mat size ", comparison_matrix.size(), self.cpe.size())
        comparison_matrix = comparison_matrix + self.cpe
        comparison_matrix =  self.norm(comparison_matrix) 

        #case 2
        comparison_matrix =  self.w_2(self.dropout(self.w_1(comparison_matrix).relu()))
        
       
        logits = torch.matmul(x, comparison_matrix.permute(0, 2, 1))


        if visited_mask is not None:
            logits = logits.float()
           
            logits = logits.masked_fill(visited_mask, -1e9)
          
       
        return logits
    
       
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)

        self.norm = nn.LayerNorm(layer.size)

    def old_forward(self, x, adj_mat):
        "Pass the input through each layer in turn."
        #modified for aafm
        for layer in self.layers:
            x = layer(x, adj_mat)
        
        return self.norm(x)
    
    def forward(self, x):
        "Pass the input through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)
    
    def old2_forward(self, x):
        "Pass the input through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        self.enc_attention_map = self.layers[-1].enc_attention_map
        return self.norm(x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def old_forward(self, x, dist_matrix):
        "Follow Figure 1 (left) for connections."
        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, dist_matrix))
        return self.sublayer[1](x, self.feed_forward)

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        #_, self.enc_attention_map = self.self_attn(x, x, x, return_attn_probs = True)
        #print("enc_attention_map", self.enc_attention_map.shape)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        #modified
        self.attn_input = None

    #modified
    def set_self_attn_input(self, self_attn_input):
        self.attn_input = self_attn_input

    def old_forward(self, x, memory, tgt_mask, dist_mat):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask,dist_mat)
        return self.norm(x)
    
    def forward(self, x, memory, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask)
        return self.norm(x)
    
    
    def old2_forward(self, x, memory, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, self.attn_input)
        return self.norm(x)

    

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        


    def old_forward2(self, x, memory, tgt_mask,attn_input=None):
        "Follow Figure 1 (right) for connections."
        m = memory
        #modified
        if attn_input != None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, attn_input, attn_input, tgt_mask))
        else : 
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[2](x, self.feed_forward)
    

    def forward(self, x, memory, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
       
        #modified for remove decoder selfattn
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[2](x, self.feed_forward)
    
    
    def old_forward(self, x, memory, tgt_mask, dist_matrix):
        "Follow Figure 1 (right) for connections."
        
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, dist_matrix, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, dist_matrix, cross=True))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size, device = "cpu"):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape, device = device), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

class AAFM(nn.Module):
    def __init__(self, graph_size, d_model, initial_alpha=1.0):

        super(AAFM, self).__init__()
        
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.graph_size = graph_size

        init_value = torch.tensor(graph_size, dtype=torch.float32)
        self.log2_N = torch.log2(init_value)
        
        self.alpha_raw = nn.Parameter(torch.tensor(torch.exp(torch.tensor(initial_alpha, dtype=torch.float32)-1))) #inversion of soft plus
          

        
    def forward(self, query, key, value, dist=None , mask=None, cross=False ):
        bs, query_size, dim = query.size()
        flag = 1
    
	
        # equation f(N, dist) = -alpha * log2_N * dist
        alpha = torch.nn.functional.softplus(self.alpha_raw) + 1e-6
        
        start_idx = self.graph_size - query_size
        
        attention_mask = mask
        if attention_mask is None:
          
            q = self.fc_q(query)

            if cross==True: #cross attn at decoder
                
                k = self.fc_k(key[:,start_idx:,:])
                v = self.fc_v(value[:,start_idx:,:])

                dist = dist[:,start_idx:,start_idx:]  #decoder 크기 맞추기
               

            else : 
                k = self.fc_k(key)
                v = self.fc_v(value)
                
            f = - alpha * self.log2_N * dist
            
            #print("\n1================= ", q.size(), k.size(), v.size())
        

        else : 
            flag=2
            dist = dist[:,start_idx:,start_idx:] #attention mask와 크기 맞추려고
            f =  - alpha * self.log2_N * dist 
            

            f = f.masked_fill(attention_mask == 0, float('-inf'))

            q = self.fc_q(query)
            k = self.fc_k(key)
            v = self.fc_v(value)
            #print("\n2================= ", dist.size(), q.size(), k.size(), v.size())
        
        
        sigma_Q = torch.sigmoid(q)
        exp_A = torch.exp(f)  # (bs, n, n)
        exp_K = torch.exp(k)  # (bs, n, d_model)

	    # Element-wise multiplication of exp(K) and V
        exp_K_V = exp_K * v  # (bs, n, dim)

	    # Weighted sum: exp(A) @ (exp(K) * V)
        #print("\n a kv", flag, exp_A.size(), exp_K_V.size(), query.size(), key.size(), value.size())
        numerator = torch.matmul(exp_A, exp_K_V)  # (bs, n, d_model)

	    

	    # Normalization term: exp(A) @ exp(K)
        denominator = torch.matmul(exp_A, exp_K)  # (bs, n, d_model)

	    # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        denominator = denominator + epsilon

        # Final attention mechanism output
        output = numerator / denominator  # (bs, n, dim)

        # Element-wise multiplication with sigma(Q)
        output =  sigma_Q * output #(bs, n, d_model)
        del q,k,v

        return output


    
class FlashMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(FlashMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = dropout
        
    def forward(self, query, key, value, mask=None, return_attn_probs = False):
        "Implements Figure 2"
        nbatches = query.size(0)
        bsz, q_len, _ = query.size()

        # 1) Do all the linear projections in batch from d_model => h x d_k
        q, k, v = [    # shape: (b, s, num_heads, head_dim)
            lin(x).view(nbatches, -1, self.h, self.d_k).contiguous()
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # reference1: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py
        # reference2: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama2_flash_attn_monkey_patch.py
        
        attention_mask = mask
        if attention_mask is None:
            if return_attn_probs:
                output, _, attention_map = flash_attn_func(q, k, v, dropout_p=(self.dropout if self.training else 0.0001), softmax_scale=None, causal=False, return_attn_probs = True)
                output = output.view(bsz, q_len, -1)
                print("q, k, v", q.shape, k.shape, v.shape, attention_map.shape)
            else:
                output = flash_attn_func(q, k, v, dropout_p=(self.dropout if self.training else 0.0), softmax_scale=None, causal=False, return_attn_probs = False).view(bsz, q_len, -1)
        else:
            if return_attn_probs:
                output, _, attention_map = flash_attn_func(q, k, v, dropout_p=(self.dropout if self.training else 0.0001), softmax_scale=None, causal=True, return_attn_probs = True)
                output = output.view(bsz, q_len, -1)
            else:
                output = flash_attn_func(q, k, v, dropout_p=(self.dropout if self.training else 0.0), softmax_scale=None, causal=True, return_attn_probs = False).view(bsz, q_len, -1)

        output = output.view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        
        if return_attn_probs:
            return self.linears[-1](output), attention_map
        else:
            return self.linears[-1](output)
        
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

#for fuzzy
class Embeddings2(nn.Module):
    def __init__(self, d_model, feat_dim=128):
        super(Embeddings2, self).__init__()
        self.d_model = d_model
        self.w = nn.Linear(feat_dim, d_model) #112 24
        #self.w = nn.Linear(56, d_model) #8 bits for 1 depth # (6*8)+16 =64 #66
        #self.w2 = nn.Linear(24, d_model) 
        
    def forward(self, x):
        # x: [B, node_size, 2]
        # visited: (only for decoder) [B, node_size]

        device = x.device 
        x = x.to(device, dtype=torch.float32)
            
        return self.w(x).relu()
    

    def old_forward(self, x):
        # x: [B, node_size, 2]
        # visited: (only for decoder) [B, node_size]

        device = x.device 
        x = x.to(device, dtype=torch.float32)
        B, V, E = x.shape

        emb1 = torch.zeros(B, V, E, device = device).to(dtype = x.dtype)
        emb2 = torch.zeros(B, V, E, device = device).to(dtype = x.dtype)
        

        emb1 = x[:,:,:48] #fuzzy encoding
        emb2 = x[:,:,48:] #feat3 encoding

        result1 = self.w(emb1).relu()
        result2 = self.w2(emb2).relu()
        result = result1 + result2

        return result
    
      
 
class PositionalEncoding_2D(nn.Module):
    "Implement the Encoder PE function."
    "Implement of xy_sum"

    def __init__(self, d_model, T, dropout):
        super(PositionalEncoding_2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.d_model = d_model
        self.d_model = self.d_model //2 
            
        # Compute the div_term once.
        div_term = 1 / torch.pow(
            T, (2.0 * (torch.arange(self.d_model))) / torch.tensor(d_model*2)
        )  # [batch_size, node_size, 128]
        self.register_buffer("div_term", div_term)

    def forward(self, embeddings, graph):
        """
        embeddings: [batch_size, node_size, 128]
        graph: [batch_size, node_size, 2]
        """
        batch_size, node_size, _ = graph.shape
        device = graph.device 
        pe_x = torch.zeros(batch_size, node_size, self.d_model, device = device)  # [batch_size, node_size, 128]
        pe_y = torch.zeros(batch_size, node_size, self.d_model, device = device)  # [batch_size, node_size, 128]
        
        xs = graph[:,:,0]
        ys = graph[:,:,1]
        
        """
        b term is y for xy, theta for rad
        a term is x for xy, r for rad
        """
        
        b_term = ys.unsqueeze(-1) * self.div_term.repeat(
            batch_size, node_size, 1
        )  # [batch_size, node_size, 128]
        
        a_term = xs.unsqueeze(-1) * self.div_term.repeat(
            batch_size, node_size, 1
        )  # [batch_size, node_size, 128]
            
        pe_x[:, :, 0::2] = torch.sin(a_term[:, :, 0::2])  # [batch_size, node_size, 32]
        pe_x[:, :, 1::2] = torch.cos(a_term[:, :, 1::2])  # [batch_size, node_size, 32]
        pe_y[:, :, 0::2] = torch.sin(b_term[:, :, 0::2])  # [batch_size, node_size, 32]
        pe_y[:, :, 1::2] = torch.cos(b_term[:, :, 1::2])  # [batch_size, node_size, 32]
        
        
        pe =pe = torch.cat([pe_x, pe_y], -1).requires_grad_(False)
        # [batch_size, node_size, 128]
        
   
        embeddings = embeddings + pe  # [batch_size, node_size, 128]
        self.pe = pe
        return self.dropout(embeddings)  # [batch_size, node_size, 128]

class PositionalEncoding_Circular(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding_Circular, self).__init__()

        self.pattern = self.Cyclic_Positional_Encoding(max_len, d_model)
        self.max_len = max_len
        self.d_model = d_model
        
    def basesin(self, x, T, fai = 0):
        return np.sin(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)
    
    def basecos(self, x, T, fai = 0):
        return np.cos(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)
    
    # implements the CPE
    def Cyclic_Positional_Encoding(self, n_position, emb_dim, mean_pooling = True, target_size=None):
        
        Td_set = np.linspace(np.power(n_position, 1 / (emb_dim // 2)), n_position, emb_dim // 2, dtype = 'int')
        x = np.zeros((n_position, emb_dim))
         
        for i in range(emb_dim):
            Td = Td_set[i //3 * 3 + 1] if  (i //3 * 3 + 1) < (emb_dim // 2) else Td_set[-1]
            fai = 0 if i <= (emb_dim // 2) else  2 * np.pi * ((-i + (emb_dim // 2)) / (emb_dim // 2))
            longer_pattern = np.arange(0, np.ceil((n_position) / Td) * Td, 0.01)
            if i % 2 ==1:
                x[:,i] = self.basecos(longer_pattern, Td, fai)[np.linspace(0, len(longer_pattern), n_position, dtype = 'int', endpoint = False)]
            else:
                x[:,i] = self.basesin(longer_pattern, Td, fai)[np.linspace(0, len(longer_pattern), n_position, dtype = 'int', endpoint = False)]
                
        pattern = torch.from_numpy(x).type(torch.FloatTensor)
        pattern_sum = torch.zeros_like(pattern)
        
        # for generalization (way 2): reuse the wavelength of the original size but make it compatible with the target size (by duplicating or discarding)
        if target_size is not None:
            pattern = pattern[np.ceil(np.linspace(0, n_position-1,target_size))]
            pattern_sum = torch.zeros_like(pattern)
            n_position = target_size
        
        # averaging the adjacient embeddings if needed (optional, almost the same performance)
        arange = torch.arange(n_position)
        pooling = [0] if not mean_pooling else[-2, -1, 0, 1, 2]
        time = 0
        for i in pooling:
            time += 1
            index = (arange + i + n_position) % n_position
            pattern_sum += pattern.gather(0, index.view(-1,1).expand_as(pattern))
        pattern = 1. / time * pattern_sum - pattern.mean(0)
        
        return pattern    

        
    def forward(self, x):
       

        CPE_embedding = self.pattern.to(x.device)
        #print("\n x cpe", x.size(), CPE_embedding.size())
        CPE_embedding = CPE_embedding[:x.size(1),:]
        CPE_embedding = CPE_embedding.unsqueeze(0)
    
        #print("\n x cpe2", x.size(), CPE_embedding.size())
      
        pe =  x+CPE_embedding
    
        return  pe, CPE_embedding

    #modified
    """
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x), self.pe[:, : x.size(1)]
    """
    
class old_PositionalEncoding_Circular(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(old_PositionalEncoding_Circular, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term + 2 * torch.pi * position / max_len)
        pe[:, 1::2] = torch.cos(position * div_term + 2 * torch.pi * position / max_len)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)


    #modified
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        #modified for trainable
        #x = x + self.pe[:, : x.size(1)]
        return self.dropout(x), self.pe[:, : x.size(1)]
    
    def original_forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
class PositionalEncoding_1D(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding_1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

def make_model(
    src_sz, 
    enc_num_layers=6, 
    dec_num_layers=6, #modified
    d_model=128, 
    d_ff=512, 
    h=8, 
    dropout=0.1,
    encoder_pe = "2D",
    decoder_pe = "circular",
    decoder_lut = "memory",
    feat_dim = 128,
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy

    attn = FlashMultiHeadedAttention(h, d_model)

    #modified for aafm
    #attn2 = AAFM(src_sz, d_model)

    #modified for value attn
    #attn_value = FlashMultiHeadedAttention_Value(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    #for fuzzy
    src_embed = Embeddings2(d_model,feat_dim)
    
    if encoder_pe == "2D":
        encoder_pe = PositionalEncoding_2D(d_model, 2, dropout)
    else:
        encoder_pe = None
        
    if decoder_pe == "circular":
        #decoder_pe = old_PositionalEncoding_Circular(d_model, dropout, src_sz)
        decoder_pe = PositionalEncoding_Circular(d_model, dropout, src_sz)
    elif decoder_pe == "1D":
        decoder_pe = PositionalEncoding_1D(d_model, dropout)
    else:
        assert False
    
    if decoder_lut == "shared":
        tgt_embed = src_embed
    elif decoder_lut == "unshared":
        tgt_embed = Embeddings2(d_model)
    elif decoder_lut == "memory":
        tgt_embed = None
    else:
        assert False

    
    #for fuzzy
    model = EncoderDecoder2(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), enc_num_layers),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), dec_num_layers),
        src_embed=src_embed, # encoder
        encoder_pe=encoder_pe,
        tgt_embed=tgt_embed, # decoder
        decoder_pe=decoder_pe,
        generator=Generator(d_model, d_ff, dropout), #modified
    )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
