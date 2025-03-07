
# In[1]:

import logging
import sys
import os
import json
import time
import datetime
import math
import random
import numpy as np
from zoneinfo import ZoneInfo

import sentencepiece as spm


# In[2]:
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.autograd import Function


# In[3]:

logging.basicConfig(level=logging.INFO)
log = logging.Logger("Main")
log.setLevel(logging.INFO)


# ## 1. Project configuration

# In[6]:

project_name='rec_tok'
model_cpu = None
model_name=f'{project_name}_v1'

use_preprocessed_data = True                      # Use already tokenized data
use_existing_model_from_checkpoint = False         # Try to load checkpoint of training
use_torch_compile = True                           # Requires a modern graphics card with torch compile backend support

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("mps") if torch.backends.mps.is_available() else device

model_path = "model"
data_path = "data"


# ##  2.1 Text data
# 

# In[7]:


use_dark_mode=False # Set to false for white background. HTML-text-compare uses background-colorization to identify different sources. Those background colors are dependent on the theme type.


# In[8]:


def get_texts(path:str = "~/BookTextLib/Texts") -> list[str]:
    texts: list[str] = []
    real_path = os.path.expanduser(path)
    if os.path.isdir(real_path) is False:
        print(f"{real_path} is not a directory!")
        return texts
    for root, dirs, files in os.walk(real_path):
        for file in files:
            if file.endswith(".txt"):
                txt_file = os.path.join(root, file)
                with open(txt_file, 'r') as f:
                    txt = f.read()
                    texts.append(txt)
    return texts


# In[9]:


text_list = get_texts()
text_corpus = '\n\n\n'.join(text_list)


# In[10]:

def progress_bar_string(progress, max_progress, bar_length=20):
    """Create a Unicode progress bar string

    This creates a string of length bar_length with a Unicode progress bar using
    fractional Unicode block characters. The returned string is always of constant
    length and is suitable for printing to a terminal or notebook.

    This pretty much obsoletes the `tqdm` or similar package for simple progress bars.

    :param progress: current progress
    :param max_progress: maximum progress
    :param bar_length: length of the progress bar
    :return: Unicode progress bar string of length `bar_length`
    """
    progress_frac = progress / max_progress
    num_blocks = int(bar_length * progress_frac)
    rem = bar_length * progress_frac - num_blocks
    blocks = " ▏▎▍▌▋▊▉█"
    remainder_index = int(rem * len(blocks))
    bar = blocks[-1] * num_blocks
    if remainder_index > 0:
        bar += blocks[remainder_index]
    bar += " " * (bar_length - len(bar))
    return bar

print(f"Number of texts: {len(text_list)}, corpus length in bytes: {len(text_corpus)}")


# In[11]:


text_list[0][:100]


# ## 2.3 Tokenize data

# In[12]:



class SentencePieceBPE:
    def __init__(self, model_name, vocab_size=32768, data_directory='data'):
        self.vocab_size = vocab_size
        self.model_name = model_name
        self.model_path = os.path.join(data_directory, model_name)
        self.temp_file = os.path.join(data_directory, "tok_blob.txt")
        self.corpus_encoded_path = os.path.join(data_path, "corpus_encoded.json")
        self.sp = None
        self.encoded_corpus = []

    def train(self, texts:list[str], verbose=True):
        if verbose:
            print("1/2: Starting tokenizer...")
        blob = '\n'.join(texts)
        with open(self.temp_file, "w", encoding="utf-8") as f:
            f.write(blob)
        spm.SentencePieceTrainer.train(
            input=self.temp_file,
            model_prefix=self.model_path,
            vocab_size=self.vocab_size,
            model_type='bpe',
            character_coverage=1.0,  # Important for multilingual
            normalization_rule_name='identity',  # No normalization
            byte_fallback=True
        )
        print("Tokenizer trained.")
        self.load()
        print("2/2: Encoding entire corpus...")
        self.encoded_corpus = self.encode(blob)
        print("Corpus encoded")
        with open(self.corpus_encoded_path, "w") as f:
            json.dump(self.encoded_corpus, f)

    def load(self):
        # Load the model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{self.model_path}.model")
        try:
            with open(self.corpus_encoded_path, "r") as f:
                self.encoded_corpus = json.load(f)
        except:
            print("Corpus not yet encoded!")

    def get_record_count(self, context_length):
        n = len(self.encoded_corpus) - context_length - 1
        if n <=0:
            return 0
        record_count = int(n)
        return record_count

    def get_random_tokens(self, context_length):
        n = self.get_record_count(context_length)
        if n==0:
            return None
        ind = random.randint(0, n)
        toks = self.encoded_corpus[ind:ind+context_length]
        return toks

    def get_random_token_pair(self, context_length):
        toks = self.get_random_tokens(context_length+1)
        pair = (toks[0:context_length], toks[1:context_length+1])
        return pair

    def encode(self, text:str):
        tokens = self.sp.encode_as_ids(text)
        return tokens

    def visualize_tokens(self, text:str):
        pieces = sp.encode_as_pieces(text)  # For visualization
        return pieces

    def decode(self, ids):
        decoded = self.sp.decode(ids)
        return decoded


# In[ ]:


vocab_size = 32768

tokenizer = SentencePieceBPE(model_name=model_name, vocab_size=vocab_size, data_directory=data_path)
if use_preprocessed_data is True:
    tokenizer.load()
else:
    tokenizer.train(text_list)


# In[ ]:


tok_tests = ["Good morning, this is a simple test sentence for tokenization",
             "Guten Morgen, dies is ein einfach Testsatz zur Aufteilung in Satzbestandteile",
             "སེམས་ཉིད་ངལ་བསོ་རྒྱུད་",
             "སྟོང་ཉིད་སྙིང་རྗེའི་སྙིང་པོ་ཅན།"]
for test in tok_tests:
    enc = tokenizer.encode(test)
    dec = tokenizer.decode(enc)
    if dec != test:
        print(f"Tokenizer failed for: \n{test} len={len(test)}\n{dec} len={len(dec)}")
    else:
        r = len(enc)/len(test)*100.0
        print(f"Tokenizer: {test}({len(test)}) -> {enc}({len(enc)}) OK, compressed size: {r:.2f}%")


# ## 3. Model metadata


# In[ ]:
def save_checkpoint(
    params, model, optimizer, current_epoch, current_loss, file_path, log=None
):
    params["current_epoch"] = current_epoch
    params["current_loss"] = current_loss

    state = {
        "params": params,
        "optimizer_states": optimizer.state_dict(),
    }

    # Really? This fixes the fact that compiled models store their stuff in a _different_ place!
    if hasattr(model, "_orig_mod"):  # means, model was compiled!
        state["model_states"] = model._orig_mod.state_dict()
    else:  # models was not compiled, 'standard' case.
        state["model_states"] = model.state_dict()

    torch.save(state, file_path)
    if log is not None:
        log.info(f"Saved model to {file_path}")


def load_model_metadata_from_checkpoint(
    params, updatable_params, file_path, device=None, log=None
):
    if not os.path.exists(file_path):
        if log is not None:
            log.info(
                f"No saved state, no {file_path}, starting with default state: {params}"
            )
        return params
    with torch.serialization.safe_globals([nn.modules.activation.Mish]):
        if device is None:
            state = torch.load(file_path)
        else:
            state = torch.load(file_path, map_location=device)
    new_params = state["params"]
    del state
    # if metadata_compatible(params, new_params, updatable_params, log) is False:
    #     if log is not None:
    #         log.info(f"Metadata incompatible, starting with default state: {params}")
    #     return params
    for key in updatable_params:
        if key in params:
            new_params[key] = params[key]
    if log is not None:
        log.info(f"Loaded model metadata from {file_path}, {new_params}")
    return new_params


def load_checkpoint(
    params, model, optimizer, file_path, updatable_keys, device=None, log=None
):
    if not os.path.exists(file_path):
        print(f"No saved state, no {file_path}, starting from scratch.")
        if log is not None:
            log.info(
                f"No saved state, no {file_path}, starting new model from scratch with default params {params}."
            )
        return None

    with torch.serialization.safe_globals([nn.modules.activation.Mish]):
        if device is None:
            state = torch.load(file_path, weights_only=True)
        else:
            state = torch.load(file_path, map_location=device, weights_only=True)
    params_new = state["params"]
    # if metadata_compatible(params, params_new, updatable_keys, log) is False:
    #     print("Metadata incompatible, starting from scratch.")
    #     del state  # Free memory
    #     if log is not None:
    #         log.info(
    #             f"Metadata incompatible, starting new model with default params {params}."
    #         )
    #     return params
    params_old = params
    params = params_new
    model.load_state_dict(state["model_states"])
    optimizer.load_state_dict(state["optimizer_states"])
    for g in optimizer.param_groups:  # Allow for different learning rates
        g["lr"] = params_old["learning_rate"]
    for key in updatable_keys:
        params[key] = params_old[key]
    epoch = params["current_epoch"]
    loss = params["current_loss"]
    print(
        f"Continuing from saved state epoch={epoch+1}, loss={loss:.3f}"
    )  # Save is not necessarily on epoch boundary, so that's approx.
    del state  # Free memory
    if log is not None:
        log.info(f"Continuing from saved state epoch={epoch+1}, loss={loss:.3f}")
    return params


# In[ ]:


params = None
updatable_keys=['learning_rate', 'batch_size', 'current_epoch', 'current_loss',
                 'sample_every_n_iterations', 'sample_size', 'save_every_n_iterations', 'max_iterations']
model_dimension = 512
context_length = 128

params = { # Multi-head self-attention
        'meta_name_template': '{prelude_layers}-{recurrent_layers}/{recurrence_steps}-{coda_layers}x{heads}x{units}x{vocab_size}',

        'prelude_layers': 8,
        'recurrent_layer_blocks': 0,
        'coda_layers': 4,
        'recurrent_layers': 0,
        'recurrence_steps': 1,
        'heads': 16,
        'vocab_size': vocab_size,
        'context_length': context_length,
        'min_dropout': 0.1,  # first layer of prelude, last layer of coda
        'max_dropout': 0.2,  # last layer of prelude, first layer of coda
        'mid_dropout': 0.1,  # Used by recurrence
        'weight_decay': 1e-3,  # L2 regularization, applied by Adam optimizer
        'non_linearity': nn.Mish,  # CriticalModule.CriticalActivationLayer,  # Default nn.ReLU
        'use_critical': False,  # Add CriticalActivationLayer before recurrent_layer
        'model_dimension': model_dimension,
        'test_iterations': 100,  # number of iterations for loss estimation

        'batch_size': 64,
    
        'learning_rate': 4e-4,  # Only used, if lr_schedule is False
        'lr_schedule': True,
        'lr_min': 5e-5,
        'lr_max': 3e-4,
        'warmup': 4000,
        'decay': 50000,
    
        'grad_clip': 0.8,

        'sample_every_n_iterations': 1024,
        'sample_size': 128,
        'save_every_n_iterations': 8192,

        'max_iterations': 100000000  # maximum number of training iterations
    }

model_file_path = os.path.join(model_path, model_name+".pt")
if use_existing_model_from_checkpoint is True:
    params = load_model_metadata_from_checkpoint(params, updatable_keys, model_file_path, device=device, log=log) # torch.device('cpu'))
if params == None or use_existing_model_from_checkpoint is False:
    use_existing_model_from_checkpoint = False

num_batches = tokenizer.get_record_count(params['context_length']) // params['batch_size']
print(f"Batches: {num_batches}")


# ## 4. Batch handling

# In[ ]:


def get_sample_batch(batch_size):
    for i in range(batch_size):
        Xi, yi = tokenizer.get_random_token_pair(params['context_length'])
        if i==0:
            # smpX=np.array(Xi, dtype=np.float32)
            smpX=np.array(Xi, dtype=np.int32)
            smpy=np.array(yi, dtype=np.int32)
        else:
            # smpX = np.vstack((smpX, np.array(Xi, dtype=np.float32)))
            smpX = np.vstack((smpX, np.array(Xi, dtype=np.int32)))
            smpy = np.vstack((smpy, np.array(yi, dtype=np.int32)))
    return np.array(smpX), np.array(smpy)


# In[ ]:


x, y = get_sample_batch(2)
x.shape, y.shape


# In[ ]:


sample_data = None

def get_torch_batch(batch_size, device, split=None):
    x, y = get_sample_batch(batch_size)
    tx = torch.tensor(x, dtype=torch.long).to(device)
    tx.requires_grad = False
    ty = torch.tensor(y, dtype=torch.long).to(device)
    ty.requires_grad = False
    return tx, ty

def get_zero_state(batch_size, context_length, hidden_size, device):
    zstate = torch.zeros(batch_size, context_length, hidden_size, device=device)
    zstate.requires_grad = False
    return zstate


# ## 5. Loss and training helpers

# In[ ]:


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, max_len=5000):
        super().__init__()
        # Precompute positional encodings
        pe = torch.zeros(max_len, model_dimension)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dimension, 2).float() * (-math.log(10000.0) / model_dimension))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [1, max_len, model_dimension]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, model_dimension]
        seq_len = x.size(0)
        pe = self.pe[:, :seq_len, :].expand(-1, x.size(1), -1)
        return x


# In[ ]:
class RecurrentMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(RecurrentMultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim  # d_model
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads  # dimension per head
        self.dropout = nn.Dropout(dropout)
        
        # Linear projections for Q, V, R (K is replaced by H computed from R)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.r_linear = nn.Linear(embed_dim, embed_dim)
        
        # Recurrent weight matrix W_h for each head, shape (num_heads, d_k, d_k)
        self.W_h = nn.Parameter(torch.randn(num_heads, self.d_k, self.d_k))
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = 1 / math.sqrt(self.d_k)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.r_linear.weight)
        nn.init.xavier_uniform_(self.W_h)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.zeros_(self.v_linear.bias)
        nn.init.zeros_(self.r_linear.bias)
        
    def forward(self, query, key, value, R, attn_mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model) - unused but kept for API compatibility
            value: (batch_size, seq_len_k, d_model)
            R: (batch_size, seq_len_k, d_model) - new recurrent input
            attn_mask: (seq_len_q, seq_len_k) or (batch_size * num_heads, seq_len_q, seq_len_k)
        
        Returns:
            output: (batch_size, seq_len_q, d_model)
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = value.size(1)
        
        # Project Q, V, R
        Q = self.q_linear(query)  # (batch_size, seq_len_q, d_model)
        V = self.v_linear(value)  # (batch_size, seq_len_k, d_model)
        R_proj = self.r_linear(R)  # (batch_size, seq_len_k, d_model)
        
        # Reshape for multi-head: (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        R_proj = R_proj.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        # Shapes: (batch_size, num_heads, seq_len, d_k)
        
        # Compute recurrent hidden states H from R_proj
        H_list = []
        h_t = torch.zeros(batch_size, self.num_heads, self.d_k, device=R.device)  # Initial hidden state
        for t in range(seq_len_k):
            R_t = R_proj[:, :, t, :]  # (batch_size, num_heads, d_k)
            # Recurrence: h_t = tanh(W_h h_{t-1} + R_t)
            h_t = torch.tanh(torch.einsum('hnk,bhn->bhn', self.W_h, h_t) + R_t)
            H_list.append(h_t)
        
        # Stack to form H: (batch_size, num_heads, seq_len_k, d_k)
        H = torch.stack(H_list, dim=2)
        
        # Compute attention scores: Q H^T
        scores = torch.matmul(Q, H.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            # print(f"scores: {scores.shape}, attn_mask: {attn_mask.shape}")
            scores = scores + attn_mask
        
        # Softmax and compute output
        attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len_q, d_k)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        output = self.out_proj(attn_output)  # (batch_size, seq_len_q, d_model)
        
        return output


class TransformerBlock(nn.Module):
    def __init__(self, model_dimension, heads, projection_dimension, dropout=0.1, non_linearity=nn.ReLU, recurrent_att=True):
        super(TransformerBlock, self).__init__()
        self.recurrent_att = recurrent_att
        if recurrent_att is True:
            self.self_attn = RecurrentMultiheadAttention(model_dimension, heads, dropout=dropout)
        else:
            self.self_attn = nn.MultiheadAttention(model_dimension, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(model_dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(model_dimension, projection_dimension),
            non_linearity(),
            nn.Dropout(dropout),
            nn.Linear(projection_dimension, model_dimension)
        )
        self.norm2 = nn.LayerNorm(model_dimension)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        if attn_mask is None and x.size(0) > 1:
            seq_len = x.size(0)
            attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            attn_mask = attn_mask.to(x.device)  # [seq_len, seq_len], upper triangle = True (masked)

        if self.recurrent_att is True:
            xt = x.transpose(0, 1)  # wants batch first
            attn_output = self.self_attn(xt, xt, xt, xt, attn_mask=attn_mask)
            attn_output = attn_output.transpose(0, 1) # no longer batch first
        else:
            attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class LatentRecurrentBlock(nn.Module):
    def __init__(self, model_dimension, heads, projection_dimension, recurrent_layers=1, recurrence_steps=3, dropout=0.1, non_linearity=nn.ReLU):
        super(LatentRecurrentBlock, self).__init__()
        self.recurrence_steps = recurrence_steps
        self.self_attn = nn.MultiheadAttention(model_dimension, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(model_dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.recurrent_layers = recurrent_layers
        self.recurrent = nn.LSTM(  # Swap GRU for LSTM
            input_size=model_dimension,
            hidden_size=model_dimension,
            num_layers=recurrent_layers,
            batch_first=True,
            bidirectional=False
        )
        self.ff = nn.Sequential(
            nn.Linear(model_dimension, projection_dimension),
            non_linearity(),
            nn.Dropout(dropout),
            nn.Linear(projection_dimension, model_dimension)
        )
        self.norm2 = nn.LayerNorm(model_dimension)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_output, _ = self.self_attn(x, x, x,
                                      attn_mask=attn_mask,
                                      key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        residual = x
        batch_size = x.size(1)
        latent = x.transpose(0, 1).contiguous()  # [batch, seq_len, model_dimension]
        latent = latent.view(batch_size * x.size(0), 1, x.size(2))  # [batch*seq, 1, model_dimension]
        h0 = torch.zeros(self.recurrent_layers, latent.size(0), x.size(2), device=x.device)
        c0 = torch.zeros(self.recurrent_layers, latent.size(0), x.size(2), device=x.device)  # Add cell state
        for _ in range(self.recurrence_steps):
            latent, (h0, c0) = self.recurrent(latent, (h0, c0))  # LSTM outputs hidden + cell
        latent = latent.view(x.size(1), x.size(0), -1).transpose(0, 1)
        latent = residual + latent
        ff_output = self.ff(latent)
        output = self.norm2(latent + self.dropout2(ff_output))
        return output

class LatentRecurrentDepthModel(nn.Module):
    def __init__(self, vocab_size, model_dimension, heads, context_length, projection_dimension,
                 n1_prelude, n2_recurrent, n3_coda, recurrent_layers=1, recurrence_steps=3, min_dropout=0.1, mid_dropout=0.2, max_dropout=0.1, non_linearity=nn.ReLU, use_critical=False):
        """
        Args:
            vocab_size (int): Size of the vocabulary (for embedding and projection).
            model_dimension (int): Transformer hidden size.
            heads (int): Number of attention heads.
            projection_dimension (int): Feedforward hidden size.
            n1_prelude, n2_recurrent, n3_coda (int): Number of blocks per stage.
            recurrence_steps (int): Recurrent steps per LRD block.
            dropout (float): Dropout rate.
        """
        super(LatentRecurrentDepthModel, self).__init__()

        self.context_length = context_length  # for generate
        self.model_dimension = model_dimension

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, model_dimension)
        self.pos_encoding = PositionalEncoding(model_dimension, max_len=context_length)
        # self.pos_encoding = PositionalEncoding(model_dimension) # , max_len=context_length)

        # Prelude blocks
        tr_list = []
        for i in range(n1_prelude):
            if n1_prelude > 1:
                drop = min_dropout + (max_dropout - min_dropout)*(i/(n1_prelude-1))
            else:
                drop = (min_dropout + max_dropout) / 2
            tr_list.append(TransformerBlock(model_dimension, heads, projection_dimension, drop, non_linearity))
 
        self.prelude = nn.ModuleList(tr_list)

        if use_critical is True:
            self.critical = CriticalModule.CriticalActivationLayer(model_dimension)
        else:
            self.critical = None

        # Latent Recurrent blocks
        if n2_recurrent > 0:
            self.recurrent = nn.ModuleList([
                LatentRecurrentBlock(model_dimension, heads, projection_dimension, recurrent_layers, recurrence_steps, mid_dropout, non_linearity)
                for _ in range(n2_recurrent)
            ])
        else:
            self.recurrent = None

        # Coda blocks
        cd_list = []
        for i in range(n3_coda):
            if n3_coda > 1:
                drop = max_dropout - (max_dropout - min_dropout)*(i/(n3_coda-1))
            else:
                drop = (min_dropout + max_dropout) / 2
            cd_list.append(TransformerBlock(model_dimension, heads, projection_dimension, drop, non_linearity))
        self.coda = nn.ModuleList(cd_list)

        # Final projection layer (e.g., to vocab size for generation)
        self.proj = nn.Linear(model_dimension, vocab_size)

    def forward(self, input_ids, attn_mask=None, key_padding_mask=None):
        """
        Args:
            input_ids (torch.Tensor): Token IDs [batch_size, seq_len].
            attn_mask (torch.Tensor, optional): Attention mask [seq_len, seq_len].
            key_padding_mask (torch.Tensor, optional): Padding mask [batch_size, seq_len].
        Returns:
            torch.Tensor: Output logits [batch_size, seq_len, vocab_size].
        """
        # Embed input tokens
        x = self.embedding(input_ids) * math.sqrt(self.model_dimension) # /2.0  # [batch_size, seq_len, model_dimension]
        # x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [seq_len, batch_size, model_dimension] for transformer
        x = self.pos_encoding(x)

        # Prelude: Entry to latent space
        for block in self.prelude:
            x = block(x, attn_mask, key_padding_mask)

        # Critical function
        if self.critical is not None:
            x = self.critical(x)

        # Recurrent: Refine latents
        if self.recurrent is not None:
            for block in self.recurrent:
                x = block(x, attn_mask, key_padding_mask)

        # Coda: Exit from latent space
        for block in self.coda:
            x = block(x, attn_mask, key_padding_mask)

        # Project to output space
        x = x.transpose(0, 1)  # [batch_size, seq_len, model_dimension]
        output = self.proj(x)  # [batch_size, seq_len, vocab_size]
        return output

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens given a context

        Note: for apple MPS, top_k is limited max 16 vor older torchs! ((01/2023) implementation limitation)
        See: https://github.com/pytorch/pytorch/issues/78915
        Solved in: https://github.com/pytorch/pytorch/pull/94639 (03/2023)

        :param idx: the context (B,T) tensor of indices
        :param max_new_tokens: the maximum number of tokens to generate
        :param temperature: the temperature to use for sampling
        :param top_k: the number of top tokens to consider
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last context_length tokens
            idx_cond = idx[:, -self.context_length :]
            # print(idx_cond.shape)
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply temperature
            if temperature != 1.0 and temperature > 0.0:
                logits = logits / temperature
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    def generate_with_beam(self, model, tokenizer, prompt="The", max_len=50, temperature=1.0, top_k=30, beam_width=3):
    # def generate(model, tokenizer, prompt="The", max_len=50, temperature=1.0, top_k=30, beam_width=3):
        """
        Beam search generation with static abort condition.

        Args:
            model: LatentRecurrentDepthModel
            tokenizer: Your custom/botok tokenizer (no [EOS])
            prompt (str): Starting text
            max_len (int): Max output length
            temperature (float): Softmax temperature
            top_k (int): Sample from top k tokens
            beam_width (int): Number of beams
        """
        model.eval()
        device = next(model.parameters()).device
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)  # [1, seq_len]
        beams = [(input_ids, 0.0)]  # (sequence, log_prob)

        with torch.no_grad():
            for step in range(max_len):
                new_beams = []
                for seq, score in beams:
                    # Forward pass
                    logits = model(seq)  # [1, seq_len, vocab_size]
                    next_logits = logits[0, -1, :] / temperature

                    # Top-k sampling
                    top_k_logits, top_k_indices = torch.topk(next_logits, top_k)

                    # Repetition penality
                    for i, token in enumerate(seq[0][-5:]):
                        penalty = 1.0 + 0.2 * i
                        top_k_logits[top_k_indices == token] /= penalty

                    probs = F.softmax(top_k_logits, dim=-1)

                    # Sample beam_width candidates
                    next_tokens = torch.multinomial(probs, num_samples=beam_width)
                    for i in range(beam_width):
                        token_id = top_k_indices[next_tokens[i]].unsqueeze(0).unsqueeze(0)  # [1, 1]
                        log_prob = torch.log(probs[next_tokens[i]]).item()
                        new_seq = torch.cat([seq, token_id], dim=1)
                        # Repetition penalty
                        # penalty = 1.0 if new_seq[0, -1].item() not in new_seq[0, -5:-1] else 0.9
                        new_beams.append((new_seq, score + log_prob * penalty))

                # Sort and prune beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

                # Static abort: all beams at max_len or repeating last 5 tokens
                # all_max_len = all(len(seq[0]) >= max_len for seq, _ in beams)
                # all_repeating = all(
                #     len(seq[0]) > 5 and seq[0, -5:].tolist() == [seq[0, -1].item()] * 5
                #     for seq, _ in beams
                # )
                # if all_max_len or all_repeating:
                #     break
                if all(len(seq[0]) >= max_len for seq, _ in beams):
                    break

        best_seq, _ = beams[0]
        return tokenizer.decode(best_seq[0].tolist())[1:]  # XXX hack to remove leading space from huggingface decoder!


# In[ ]:


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.GRU, nn.LSTM)):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)


# In[ ]:


print("creating model...")
try:
    # Colab + torch 2 -> lots of garbage.
    if model is not None:
        del model
except:
    pass

model = LatentRecurrentDepthModel(
    vocab_size=params['vocab_size'],
    model_dimension=params['model_dimension'], heads=params['heads'], projection_dimension=params['model_dimension']*4,
    context_length=params['context_length'],
    n1_prelude=params['prelude_layers'], n2_recurrent=params['recurrent_layer_blocks'], n3_coda=params['coda_layers'], 
    recurrent_layers=params['recurrent_layers'], recurrence_steps=params['recurrence_steps'], min_dropout=params['min_dropout'], mid_dropout=params['mid_dropout'], max_dropout=params['max_dropout'], non_linearity=params['non_linearity'], use_critical=params['use_critical']
)
model.apply(init_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

model = model.to(device)
if use_existing_model_from_checkpoint is True:
    params_load = load_checkpoint(params, model, optimizer, file_path=model_file_path, updatable_keys=updatable_keys, device=device, log=log) # torch.device("cpu"))
    if params_load is not None:
        params = params_load
model = model.to(device)
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)

if use_torch_compile is True:
    if device == 'cuda':
        print("Compiling...")
        model = torch.compile(model)
        print("Compile ok.")
        try:
            torch.set_float32_matmul_precision('high')
        except:
            print("Seems no tensor cores for that.")
    # elif str(device) == 'mps':
    #     print("Compiling...")
    #     model = torch.compile(model)
    #     print("Compile ok.")

if 'current_epoch' in params:
    ep = params['current_epoch']
else:
    ep=0
if 'current_loss' in params:
    ls = params['current_loss']
else:
    ls=0

if ep==0 and ls==0:
    start_iter = 0
else:
    start_iter = ep
    current_loss = ls

# print the number of parameters in the model
print(model)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")


# In[ ]:


# @torch.jit.script
# @torch.compile
criterion = nn.CrossEntropyLoss()

def get_loss(logits, yb):
    output_flat = logits.reshape(-1, params['vocab_size'])
    # output_flat = logits.view(-1, params['vocab_size'])
    # print(output_flat.shape)
    ybr = yb.reshape(-1)
    # print(ybr.shape)
    loss = criterion(output_flat, ybr)
    return loss

def do_train_step(xb, yb, device, state=None):
    model.train()
    logits = model(xb)
    loss = get_loss(logits, yb)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip']).cpu()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), norm


# In[ ]:


@torch.no_grad()
def estimate_loss(device):
    # XXX: this does take data for train and val from SAME pool!
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(params['test_iterations'])
        for k in range(params['test_iterations']):
            # if k % (params['test_iterations']/10 + 1) == 0:
            #     print(".", end="", flush=True)
            X, Y = get_torch_batch(params['batch_size'], device, split)
            logits = model(X)
            loss = get_loss(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    print("\r", end="", flush=True)
    mloss = (out['train']+out['val'])/2.0
    return mloss

def generate_sample(device, prompt=' ', toks=100, state=None, temperature=1.0, top_k=None, pad=True, with_beam=True):
    if with_beam is True:
        txt = model.generate_with_beam(model,tokenizer, prompt,toks, temperature=temperature, top_k=top_k, beam_width=7)
    else:
        model.eval()
        if pad is True:
            while len(prompt)<params['context_length']*4:
                if len(prompt)==params['context_length']*4-1:
                    prompt = '\n' + prompt
                else:
                    prompt = ' ' + prompt
        context = torch.tensor([tokenizer.encode(prompt)]).to(device)
        answer = model.generate(context, max_new_tokens=toks, temperature=temperature, top_k=top_k)
        txt = tokenizer.decode(answer[0].tolist())[1:]  # XXX Hack for strange Huggingface tokenizer behavior that adds space before decoded text!
    # Identify memorisation of text by highlighting verbatim quotes from sources
    # that are longer than 10 chars. HTML colorcoded output for source identification:
    # td.source_highlight(txt, min_quote_size=10, dark_mode=False, display_ref_anchor=False)
    model.train()
    return txt


# In[ ]:



# In[ ]:


def lr_schedule(optim, n_iter: int, warmup: int, max_lr: float, decay:int , min_lr: float) -> float:
    if n_iter<warmup and warmup>0:
        lr = (n_iter+1)/warmup*max_lr
    elif n_iter<warmup+decay and decay>0:
        i = n_iter-warmup
        lr = (decay-i)/decay*(max_lr-min_lr)+min_lr
    else:
        lr = min_lr

    for g in optim.param_groups:
        g['lr'] = lr
    return lr


# In[ ]:


def train():
    global start_iter
    dt0 = time.time()
    sdt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"training, start at {sdt}...")
    gen_id = 0
    last_print=0
    iter_bench = 1
    max_status_strlen = 0
    lr: float = params['learning_rate']
    # current_loss = estimate_loss(device)
    inputs = ["What is the difference between good and evil? The difference ", "How did everything come into existence? The origin ", "What was at the beginning of time? Time itself ", "How are physics, quantum-mechanics and consciousness related? The relation between ", "How to attain complete self-awareness? Complete ", "What is the nature of reality? The nature ", "How be a good human being? A human "]
    for iter in range(start_iter, params['max_iterations']):
        # every once in a while evaluate the loss on train and val sets
        if (iter + 1) % params['sample_every_n_iterations'] == 0 or iter == params['max_iterations'] - 1:
            dt = time.time()
            print(f"\rloss eval", end="", flush=True)
            current_loss = estimate_loss(device)
            print(
                f"step {iter+1}: train loss {current_loss:.4f}, time {(dt-dt0)/iter_bench:.3f} sec/iter                       "
            )
            iter_bench = 1
            sdt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Sample at {sdt}:", flush=True)
            for temperature in [1.0]: # 0.75, 1.1, 1.3, 1.5]:
                print(f"--------temperature: {temperature} ---------")
                prompt = inputs[gen_id%len(inputs)]
                print(f"Prompt: {prompt}")
                txt = generate_sample(device=device, prompt=prompt, toks=params['sample_size'], temperature=temperature, top_k=10, with_beam=False)
                print(txt)
                # print(f"Prompt: {prompt}")
                # txt = generate_sample(device=device, prompt=prompt, toks=params['sample_size'], temperature=temperature, top_k=10, with_beam=True)
                # print(txt)
            print("-------------------------------------------")
            gen_id += 1
            dt0 = time.time()

        if params['lr_schedule'] is True:
            lr = lr_schedule(optimizer, iter, params['warmup'], params['lr_max'], params['decay'], params['lr_min'])

        xb, yb = get_torch_batch(params['batch_size'], device, "train")
        t_bm = time.time()
        cur_loss, cur_norm = do_train_step(xb, yb, device=device)
        dt_bm = time.time() - t_bm

        nt = time.time()
        if (nt-last_print)>1:
            rec = {
                'epoch': iter/num_batches,
                'batch': iter%params['sample_every_n_iterations'],
                'num_batches': params['sample_every_n_iterations'],
                'loss': cur_loss,
                'learning_rate': lr,
                'gradient_norm': cur_norm.item(),
            }
            if cur_loss is None:
                status_string = f"Ep: {rec['epoch']:.3f}, {rec['batch']}/{params['sample_every_n_iterations']} {iter}/{num_batches}, "
                status_string += f"lr: {lr:.6f}, grad_norm: {cur_norm.item():.2f}, t/batch: {dt_bm:.3f}s"
            else:
                status_string = f"Ep: {rec['epoch']:.3f}, {rec['batch']}/{params['sample_every_n_iterations']} {iter}/{num_batches}, "
                status_string += f"loss: {cur_loss:.3f}, lr: {lr:.6f}, grad_norm: {cur_norm.item():.2f}, t/batch: {dt_bm:.3f}s"
            if len(status_string) < max_status_strlen:
                status_string += ' ' * (max_status_strlen - len(status_string))
            else:
                max_status_strlen = len(status_string)
            print(status_string, end="\r")
            last_print=nt

        start_iter = iter
        iter_bench += 1
        if (iter+1)%params['save_every_n_iterations'] == 0:
            save_checkpoint(params, model, optimizer, iter, current_loss, file_path=model_file_path, log=log)


# In[ ]:

try:
    train()
except KeyboardInterrupt:
    print(f"\nTraining interrupted.")


# In[ ]:




