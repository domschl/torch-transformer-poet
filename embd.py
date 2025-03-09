import logging
import os
import json
import time
import datetime
import math
import random
import numpy as np

import sentencepiece as spm

import torch
import torch.nn as nn
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO)
log = logging.Logger("Main")
log.setLevel(logging.INFO)

project_name='embd'
model_cpu = None
model_name=f'{project_name}_v1'

use_preprocessed_data = True                      # Use already tokenized data
use_existing_model_from_checkpoint = False         # Try to load checkpoint of training
use_torch_compile = False                           # Requires a modern graphics card with torch compile backend support

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("mps") if torch.backends.mps.is_available() else device

model_path = "model"
data_path = "data"

use_dark_mode=False # Set to false for white background. HTML-text-compare uses background-colorization to identify different sources. Those background colors are dependent on the theme type.

if use_preprocessed_data is False:
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

    text_list = get_texts()
    text_corpus = '\n\n\n'.join(text_list)

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
            user_defined_symbols = [],
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
        n = len(self.encoded_corpus) - context_length
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

    def encode(self, text:str):
        tokens = self.sp.encode_as_ids(text)
        return tokens

    def visualize_tokens(self, text:str):
        pieces = sp.encode_as_pieces(text)  # For visualization
        return pieces

    def decode(self, ids):
        decoded = self.sp.decode(ids)
        return decoded

vocab_size = 32768

tokenizer = SentencePieceBPE(model_name=model_name, vocab_size=vocab_size, data_directory=data_path)
if use_preprocessed_data is True:
    tokenizer.load()
else:
    tokenizer.train(text_list)

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

params = None
updatable_keys=['learning_rate', 'batch_size', 'current_epoch', 'current_loss',
                 'sample_every_n_iterations', 'save_every_n_iterations', 'max_iterations']
model_dimension = 192
context_length = 128

params = { # Multi-head self-attention
        'meta_name_template': '{prelude_layers}-{recurrent_layers}/{recurrence_steps}-{coda_layers}x{heads}x{units}x{vocab_size}',

        'prelude_layers': 3,
        'coda_layers': 3,
        'yoke_layers': 3,
        'heads': 4,
        'vocab_size': vocab_size,
        'context_length': context_length,
        'min_dropout': 0.1,  # first layer of prelude, last layer of coda
        'max_dropout': 0.4,  # last layer of prelude, first layer of coda
        'mid_dropout': 0.1,  # Used by recurrence
        'weight_decay': 1e-3,  # L2 regularization, applied by Adam optimizer
        'non_linearity': nn.Mish,  # CriticalModule.CriticalActivationLayer,  # Default nn.ReLU
        'model_dimension': model_dimension,
        'yoke_dimension': model_dimension // 8,
        'hard_yoke': 64,
        'test_iterations': 100,  # number of iterations for loss estimation

        'batch_size': 64,
    
        'learning_rate': 4e-4,  # Only used, if lr_schedule is False
        'lr_schedule': True,
        'lr_min': 2e-4,
        'lr_max': 1e-3,
        'warmup': 4000,
        'decay': 50000,
    
        'grad_clip': 0.8,

        'sample_every_n_iterations': 1024,
        'save_every_n_iterations': 1024,

        'max_iterations': 100000000  # maximum number of training iterations
    }

model_file_path = os.path.join(model_path, model_name+".pt")
if use_existing_model_from_checkpoint is True:
    params = load_model_metadata_from_checkpoint(params, updatable_keys, model_file_path, device=device, log=log) # torch.device('cpu'))
if params == None or use_existing_model_from_checkpoint is False:
    use_existing_model_from_checkpoint = False

num_batches = tokenizer.get_record_count(params['context_length']) // params['batch_size']
print(f"Batches: {num_batches}")

def get_sample_batch(batch_size):
    for i in range(batch_size):
        Xi= tokenizer.get_random_tokens(params['context_length'])
        yi = Xi.copy()
        if i==0:
            # smpX=np.array(Xi, dtype=np.float32)
            smpX=np.array(Xi, dtype=np.int32)
            smpy=np.array(yi, dtype=np.int32)
        else:
            # smpX = np.vstack((smpX, np.array(Xi, dtype=np.float32)))
            smpX = np.vstack((smpX, np.array(Xi, dtype=np.int32)))
            smpy = np.vstack((smpy, np.array(yi, dtype=np.int32)))
    return np.array(smpX), np.array(smpy)

x, y = get_sample_batch(1)
x.shape, y.shape

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

class TransformerBlock(nn.Module):
    def __init__(self, model_dimension, heads, projection_dimension, dropout=0.1, non_linearity=nn.ReLU):
        super(TransformerBlock, self).__init__()
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

    def forward(self, x, att_mask, key_pad):
        # if attn_mask is None and x.size(0) > 1:
         #   seq_len = x.size(0)
         #   attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
         #   attn_mask = attn_mask.to(x.device)  # [seq_len, seq_len], upper triangle = True (masked)

        attn_output, _ = self.self_attn(x, x, x,
                                      attn_mask=None,
                                      key_padding_mask=None,
                                      is_causal=False)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class DepthModel(nn.Module):
    def __init__(self, vocab_size, model_dimension, heads, context_length, projection_dimension, yoke_dimension,
                 n1_prelude, n2_yoke, n3_coda, hard_yoke, min_dropout=0.1, mid_dropout=0.2, max_dropout=0.1, non_linearity=nn.ReLU):
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
        super(DepthModel, self).__init__()

        self.context_length = context_length  # for reshaping stuff
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
            tr_list.append(TransformerBlock(model_dimension, heads, model_dimension, drop, non_linearity))
        self.prelude = nn.ModuleList(tr_list)
 
        self.yoke_entry = nn.Linear(model_dimension, yoke_dimension)
        yk_list_1 = []
        for i in range(n2_yoke):
            yk_list_1.append(TransformerBlock(yoke_dimension, heads, yoke_dimension, mid_dropout, non_linearity))
        self.yoke_1 = nn.ModuleList(yk_list_1)
        self.hard_yoke_in = nn.Linear(yoke_dimension * context_length, hard_yoke)
        self.hard_yoke_out = nn.Linear(hard_yoke, yoke_dimension*context_length)
        yk_list_2 = []
        for i in range(n2_yoke):
            yk_list_2.append(TransformerBlock(yoke_dimension, heads, yoke_dimension, mid_dropout, non_linearity))
        self.yoke_2 = nn.ModuleList(yk_list_2)
        self.yoke_exit = nn.Linear(yoke_dimension, model_dimension)
        
        # Coda blocks
        cd_list = []
        for i in range(n3_coda):
            if n3_coda > 1:
                drop = max_dropout - (max_dropout - min_dropout)*(i/(n3_coda-1))
            else:
                drop = (min_dropout + max_dropout) / 2
            if i+1 == n3_coda:
                c_dim = model_dimension
            else:
                c_dim = projection_dimension
            cd_list.append(TransformerBlock(model_dimension, heads, c_dim, drop, non_linearity))
        self.coda = nn.ModuleList(cd_list)

        # Final projection layer (e.g., to vocab size for generation)
        self.proj = nn.Linear(model_dimension, vocab_size)

    def compress(self, input_ids, attn_mask=None, key_padding_mask=None):
        # Embed input tokens
        x = self.embedding(input_ids) * math.sqrt(self.model_dimension) # /2.0  # [batch_size, seq_len, model_dimension]
        # x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [seq_len, batch_size, model_dimension] for transformer
        x = self.pos_encoding(x)

        # Prelude: Entry to latent space
        for block in self.prelude:
            x = block(x, attn_mask, key_padding_mask)

        # Yoke
        x = x.transpose(0, 1)  # [batch_size, seq_len, model_dimension]
        x = self.yoke_entry(x)
        x = x.transpose(0, 1)  # [seq_len, batch_size, model_dimension]
        for block in self.yoke_1:
            x = block(x, attn_mask, key_padding_mask)
        x = x.transpose(0, 1)  # [batch_size, seq_len, model_dimension]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.hard_yoke_in(x)
        return x, batch_size

    def decompress(self, x, batch_size, attn_mask=None, key_padding_mask=None):
        x = self.hard_yoke_out(x)
        x = x.reshape(batch_size, self.context_length, -1)
        x = x.transpose(0, 1)  # [seq_len, batch_size, model_dimension]
        for block in self.yoke_2:
            x = block(x, attn_mask, key_padding_mask)
        x = x.transpose(0, 1)  # [batch_size, seq_len, model_dimension]
        x = self.yoke_exit(x)
        x = x.transpose(0, 1)  # [seq_len, batch_size, model_dimension]
        # Coda: Exit from latent space
        for block in self.coda:
            x = block(x, attn_mask, key_padding_mask)

        # Project to output space
        x = x.transpose(0, 1)  # [batch_size, seq_len, model_dimension]
        output = self.proj(x)  # [batch_size, seq_len, vocab_size]
        return output

    def forward(self, input_ids, attn_mask=None, key_padding_mask=None):
        """
        Args:
            input_ids (torch.Tensor): Token IDs [batch_size, seq_len].
            attn_mask (torch.Tensor, optional): Attention mask [seq_len, seq_len].
            key_padding_mask (torch.Tensor, optional): Padding mask [batch_size, seq_len].
        Returns:
            torch.Tensor: Output logits [batch_size, seq_len, vocab_size].
        """
        x, bs = self.compress(input_ids, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        output = self.decompress(x, bs, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return output

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.GRU, nn.LSTM)):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)


print("creating model...")
try:
    # Colab + torch 2 -> lots of garbage.
    if model is not None:
        del model
except:
    pass

model = DepthModel(
    vocab_size=params['vocab_size'],
    model_dimension=params['model_dimension'], heads=params['heads'], projection_dimension=params['model_dimension']*4, yoke_dimension=params['yoke_dimension'],
    context_length=params['context_length'],
    n1_prelude=params['prelude_layers'], n2_yoke=params['yoke_layers'], n3_coda=params['coda_layers'], 
    min_dropout=params['min_dropout'], mid_dropout=params['mid_dropout'], max_dropout=params['max_dropout'], 
    non_linearity=params['non_linearity'], hard_yoke=params['hard_yoke']
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
    elif str(device) == 'mps':
        print("Compiling NOT SUPPORTED (yet) for MPS devices")

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
            model.eval()
            test, _ = get_torch_batch(1, device, "train")
            test_bt = test.reshape(1,-1)
            compr, bs = model.compress(test_bt)
            ans = model.decompress(compr, bs)
            ans_ids = torch.argmax(ans, dim=2)
            model.train()
            str1 = tokenizer.decode(test_bt[0].tolist()).replace("\n", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
            str2 = tokenizer.decode(ans_ids[0].tolist()).replace("\n", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")
            print(str1)
            print(compr.shape) # tolist())
            print(str2)
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

try:
    train()
except KeyboardInterrupt:
    print(f"\nTraining interrupted.")
