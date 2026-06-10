import torch

block_size = 1024
batch_size = 8
max_iters = 20000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.2