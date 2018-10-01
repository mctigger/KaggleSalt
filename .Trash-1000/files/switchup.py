import torch

m = []
for i in range(20000):
    x = torch.empty(1, 3, 101, 101).uniform_(0, 1)
    m.append(x)
input("SADSD")