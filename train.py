from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import copy

from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_dataset, accuracy, normalize_adj
from models import GCN
import loss


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--triplet_lr', type=float, default=0.01,
                    help='Initial learning rate for SoftTriple Loss.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--num_epoch_per_adj_mat_update', type=int, default=30,
                    help='Adjacency matrix will get updated every N epochs.')
parser.add_argument('--lamda', type=float, default=0.8,
                    help='Weighted averaging between updated and input adjacency matrix.')
parser.add_argument('--beta', type=float, default=0.8,
                    help='Weighted averaging between triplet Loss and MSE loss.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_dataset(dataset_str="cora")

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)

tripleLoss=loss.SoftTriple(dim=args.hidden, cN=7, K=2)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    tripleLoss = tripleLoss.cuda()


optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr},
                        {"params": tripleLoss.parameters(), "lr": args.triplet_lr}],
                        weight_decay=args.weight_decay)

A=adj

saved_accuracy=0.0
saved_loss = 1e10

t_total = time.time()

for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    inter_out1,output = model(features, adj, A)
    loss_train = (1-args.beta)*F.nll_loss(output[idx_train], labels[idx_train])+\
        args.beta*tripleLoss(inter_out1[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        inter_out,output = model(features, adj, A)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    # Choose model at best validation accuracy
    if saved_accuracy < acc_val.item() and saved_loss >= loss_val.item() and epoch>args.num_epoch_per_adj_mat_update:
        model_saved = copy.deepcopy(model)
        saved_accuracy = acc_val.item()
        saved_loss = loss_val.item()
        A_saved = A

    # Update adjacency matrix every 'num_epoch_per_adj_mat_update' epochs
    if (epoch+1)% args.num_epoch_per_adj_mat_update == 0:
        A = cosine_similarity(inter_out1.data.cpu().numpy())
        A = sp.csr_matrix(A).tocoo()
        A = normalize_adj(A).todense()
        A = torch.from_numpy(A).float()
        if args.cuda:
            A = A.cuda()
        A = (1 - args.lamda)*adj + args.lamda*A

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

model_saved.eval()
inter_out,output = model_saved(features, adj, A_saved)
loss_test = F.nll_loss(output[idx_test], labels[idx_test])
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
      "loss= {:.4f}".format(loss_test.item()),
      "accuracy= {:.4f}".format(acc_test.item()))

