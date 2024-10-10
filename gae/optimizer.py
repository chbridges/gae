import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_ae(preds, labels, pos_weight, norm):
    return norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)


def loss_vae(preds, labels, z_mean, z_log_std, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * z_log_std - z_mean.pow(2) - z_log_std.exp().pow(2), 1))
    return cost + KLD
