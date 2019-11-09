import math
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import utils as utils
import lib.dist as dist
metric_name = 'MIG'
q_dist = dist.Normal()

def MIG(mi_normed):
    return torch.mean(mi_normed[:, 0] - mi_normed[:, 1])


def compute_metric_ecg(marginal_entropies, cond_entropies):
    factor_entropies = [6, 40, 32, 32]
    mutual_infos = marginal_entropies[None] - cond_entropies
    mutual_infos = torch.sort(mutual_infos, dim=1, descending=True)[0].clamp(min=0)
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    metric = eval(metric_name)(mi_normed)
    return metric


def compute_metric_faces(marginal_entropies, cond_entropies):
    factor_entropies = [21, 11, 11]
    mutual_infos = marginal_entropies[None] - cond_entropies
    mutual_infos = torch.sort(mutual_infos, dim=1, descending=True)[0].clamp(min=0)
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    metric = eval(metric_name)(mi_normed)
    return metric


def estimate_entropies(qz_samples, qz_params, q_dist, n_samples=10000, weights=None):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    where q(z) = 1/N sum_n=1^N q(z|x_n).
    Assumes samples are from q(z|x) for *all* x in the dataset.
    Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).

    Computes numerically stable NLL:
        - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)

    Inputs:
    -------
        qz_samples (K, N) Variable
        qz_params  (N, K, nparams) Variable
        weights (N) Variable
    """

    # Only take a sample subset of the samples
    if weights is None:
        qz_samples = qz_samples.index_select(1, Variable(torch.randperm(qz_samples.size(1))[:n_samples].cuda()))
    else:
        sample_inds = torch.multinomial(weights, n_samples, replacement=True)
        qz_samples = qz_samples.index_select(1, sample_inds)

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()
    # assert(nparams == q_dist.nparams)
    assert (nparams == 2)
    # assert(K == qz_params.size(1))
    assert (K == 100)

    if weights is None:
        weights = -math.log(N)
    else:
        weights = torch.log(weights.view(N, 1, 1) / weights.sum())

    entropies = torch.zeros(K).cuda()

    #pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(10, S - k)
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        # entropies += - utils.logsumexp(logqz_i + weights, dim=0, keepdim=False).data.sum(1)
        # print("entropies type: ", type(entropies))
        # print("type(weights): ", type(weights))
        # print("type(logqz_i): ", type(logqz_i))
        # print("2nd arg: ", type(- torch.logsumexp(torch.from_numpy(logqz_i) + weights, dim=0, keepdim=False).data.sum(1)))

        entropies = (entropies) + (- torch.logsumexp(torch.from_numpy(logqz_i).cuda() + weights, dim=0, keepdim=False).data.sum(1))
        #pbar.update(batch_size)
    #pbar.close()

    entropies /= S

    return entropies


def reparametrize_gaussian(params):
    # noise = Variable(mu.data.clone().normal_(0, 1), requires_grad=False)
    # return mu + (noise * logvar.exp())
    mu = params[:, :, :, :, :, :, :, 0]
    logvar = params[:, :, :, :, :, :, :, 1]

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

 
def mutual_info_metric_shapes(vae, dataset_loader, batch_size):
    # dataset_loader = DataLoader(shapes_dataset, batch_size=1000, num_workers=1, shuffle=False)

    N = 404595  # len(dataset_loader.dataset)  # number of data samples
    K = 100                   # number of latent variables
    # nparams = vae.q_dist.nparams
    nparams = 2
    vae.double().eval().cuda()

    print('Computing q(z|x) distributions.')
    print("N: ", N)
    qz_params = torch.Tensor(N, K, nparams)

    n = 0
    # for xs in dataset_loader:
    for batch_idx, (xs, zs) in enumerate(dataset_loader):
        batch_size = xs.size(0)

        # xs = Variable(xs.view(batch_size, 1, 64, 64).cuda(), volatile=True)

        #qz_params[n:n + batch_size] = vae.encoder.forward(xs).view(batch_size, vae.z_dim, nparams).data
        #_, _, _, _, z_mean, z_logvar, z_discrete, _ = vae(input)

        xs = xs.double().cuda()
        # _, z_mean, z_logvar = vae.encode(xs)
        z_mean, z_logvar, _, _ = vae.encode(xs)

        # print("z_mean.data.size(): ", z_mean.data.size())
        qz_params[n:n + batch_size, :, 0] = z_mean.data.squeeze(2)
        qz_params[n:n + batch_size, :, 1] = z_logvar.data.squeeze(2)

        n += batch_size
    print("qz_params.size(): ", qz_params.size())
    # qz_params = Variable(qz_params.view(3, 6, 40, 32, 32, K, nparams).cuda())

    # need config numbers for last 3 factors
    # can keep it as (3, 3, 3, 3, 3, 1665) for first 5 factors
    # Each file has 1665 configurations for the x,y,z coordinates #x_cor*y_cor*z_cor = 1665.
    # qz_params = Variable(qz_params.view(3, 3, 3, 3, 3, K, nparams).cuda())
    qz_params = Variable(qz_params.view(3, 3, 3, 3, 3, 1665, K, nparams).cuda())

    # qz_samples = vae.q_dist.sample(params=qz_params)
    qz_samples = reparametrize_gaussian(params=qz_params)

    print('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qz_samples.view(N, K).transpose(0, 1),
        qz_params.view(N, K, nparams),
        q_dist)  # changed from vae.q_dist to just q_dist

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(4, K)

    print('Estimating conditional entropies for Factor 1.')
    for i in range(3):
        qz_samples_scale = qz_samples[i, :, :, :, :, :].contiguous()
        qz_params_scale = qz_params[i, :, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 3, K).transpose(0, 1),
            qz_params_scale.view(N // 3, K, nparams),
            q_dist)  # changed from vae.q_dist to just q_dist

        cond_entropies[0] += cond_entropies_i.cpu() / 6

    print('Estimating conditional entropies for Factor 2.')
    for i in range(3):
        qz_samples_scale = qz_samples[:, i, :, :, :, :].contiguous()
        qz_params_scale = qz_params[:, i, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 3, K).transpose(0, 1),
            qz_params_scale.view(N // 3, K, nparams),
            q_dist)

        cond_entropies[1] += cond_entropies_i.cpu() / 3

    print('Estimating conditional entropies for Factor 3.')
    for i in range(3):
        qz_samples_scale = qz_samples[:, :, i, :, :, :].contiguous()
        qz_params_scale = qz_params[:, :, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 3, K).transpose(0, 1),
            qz_params_scale.view(N // 3, K, nparams),
            q_dist)

        cond_entropies[2] += cond_entropies_i.cpu() / 3

    print('Estimating conditional entropies for Factor 4.')
    for i in range(3):
        qz_samples_scale = qz_samples[:, :, :, :, i, :].contiguous()
        qz_params_scale = qz_params[:, :, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 3, K).transpose(0, 1),
            qz_params_scale.view(N // 3, K, nparams),
            q_dist)

        cond_entropies[3] += cond_entropies_i.cpu() / 3

    print('Estimating conditional entropies for Factor 5.')
    for i in range(1665):
        qz_samples_scale = qz_samples[:, :, :, :, :, i].contiguous()
        qz_params_scale = qz_params[:, :, :, :, :, i].contiguous()

        cond_entropies_i = estimate_entropies(
            qz_samples_scale.view(N // 1665, K).transpose(0, 1),
            qz_params_scale.view(N // 1665, K, nparams),
            q_dist)

        cond_entropies[1665] += cond_entropies_i.cpu() / 1665

    metric = compute_metric_ecg(marginal_entropies, cond_entropies)
    return metric, marginal_entropies, cond_entropies