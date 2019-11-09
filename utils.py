import torch
import numpy as np
import copy
import importlib.util
import torch.utils.data as utils
from gvae_ecg.TMI_Datasets import TMI_Dataset
import matplotlib.pyplot as plt
# from main import arg_parser  # this import now deferred inside functions - to avoid circular import issues
from argparse import Namespace
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

import gvae_ecg.cnn_model, cnn_model_new1, mlp_model
SMALL = 1e-16


# crucial - required for ibp
def logit(x):
    return (x + SMALL).log() - (1. - x + SMALL).log()


def recon_loss(args, pred, data):
    # We'd like to maximize p(x_hat | z) - maximizing this likelihood is same as maximizing log-likelihood,
    # which is same as minimising negative-log-likelihood - can be estimated by reconstruction squared difference
    # in our case, which we'll minimize.
    # print("pred.size(): ", pred.size())
    # print("data.size(): ", data.size())
    my_data = data
    if args.dataset_type.lower() == 'simulated':
        my_pred = pred.squeeze(1)
    else:
        my_pred = pred
    return (my_pred - my_data).pow(2)


def nll_and_kl_group(recon_x, x, mu_group, logvar_group, mu_variations, logvar_variations, content_mu, content_logvar, indexes, a, b, logsample, z_discrete, logit_post, log_prior
               , dataset_size, model_type='', args=Namespace(), test=False):
    from main import arg_parser
    args = arg_parser()
    model_type = model_type.lower()
    if model_type == 'ibp':
        from commonTraining import kl_divergence, kl_discrete, log_sum_exp, print_in_epoch_summary, print_epoch_summary, \
            mse_loss
        batch_size = x.size()[0]
        if args.dataset_type.lower() == 'simulated':
            x = x[:, 1:-1]
        recon_x = recon_x.squeeze(1)
        # NLL = -1 * log_likelihood(recon_x, x)
        NLL = torch.sum(recon_loss(recon_x, x), dim=1).sum()
        n_groups = content_mu.size(0)
        KL_group = -0.5 * torch.sum(1. + content_logvar - content_mu ** 2 - content_logvar.exp())
        KL_variations = -0.5 * torch.sum(1. + logvar_variations[indexes, :] - mu_variations[indexes, :] ** 2 - logvar_variations[indexes, :].exp())
        KL_zreal = KL_group + KL_variations

        KL_beta = kl_divergence(a, b, prior_alpha=args.alpha0, log_beta_prior=np.log(1. / args.alpha0),
                                args=args).repeat(batch_size, 1) * (1. / dataset_size)

        # in test mode, our samples are essentially coming from a Bernoulli
        if not test:
            KL_discrete = kl_discrete(logit_post, logit(log_prior.exp()), logsample, args.temp, args.temp_prior)
        else:
            pi_prior = torch.exp(log_prior)
            pi_posterior = torch.sigmoid(logit_post)
            kl_1 = z_discrete * (pi_posterior + SMALL).log() + (1 - z_discrete) * (1 - pi_posterior + SMALL).log()
            kl_2 = z_discrete * (pi_prior + SMALL).log() + (1 - z_discrete) * (1 - pi_prior + SMALL).log()
            KL_discrete = kl_1 - kl_2
        KLD = KL_zreal.squeeze(2) + KL_beta + KL_discrete
        return NLL, KLD.sum(), n_groups
    else:
        if args.dataset_type.lower() == 'simulated':
            x = x[:, 1:-1]
        recon_x = recon_x.squeeze(1)

        # NLL = torch.sum(torch.sum(log_likelihood(recon_x, x), dim=1), dim=1).sum() * (1/recon_x.size(0))
        NLL = torch.sum(recon_loss(args, recon_x, x), dim=1).sum()
        n_groups = content_mu.size(0)
        KL_group = - 0.5 * torch.sum(1 + content_logvar - content_mu**2 - content_logvar.exp())
        KL_variations = - 0.5 * torch.sum(1 + logvar_variations[indexes, :] - mu_variations[indexes, :] ** 2 - logvar_variations[indexes, :].exp())
        KLD = KL_group + KL_variations

        return NLL, KLD, n_groups

def nll_and_kl_iid(recon_x, x, mu, logvar, a, b, logsample, z_discrete, logit_post, log_prior, dataset_size,
                   model_type='', args=Namespace(), test=False):
    from main import arg_parser
    args = arg_parser()
    model_type = model_type.lower()
    if model_type == 'ibp':
        from commonTraining import kl_divergence, kl_discrete, log_sum_exp, print_in_epoch_summary, print_epoch_summary, \
            mse_loss
        batch_size = x.size()[0]
        if args.dataset_type.lower() == 'simulated':
            x = x[:, 1:-1]
        recon_x = recon_x.squeeze(1)
        # NLL = -1 * log_likelihood(recon_x, x)
        NLL = torch.sum(recon_loss(recon_x, x), dim=1).sum()
        KL_zreal = -0.5 * torch.sum(1. + logvar - mu ** 2 - logvar.exp())

        KL_beta = kl_divergence(a, b, prior_alpha=args.alpha0, log_beta_prior=np.log(1. / args.alpha0),
                                args=args).repeat(batch_size, 1) * (1. / dataset_size)

        # in test mode, our samples are essentially coming from a Bernoulli
        if not test:
            KL_discrete = kl_discrete(logit_post, logit(log_prior.exp()), logsample, args.temp, args.temp_prior)
        else:
            pi_prior = torch.exp(log_prior)
            pi_posterior = torch.sigmoid(logit_post)
            kl_1 = z_discrete * (pi_posterior + SMALL).log() + (1 - z_discrete) * (1 - pi_posterior + SMALL).log()
            kl_2 = z_discrete * (pi_prior + SMALL).log() + (1 - z_discrete) * (1 - pi_prior + SMALL).log()
            KL_discrete = kl_1 - kl_2
        KLD = KL_zreal.squeeze(2) + KL_beta + KL_discrete
        return NLL, KLD.sum()
    else:
        if args.dataset_type.lower() == 'simulated':
            x = x[:, 1:-1]
        recon_x = recon_x.squeeze(1)

        # NLL = torch.sum(torch.sum(log_likelihood(recon_x, x), dim=1), dim=1).sum() * (1/recon_x.size(0))
        NLL = torch.sum(recon_loss(args, recon_x, x), dim=1).sum()
        KLD = - 0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

        return NLL, KLD


def elbo_group(recon_x, x, mu_group, logvar_group, mu_variations, logvar_variations, content_mu, content_logvar, indexes, beta, a, b, logsample, z_discrete, logit_post, log_prior,
         dataset_size, model_type='', args=Namespace(), test=False):
    from main import arg_parser
    args = arg_parser()
    NLL, KLD, n_groups = nll_and_kl_group(recon_x, x, mu_group, logvar_group, mu_variations, logvar_variations, content_mu, content_logvar, indexes, a, b, logsample, z_discrete, logit_post, log_prior,
                          dataset_size, model_type, args=args, test=False)
    elbo_loss = ((beta * KLD) + NLL)/n_groups
    return elbo_loss, (NLL/n_groups, KLD/n_groups)


def elbo_iid(recon_x, x, mu, logvar, beta, a, b, logsample, z_discrete, logit_post, log_prior, dataset_size,
             model_type='', args=Namespace(), test=False):
    from main import arg_parser
    args = arg_parser()
    NLL, KLD = nll_and_kl_iid(recon_x, x, mu, logvar, a, b, logsample, z_discrete, logit_post, log_prior,
                          dataset_size, model_type, args=args, test=False)
    elbo_loss = (beta * KLD) + NLL
    return elbo_loss, (NLL, KLD)



def cross_entropy(y, logits):
    # return -torch.sum(y * torch.log(logits + SMALL), dim=1)
    return F.binary_cross_entropy(input=logits + SMALL, target=y)


def get_model_reference(args, model_store_dir, training_flag=True, load_from_checkpoint=False):
    from main import arg_parser
    args = arg_parser()
    if args.dataset_type == 'clinical' and args.architecture_type == 'cnn':
        # spec = importlib.util.spec_from_file_location("model", args.architecture_type + "_model_new1.py")
        model = cnn_model_new1.VAE_Concrete_Simulated(args, training_flag=training_flag)
    elif args.dataset_type == 'simulated' and args.architecture_type == 'cnn':
        model = cnn_model.VAE_Concrete_Simulated(args, training_flag=training_flag)
    elif args.dataset_type == 'simulated' and args.architecture_type == 'mlp':
        model = mlp_model.VAE_Concrete_Simulated(args)
    else:
        spec = importlib.util.spec_from_file_location("model", args.architecture_type+"_model.py")
    # ref = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(ref)

    # model = ref.Model(args, training_flag=training_flag).to(args.device)
    if load_from_checkpoint:
        print("Loading weights from: ", model_store_dir + '/' + args.model_name + '.pt')
        model.load_state_dict(torch.load(model_store_dir + '/' + args.model_name + '_classification.pt'))

    model = model.double().to(args.device)

    print("Total number of parameters: ", sum(p.numel() for p in model.parameters()))
    print("Total number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


def min_max_scaling(x):
    data = copy.deepcopy(x)
    if len(data) > 0:
        data = np.array(data)
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    return data


def plot_roc(args, model_store_dir):
    try:
        predictions = np.load(model_store_dir + '/predictions.npy')
        labels = np.load(model_store_dir + '/labels.npy')
    except:
        print("Predictions and labels not passed to plot_roc function. Also, not found at model_store_dir: {}".format(model_store_dir))
        print("Cannot plot ROC, Returning ...")
        return True

    # Binarize the output
    label_vals = np.argmax(labels, axis=1)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(args.num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot of a ROC curve for a specific class
    plt.figure()

    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=2, label='AUC-ROC (area = %0.4f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(model_store_dir+'/ROC_plot.png', dpi=500)
    plt.close()


def plot_roc(args, model_store_dir):
    try:
        predictions = np.load(model_store_dir + '/predictions.npy')
        labels = np.load(model_store_dir + '/labels.npy')
    except:
        print("Predictions and labels not passed to plot_roc function. Also, not found at model_store_dir: {}".format(model_store_dir))
        print("Cannot plot ROC, Returning ...")
        return True

    # Binarize the output
    label_vals = np.argmax(labels, axis=1)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(args.num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot of a ROC curve for a specific class
    plt.figure()

    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=2, label='AUC-ROC (area = %0.4f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(model_store_dir+'/ROC_plot.png', dpi=500)
    plt.close()


def plot_roc_multiclass(args, model_store_dir):
    try:
        predictions = np.load(model_store_dir + '/predictions.npy')
        labels = np.load(model_store_dir + '/labels.npy')
    except:
        print("Predictions and labels not passed to plot_roc function. Also, not found at model_store_dir: {}".format(model_store_dir))
        print("Cannot plot ROC, Returning ...")
        return True

    # Binarize the output
    label_vals = np.argmax(labels, axis=1)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(args.num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # -----------------------------------------------------------------------------
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(args.num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(args.num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= args.num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # -----------------------------------------------------------------------------
    np.save(file=model_store_dir+'/fpr_macro.npy', arr=fpr["macro"])
    np.save(file=model_store_dir+'/tpr_macro.npy', arr=tpr["macro"])
    np.save(file=model_store_dir + '/roc_auc_macro.npy', arr=roc_auc["macro"])

    # Plot of a ROC curve for a specific class
    plt.figure()

    plt.plot(fpr['macro'], tpr['macro'], color='darkorange', lw=2, label='AUC-ROC (area = %0.4f)' % roc_auc["macro"])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Draws a diagonal line through center of the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(model_store_dir+'/ROC_plot.png', dpi=500)
    plt.close()


def plot_training_history(train_scores, validation_scores, PLOTS_STORE_DIR, learning_rate_log,
            NLL_log_train, NLL_log_val, KLD_log_train, KLD_log_val, ELBO_log_train, ELBO_log_val,
            XENTROPY_log_train, XENTROPY_log_val, ACCURACY_log_train, ACCURACY_log_val, validate_during_training,
            learning_type, NLL_log_train_epoch, KLD_log_train_epoch, ELBO_log_train_epoch, XENTROPY_log_train_epoch,
            ACCURACY_log_train_epoch, NLL_log_val_epoch, KLD_log_val_epoch, ELBO_log_val_epoch, XENTROPY_log_val_epoch,
            ACCURACY_log_val_epoch, normalize_plot):

    # backup logs
    np.save(file=PLOTS_STORE_DIR+'/NLL_log_train.npy', arr=NLL_log_train)
    np.save(file=PLOTS_STORE_DIR+'/KLD_log_train.npy', arr=KLD_log_train)
    np.save(file=PLOTS_STORE_DIR+'/ELBO_log_train.npy', arr=ELBO_log_train)
    np.save(file=PLOTS_STORE_DIR+'/XENTROPY_log_train.npy', arr=XENTROPY_log_train)
    np.save(file=PLOTS_STORE_DIR+'/ACCURACY_log_train.npy', arr=ACCURACY_log_train)

    from main import arg_parser
    args = arg_parser()

    if validate_during_training:
        legend_list = ['train', 'validation']
    else:
        legend_list = ['train']

    if normalize_plot:
        fig_name_extention = 'normalized'
    else:
        fig_name_extention = 'unnormalized'
    # -----------------------------------------------------------------
    if normalize_plot:
        plt.plot(min_max_scaling(train_scores))
    else:
        plt.plot(train_scores)

    if validate_during_training:
        if normalize_plot:
            plt.plot(min_max_scaling(validation_scores))
        else:
            plt.plot(validation_scores)

    plt.xlabel("Epoch")
    plt.ylabel("Average ELBO Loss")
    plt.legend(legend_list)
    plt.grid(b=True)
    plt.title('ELBO loss Progression during training')
    plt.savefig(PLOTS_STORE_DIR+"/ELBO_loss_plot_"+fig_name_extention, dpi=400)
    plt.close()
    print("ELBO Loss plotted ...")
    # -----------------------------------------------------------------
    plt.plot(learning_rate_log)
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.legend(['Learning Rate'])
    plt.grid(b=True)
    plt.title("Learning Rate Progression")
    plt.savefig(PLOTS_STORE_DIR+"/Learning_rate_schedule", dpi=400)
    plt.close()
    print("Learning Rate Progression plotted ...")
    # -----------------------------------------------------------------
    if learning_type == 'supervised':
        num_plots = 4
    elif learning_type == 'unsupervised':
        num_plots = 3

    plt.figure(figsize=(18, 15))
    plt.subplot(num_plots, 1, 1)
    if normalize_plot:
        plt.plot(min_max_scaling(NLL_log_train))
    else:
        plt.plot(NLL_log_train)

    if validate_during_training:
        if normalize_plot:
            plt.plot(min_max_scaling(NLL_log_val))
        else:
            plt.plot(NLL_log_val)
    plt.legend(legend_list)
    plt.ylabel("NLL loss")

    plt.subplot(num_plots, 1, 2)
    if normalize_plot:
        plt.plot(min_max_scaling(KLD_log_train))
    else:
        plt.plot(KLD_log_train)
    if validate_during_training:
        if normalize_plot:
            plt.plot(min_max_scaling(KLD_log_val))
        else:
            plt.plot(KLD_log_val)
    plt.ylabel("KLD loss")
    plt.legend(legend_list)

    plt.subplot(num_plots, 1, 3)
    if normalize_plot:
        plt.plot(min_max_scaling(ELBO_log_train))
    else:
        plt.plot(ELBO_log_train)

    if validate_during_training:
        if normalize_plot:
            plt.plot(min_max_scaling(ELBO_log_val))
        else:
            plt.plot(ELBO_log_val)
    plt.ylabel("ELBO loss")
    plt.legend(legend_list)

    if learning_type == 'supervised':
        plt.subplot(num_plots, 1, 4)
        if normalize_plot:
            plt.plot(min_max_scaling(XENTROPY_log_train))
        else:
            plt.plot(XENTROPY_log_train)

        if validate_during_training:
            if normalize_plot:
                plt.plot(min_max_scaling(XENTROPY_log_val))
            else:
                plt.plot(XENTROPY_log_val)
        plt.xlabel("Number of Mini-Batches, with batch size: {:}".format(args.batch_size))
        plt.ylabel("XENTROPY loss")
        plt.legend(legend_list)
    plt.tight_layout()
    plt.title('Detailed Training Loss History')
    plt.savefig(PLOTS_STORE_DIR + "/all_losses_plot_"+fig_name_extention, dpi=400)
    plt.close()
    print("Consolidated loss plot plotted ...")
    # -----------------------------------------------------------------
    if learning_type == 'supervised':
        plt.figure()
        plt.plot(ACCURACY_log_train)
        if validate_during_training:
            plt.plot(ACCURACY_log_val)
        plt.xlabel("Mini-Batches")
        plt.ylabel("ACCURACY")
        plt.legend(legend_list)
        plt.title("ACCURACY Progression")
        plt.savefig(PLOTS_STORE_DIR + "/accuracy_plot", dpi=400)
        plt.close()
        print("Accuracy plot plotted ...")
    # -----------------------------------------------------------------
    # Plot per-epoch loss values
    if learning_type == 'supervised':
        num_plots = 4
    elif learning_type == 'unsupervised':
        num_plots = 3

    plt.figure(figsize=(18, 15))
    plt.subplot(num_plots, 1, 1)
    if normalize_plot:
        plt.plot(min_max_scaling(NLL_log_train_epoch))
    else:
        plt.plot(NLL_log_train_epoch)
    if normalize_plot:
        plt.plot(min_max_scaling(NLL_log_val_epoch))
    else:
        plt.plot(NLL_log_val_epoch)
    plt.legend(legend_list)
    plt.ylabel("NLL loss")

    plt.subplot(num_plots, 1, 2)
    if normalize_plot:
        plt.plot(min_max_scaling(KLD_log_train_epoch))
    else:
        plt.plot(KLD_log_train_epoch)
    if normalize_plot:
        plt.plot(min_max_scaling(KLD_log_val_epoch))
    else:
        plt.plot(KLD_log_val_epoch)
    plt.legend(legend_list)
    plt.ylabel("KLD loss")

    plt.subplot(num_plots, 1, 3)
    if normalize_plot:
        plt.plot(min_max_scaling(ELBO_log_train_epoch))
    else:
        plt.plot(ELBO_log_train_epoch)
    if normalize_plot:
        plt.plot(min_max_scaling(ELBO_log_val_epoch))
    else:
        plt.plot(ELBO_log_val_epoch)
    plt.legend(legend_list)
    plt.ylabel("ELBO loss")

    if learning_type == 'supervised':
        plt.subplot(num_plots, 1, 4)
        if normalize_plot:
            plt.plot(min_max_scaling(XENTROPY_log_train_epoch))
        else:
            plt.plot(XENTROPY_log_train_epoch)
        if normalize_plot:
            plt.plot(min_max_scaling(XENTROPY_log_val_epoch))
        else:
            plt.plot(XENTROPY_log_val_epoch)

        plt.legend(legend_list)
        plt.ylabel("XENTROPY loss")
        plt.xlabel("Epoch")
    plt.tight_layout()
    plt.title('Per-epoch Training Loss History')
    plt.savefig(fname=PLOTS_STORE_DIR + "/per_epoch_losses_plot_"+fig_name_extention, dpi=400)
    plt.close()
    print("Per epoch losses plotted ...")
    # ------------------------------------
    # Plot per-epoch accuracy
    if learning_type == 'supervised':
        plt.figure()
        plt.plot(ACCURACY_log_train_epoch)
        if validate_during_training:
            plt.plot(ACCURACY_log_val_epoch)
        plt.xlabel("Epochs")
        plt.ylabel("Average Epoch Accuracy")
        plt.legend(legend_list)
        plt.title("Per-epoch Accuracy Progression")
        plt.savefig(PLOTS_STORE_DIR + "/per_epoch_accuracy_plot", dpi=400)
        plt.close()
        print("Per-epoch accuracy plotted ...")

    print("Training history plots have been created...")
    print("Plots stored at: ", PLOTS_STORE_DIR)


def get_data_loader(args, mode, data_dir):
    mode = mode.lower()
    print("Loading {} data-set ...".format(mode))
    shuffle_flag = {'train': True, 'validation': False, 'test': False}
    print("dataset_type in get_data_loader: ", args.dataset_type)
    if args.dataset_type == 'simulated':
        x = torch.stack([torch.Tensor(i) for i in TMI_Dataset(
            mode=mode, root=data_dir, dataset_type=args.dataset_type).np_data[:args.samples_to_read]])
        y = torch.stack([torch.Tensor(i) for i in TMI_Dataset(
            mode=mode, root=data_dir, dataset_type=args.dataset_type).np_targets[:args.samples_to_read]])
    elif args.dataset_type == 'clinical':
        x = TMI_Dataset(mode=mode, root=data_dir, dataset_type=args.dataset_type).data[:args.samples_to_read, :]
        y = TMI_Dataset(mode=mode, root=data_dir, dataset_type=args.dataset_type).label[:args.samples_to_read]

    dataset = utils.TensorDataset(x, y)
    dataloader = utils.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle_flag[mode])

    return dataloader


def neighbourhood(predicted, true):
    neigh = []
    # neigh.append(np.array((0, 1, 1, 1, 0, 0, 0, 0, 0, 0)))
    # neigh.append(np.array((1, 0, 1, 0, 1, 0, 0, 0, 0, 0)))
    # neigh.append(np.array((1, 1, 0, 0, 0, 1, 0, 0, 0, 0)))
    # neigh.append(np.array((1, 0, 0, 0, 1, 1, 1, 1, 0, 0)))
    # neigh.append(np.array((0, 1, 0, 1, 0, 1, 0, 1, 1, 0)))
    # neigh.append(np.array((0, 0, 1, 1, 1, 0, 1, 0, 1, 1)))
    # neigh.append(np.array((0, 0, 0, 1, 0, 1, 0, 1, 0, 1)))
    # neigh.append(np.array((0, 0, 0, 1, 1, 0, 1, 0, 1, 0)))
    # neigh.append(np.array((0, 0, 0, 0, 1, 1, 0, 1, 0, 1)))
    # neigh.append(np.array((0, 0, 0, 0, 0, 1, 1, 0, 1, 0)))

    neigh.append(np.array((1, 1, 1, 1, 0, 0, 0, 0, 0, 0)))
    neigh.append(np.array((1, 1, 1, 0, 1, 0, 0, 0, 0, 0)))
    neigh.append(np.array((1, 1, 1, 0, 0, 1, 0, 0, 0, 0)))
    neigh.append(np.array((1, 0, 0, 1, 1, 1, 1, 1, 0, 0)))
    neigh.append(np.array((0, 1, 0, 1, 1, 1, 0, 1, 1, 0)))
    neigh.append(np.array((0, 0, 1, 1, 1, 1, 1, 0, 1, 1)))
    neigh.append(np.array((0, 0, 0, 1, 0, 1, 1, 1, 0, 1)))
    neigh.append(np.array((0, 0, 0, 1, 1, 0, 1, 1, 1, 0)))
    neigh.append(np.array((0, 0, 0, 0, 1, 1, 0, 1, 1, 1)))
    neigh.append(np.array((0, 0, 0, 0, 0, 1, 1, 0, 1, 1)))
    num = predicted.size(0)

    correct = 0
    for i in range(num):
        if neigh[true[i] ][predicted[i]] == 1:
            correct += 1

    return correct


def neighbourhood_hard(softmax_y, true):
    correct = 0
    neigh = []
    neigh.append(np.array((1, 1, 1, 1, 0, 0, 0, 0, 0, 0)))
    neigh.append(np.array((1, 1, 1, 0, 1, 0, 0, 0, 0, 0)))
    neigh.append(np.array((1, 1, 1, 0, 0, 1, 0, 0, 0, 0)))
    neigh.append(np.array((1, 0, 0, 1, 1, 1, 1, 1, 0, 0)))
    neigh.append(np.array((0, 1, 0, 1, 1, 1, 0, 1, 1, 0)))
    neigh.append(np.array((0, 0, 1, 1, 1, 1, 1, 0, 1, 1)))
    neigh.append(np.array((0, 0, 0, 1, 0, 1, 1, 1, 0, 1)))
    neigh.append(np.array((0, 0, 0, 1, 1, 0, 1, 1, 1, 0)))
    neigh.append(np.array((0, 0, 0, 0, 1, 1, 0, 1, 1, 1)))
    neigh.append(np.array((0, 0, 0, 0, 0, 1, 1, 0, 1, 1)))

    num = softmax_y.size(0)
    a = np.argsort(softmax_y.data.cpu().numpy(), 1)

    _, predicted = torch.max(softmax_y.data.cpu(), 1)
    for i in range(num):
        if predicted[i] == true[i]:
            correct += 1
        if a[i, 8] == true[i]:
            if neigh[true[i]][a[i, 9]] == 1:
                correct += 1

    return correct


def changingFactorAll_Big(factor, allLabels, coord=True):
    returnIndex = []
    allOtherFactor = allLabels
    allOtherFactor = allOtherFactor.tolist()

    for row in allOtherFactor: #deleting the label correponding to factor
        del row[factor]

    if coord == False:
        for row in allOtherFactor:
            del row[6]
            del row[6]
            del row[6]

    uniqueOtherFac, indxOtherFac = np.unique(allOtherFactor, return_inverse=True, axis=0)
    uniqueFac, indxFac = np.unique(allLabels[:, factor], return_inverse=True, axis=0)

    countHowManySameOtherFactor = 0
    lenOfUniqueOtherFac = len(uniqueOtherFac)
    set_t_k2 = set()
    for i in range(lenOfUniqueOtherFac):
        label_i = np.where(indxOtherFac == i)
        if len(label_i) == 0:
            print("label_i == 0 for i=",i)
        factorReturnIndx = []
        if len(label_i[0]) > 1:
            countHowManySameOtherFactor += 1
            factorIndx = []
            for j in range(len(label_i[0])):
                factorIndx.append(indxFac[label_i[0][j]])
            uniqueFacIndx, uniqueFacIndx_idx = np.unique(factorIndx, return_inverse=True)
            if len(uniqueFacIndx) == 0:
                print("len(uniqueFacIndx) == 0")
            if len(uniqueFacIndx) > 1:
                for k in range(len(uniqueFacIndx)):
                    t_k1 = np.where(factorIndx == uniqueFacIndx[k])
                    t_k2 = label_i[0][t_k1[0][0]]
                    factorReturnIndx.append(t_k2)

        set_t_k2.add(len(factorReturnIndx))
        returnIndex.append(factorReturnIndx)
    return returnIndex


def get_optimizer(args, MODEL_SAVE_DIR, model):
    optimizer_type = args.optimizer_type.lower()
    supported_optimizers = ['adam', 'sgd']

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_penalty)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.SGD_decay,
                                    nesterov=args.SGD_nesterov,
                                    momentum=args.SGD_momentum)
    else:
        print("Unsupported optimizer type requested: {}. Supported optimizers: {}".format(optimizer_type,
                                                                                          supported_optimizers))
        return None

    # useful, in case where checkpoint model is trained on GPU and needs to be run on CPU; or vice-versa
    if args.load_optim_from_checkpoint:
        checkpoint = torch.load(MODEL_SAVE_DIR + '/' + args.model_name + '_optimizer.pt')
        optimizer.load_state_dict(checkpoint)
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value, torch.Tensor):
                    state[key] = value.to(args.device)

        print("Optimizer loaded from the checkpoint at: ", MODEL_SAVE_DIR + '/' + args.model_name + '.pt')

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.best_epoch_tolerance, verbose=True, min_lr=1e-8)

    return optimizer, scheduler
