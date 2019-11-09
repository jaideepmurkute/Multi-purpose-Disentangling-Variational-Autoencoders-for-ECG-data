from torch.autograd import Variable
import torch
from argparse import Namespace
import numpy as np
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from gvae_ecg.utils import elbo_group, elbo_iid, cross_entropy
SMALL = 1e-16


def train_unsupervised_classifier(train_loader, model, dataset_type, optimizer, args):
    xentropy_clf = []
    accuracy_clf = []
    batch_count = 0
    for batch_idx, (data, zs) in enumerate(train_loader):
        batch_count += 1
        if dataset_type == 'simulated':
            zs = (zs[:, 9] - 1).long()
        elif dataset_type == 'clinical':
            zs = zs.long()

        data = Variable(data.double(), requires_grad=False)
        label_oneHot = torch.DoubleTensor(zs.size(0), 10)
        label_oneHot.zero_()
        label_oneHot.scatter_(1, zs.view(-1, 1), 1)
        label = Variable(label_oneHot, requires_grad=False)
        
        data, label = data.to(args.device), label.to(args.device)
        
        if args.model_type == 'ibp':
            mu_group, logvar_group, mu_variations, logvar_variations, z, logit_y, encoding = model.encode(data, zs)
        else:
            if args.sample_treatment == 'group':
                mu_group, logvar_group, mu_variations, logvar_variations, logit_y, encoding = model.encode(data, zs)
            elif args.sample_treatment == 'iid':
                mu, logvar, logit_y, encoding = model.encode(data, zs)

        classifyLoss = cross_entropy(label, logit_y) * args.batch_size

        _, predicted = torch.max(logit_y.data.cpu(), 1)
        accuracy = torch.mean((zs == predicted).float())

        loss = classifyLoss
        optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm_(parameters=model.parameters(), max_norm=0.5, norm_type=2)  # clip grads in-place
        optimizer.step()

        xentropy_clf.append(loss.item())
        accuracy_clf.append(accuracy)

    print("Epoch Summary: ======>")
    print("XEntropy: {:.5f} \t Accuracy: {:.5f}".format(np.mean(xentropy_clf), np.mean(accuracy_clf)))

    return model, xentropy_clf, accuracy_clf, optimizer


def train(args, train_loader, model, optimizer, epoch, scheduler=None):
    model_type = args.model_type.lower()
    model.train()
    model.double()
    architecture_type = args.architecture_type.lower()
    global step

    train_loss = []
    total_elbo_loss = 0
    total_classify_loss = 0
    all_NLL = []
    all_KLD = []
    all_elbo = []
    all_xentropy = []
    all_accuracy = []
    all_elbo_plot = []
    batch_count = 0

    for batch_idx, (data, zs) in enumerate(train_loader):
        batch_count += 1
        
        if args.dataset_type == 'simulated':
            zs = (zs[:, 9] - 1).long()
        elif args.dataset_type == 'clinical':
            zs = zs.long()

        data = Variable(data.double(), requires_grad=False)
        label_oneHot = torch.DoubleTensor(zs.size(0), 10)
        label_oneHot.zero_()
        label_oneHot.scatter_(1, zs.view(-1, 1), 1)
        label = Variable(label_oneHot, requires_grad=False)
        
        data, label = data.to(args.device), label.to(args.device)

        if model_type == 'ibp':
            mu, logvar, logit_y, recon_batch, logsample, logit_post, log_prior, z_discrete, z_continuous = \
                model(data, zs, args.learning_type, training=True)
            # shouldn't a and b be just self.beta_a & self.beta_b  ????????
            a = F.softplus(model.beta_a) + 0.01
            b = (F.softplus(model.beta_b) + 0.01)
            elbo_loss, (NLL, KL_zreal) = elbo_iid(recon_batch, data, mu, logvar, indexes, args.beta, a, b, logsample,
                                              z_discrete, logit_post, log_prior, dataset_size=data.shape[0],
                                              model_type=model_type, args=args, test=False)
        else:
            if args.model_type == 'baseline':
                if args.sample_treatment == 'group':
                    mu_group, logvar_group, mu_variations, logvar_variations, indexes, logit_y, encoding = \
                                        model.encode(data, zs)
                elif args.sample_treatment == 'iid':
                    mu, logvar, logit_y, encoding = model.encode(data, zs)
            else:
                if args.sample_treatment == 'group':
                    mu_group, logvar_group, mu_variations, logvar_variations, content_mu, content_logvar, indexes, \
                        logit_y, recon_batch = model(data, zs, args.learning_type, training=True)
                elif args.sample_treatment == 'iid':
                    mu, logvar, logit_y, recon_batch = model(data, zs, args.learning_type, training=True)
                # if batch_idx == 0:
                #     for lead_number in range(12):
                #         plt.figure()
                #         plt.plot(recon_batch[0, lead_number, :].detach().cpu().numpy(), color='orange')
                #         plt.plot(data[0, lead_number, :].detach().cpu().numpy(), color='green')
                #         # np_data_signal = data[0, 0, :].detach().cpu().numpy()
                #         # np_data_signal = np_data_signal / (np.sqrt(np.sum(np_data_signal**2))/np_data_signal.shape[0])
                #         # plt.plot(np_data_signal, color='blue')
                #         plt.legend(['reconstruction', 'original'])
                #         plt.savefig('./unsup_reconstruction/epoch_'+str(epoch)+'_batch_'+str(batch_idx)+"_"+str(lead_number))
                #         plt.close()
                # # recon_x, x, log_likelihood, mu, logvar, beta, a, b, logsample, z_discrete, logit_post, log_prior, \
                # # dataset_size, model_type = '', args=Namespace(), test=False

                if args.sample_treatment == 'group':
                    elbo_loss, (NLL, KL_zreal) = elbo_group(recon_batch, data, mu_group, logvar_group, mu_variations,
                        logvar_variations, content_mu, content_logvar, indexes, args.beta, a=None, b=None,
                        logsample=None, z_discrete=None, logit_post=None, log_prior=None, dataset_size=None,
                        model_type=model_type, args=args, test=False)
                elif args.sample_treatment == 'iid':
                    elbo_loss, (NLL, KL_zreal) = elbo_iid(recon_batch, data, mu, logvar, args.beta, a=None, b=None,
                        logsample=None, z_discrete=None, logit_post=None, log_prior=None, dataset_size=None,
                        model_type=model_type, args=args, test=False)


        if model_type != 'baseline':
            elbo_plot = elbo_loss.item()
            total_elbo_loss += elbo_loss.item()

        if model_type != 'baseline' and len(list(logit_y.size())) == 3:
            logit_y = logit_y.squeeze(2)

        classifyLoss = cross_entropy(label, logit_y) * args.batch_size

        _, predicted = torch.max(logit_y.data.cpu(), 1)
        accuracy = torch.mean((zs == predicted).float())

        # if model_type != 'baseline':
        #     del mu_group, logvar_group, mu_variations, logvar_variations, recon_batch
        # else:
        #     del mu_group, logvar_group, mu_variations, logvar_variations

        if args.learning_type == 'supervised' and args.model_type == 'other':
            loss = elbo_loss + classifyLoss
        elif args.learning_type == 'supervised' and args.model_type == 'baseline':
            loss = classifyLoss
        elif args.learning_type == 'unsupervised':
            loss = elbo_loss

        optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm_(parameters=model.parameters(), max_norm=0.5, norm_type=2)  # clip grads in-place
        optimizer.step()

        total_classify_loss += classifyLoss.sum()
        all_xentropy.append(classifyLoss.sum().item())
        all_accuracy.append(accuracy)
        train_loss.append(loss.item())

        if model_type != 'baseline':
            all_NLL.append(NLL.item())
            all_KLD.append(KL_zreal.item())
            all_elbo.append(elbo_loss.item())
            all_elbo_plot.append(elbo_plot)
        # if args.use_tensorboard:
        #     #  Tensor-board Logging -----------------------------------------
        #     # 1. Log scalar values (scalar summary)
        #     info = {'loss': loss.item(), 'accuracy': accuracy.item()}
        #
        #     for tag, value in info.items():
        #         logger.scalar_summary(tag, value, step + 1)
        #
        #     # 2. Log values and gradients of the parameters (histogram summary)
        #     for tag, value in model.named_parameters():
        #         tag = tag.replace('.', '/')
        #         logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)
        #         logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)
        #     #  ---------------------------------------------------------------
        # step += 1

        if batch_idx % args.print_log_every == 0:
            print("Batch #{:} performance: ".format(batch_idx))
            if model_type == 'baseline':
                print("XENTROPY: {:.4f}   ACCURACY: {:.4f}".format(all_xentropy[-1], all_accuracy[-1]))
            else:
                print("ELBO: {:.4f}   KLD: {:.4f}   NLL: {:.4f}   XENTROPY: {:.4f}   ACCURACY: {:.4f}".format(
                    all_elbo[-1], all_KLD[-1], all_NLL[-1], all_xentropy[-1], all_accuracy[-1]))

            print()
    if args.use_lr_schedule:
            scheduler.step(metrics=loss)
    
    return np.mean(train_loss), all_accuracy, all_NLL, all_KLD, all_elbo, all_xentropy, all_elbo_plot, optimizer, scheduler

# args, val_loader, model, log_liklihood_VT, epoch
def test(args, model_store_dir, test_loader, model, epoch, store_outputs=False):
    model_type = args.model_type.lower()
    model.eval()
    model.double()
    test_loss = []
    total_classify_loss = 0
    architecture_type = args.architecture_type.lower()
    total_elbo_loss = 0
    all_NLL = []
    all_KLD = []
    all_elbo = []
    all_xentropy = []
    all_accuracy = []
    batch_count = 0

    labels_numpy = None
    predictions_numpy = None
    encoding_numpy = None
    with torch.no_grad():
        for batch_idx, (data, zs) in enumerate(test_loader):
            batch_count += 1
            if args.dataset_type == 'simulated':
                zs = (zs[:, 9] - 1).long()
            elif args.dataset_type == 'clinical':
                zs = zs.long()

            data = Variable(data.double(), requires_grad=False)
            label_oneHot = torch.DoubleTensor(zs.size(0), 10)
            label_oneHot.zero_()
            label_oneHot.scatter_(1, zs.view(-1, 1), 1)
            label = Variable(label_oneHot, requires_grad=False)

            data = data.to(args.device)
            label = label.to(args.device)

            if model_type == 'ibp':
                mu, logvar, logit_y, recon_batch, logsample, logit_post, log_prior, z_discrete, z_continuous = \
                    model(data, zs, args.learning_type, training=False)
                # shouldn't a and b be just self.beta_a & self.beta_b  ????????
                a = F.softplus(model.beta_a) + 0.01
                b = (F.softplus(model.beta_b) + 0.01)
                elbo_loss, (NLL, KL_zreal) = elbo_iid(recon_batch, data, mu, logvar, args.beta, a, b, logsample,
                                                  z_discrete, logit_post, log_prior, dataset_size=data.shape[0],
                                                  model_type='', args=Namespace(), test=True)
                total_elbo_loss += elbo_loss.item()
            elif model_type == 'baseline':
                mu, logvar, logit_y, recon_batch = model.encode(data, zs)
            else:
                mu_group, logvar_group, mu_variations, logvar_variations, content_mu, content_logvar, indexes, logit_y, recon_batch = model(data, zs, args.learning_type, training=False)

                # plt.plot(recon_batch[0, 3, :].detach().cpu().numpy(), color='green')
                # plt.plot(data[0, 3, :].detach().cpu().numpy(), color='red')
                # plt.savefig('./reconstruction_example - first sample - first channel')
                # plt.close()

                if args.sample_treatment == 'iid':
                    elbo_loss, (NLL, KL_zreal) = elbo_iid(recon_batch, data, mu_group, logvar_group, mu_variations,
                        logvar_variations, content_mu, content_logvar, indexes, args.beta, a=0, b=0, logsample=0,
                        z_discrete=0, logit_post=0, log_prior=0, dataset_size=data.shape[0], model_type='',
                        args=Namespace(), test=True)
                elif args.sample_treatment == 'group':
                    elbo_loss, (NLL, KL_zreal) = elbo_iid(recon_batch, data, mu, logvar, args.beta, a=0, b=0, logsample=0,
                        z_discrete=0, logit_post=0, log_prior=0, dataset_size=data.shape[0], model_type='',
                        args=Namespace(), test=True)

                total_elbo_loss += elbo_loss.item()

            if model_type != 'baseline' and len(list(logit_y.size())) == 3:
                logit_y = logit_y.squeeze(2)

            classifyLoss = args.batch_size * cross_entropy(label, logit_y)
            _, predicted = torch.max(logit_y.data.cpu(), 1)
            accuracy = torch.sum(zs == predicted, dtype=torch.float32) / zs.size(0)
            # accuracy = torch.mean((zs == predicted).float())

            total_classify_loss += classifyLoss.sum()
            all_xentropy.append(classifyLoss.sum().data.item())
            all_accuracy.append(accuracy)

            if store_outputs:
                if labels_numpy is None:
                    labels_numpy = label_oneHot.numpy()
                    predictions_numpy = logit_y.detach().cpu().numpy()
                    encoding_numpy = mu_group.detach().cpu().numpy()
                else:
                    labels_numpy = np.concatenate((labels_numpy, label_oneHot.numpy()), axis=0)
                    predictions_numpy = np.concatenate((predictions_numpy, logit_y.detach().cpu().numpy()), axis=0)
                    encoding_numpy = np.concatenate((encoding_numpy, mu_group.detach().cpu().numpy()), axis=0)

            del data, zs, label, mu_group, logvar_group, mu_variations, logvar_variations, logit_y, recon_batch

            if args.learning_type == 'supervised' and model_type == 'other':
                loss = elbo_loss + classifyLoss.sum()
            elif args.learning_type == 'supervised' and model_type == 'baseline':
                loss = classifyLoss
            elif args.learning_type == 'unsupervised':
                loss = elbo_loss

            test_loss.append(loss.data.item())

            if model_type != 'baseline':
                all_NLL.append(NLL.item())
                all_KLD.append(KL_zreal.item())
                all_elbo.append(elbo_loss.item())
                del loss, KL_zreal, elbo_loss, NLL

        if store_outputs:
            labels_numpy = np.array(labels_numpy)
            predictions_numpy = np.array(predictions_numpy)
            np.save(arr=labels_numpy, file=model_store_dir+'/labels.npy')
            np.save(arr=predictions_numpy, file=model_store_dir+'/predictions.npy')

        return np.mean(test_loss), all_accuracy, all_NLL, all_KLD, all_elbo, all_xentropy
