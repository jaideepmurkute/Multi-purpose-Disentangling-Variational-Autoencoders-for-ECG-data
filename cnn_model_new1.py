import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gvae_ecg.commonModels import reparametrize, reparametrize_discrete
SMALL = 1e-16


class VAE_Concrete_Simulated(nn.Module):
    def __init__(self, args, alpha0=10., training_flag=True, temp=1):
        super(VAE_Concrete_Simulated, self).__init__()
        self.temp = temp
        self.dataset = args.dataset_type
        self.bottleneck_size = args.bottleneck_size
        self.learning_type = args.learning_type
        self.model_type = args.model_type.lower()
        self.device = args.device
        self.num_classes = args.num_classes
        self.sample_treatment = args.sample_treatment

        self.kernel_size = 4
        self.kernel_size_sustain = 5
        self.stride = 2
        self.pad = 2
        self.vae_dropout = 0.4  # drop-probability of conv channels in the VAE path ()
        self.classifier_dropout = 0.4  # drop-probability of first FC layer units in the classifier branch
        self.bias_before_bn = False

        if self.dataset == 'clinical':
            self.last_kernel_size = 6
        if self.dataset == 'simulated':
            self.last_kernel_size = 10


        if self.model_type == 'ibp':
            self.params_to_learn = 3
        else:
            self.params_to_learn = 2

        if self.model_type == 'ibp':
            self.training = training_flag
            self.beta_a = 0
            self.beta_b = 0
            self.beta_a = F.softplus(self.beta_a) + 0.01
            self.beta_b = F.softplus(self.beta_b) + 0.01

        # ENCODER -----------------------------------------------
        # channels progression: 1 -> [32, 64, 128, 200]
        self.conv1_sus = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=self.kernel_size_sustain, padding=self.pad,
                                   stride=1, bias=True)

        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=self.kernel_size, padding=self.pad,
                               stride=self.stride, bias=self.bias_before_bn)
        self.bn1 = nn.BatchNorm1d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        self.dropout_vae = nn.Dropout(p=self.vae_dropout)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, padding=self.pad,
                               stride=self.stride, bias=self.bias_before_bn)
        self.bn2 = nn.BatchNorm1d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, padding=self.pad,
                               stride=self.stride, bias=self.bias_before_bn)
        self.bn3 = nn.BatchNorm1d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=self.bottleneck_size*self.params_to_learn,
                               kernel_size=self.kernel_size, padding=self.pad, stride=self.stride,
                               bias=self.bias_before_bn)
        self.bn4 = nn.BatchNorm1d(num_features=self.bottleneck_size*self.params_to_learn, eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)

        self.conv5 = nn.Conv1d(in_channels=self.bottleneck_size*self.params_to_learn,
                               out_channels=self.bottleneck_size * self.params_to_learn,
                               kernel_size=self.kernel_size, padding=self.pad, stride=self.stride,
                               bias=self.bias_before_bn)
        self.bn5 = nn.BatchNorm1d(num_features=self.bottleneck_size * self.params_to_learn, eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)

        self.conv6 = nn.Conv1d(in_channels=self.bottleneck_size*self.params_to_learn,
                               out_channels=self.bottleneck_size * self.params_to_learn,
                               kernel_size=self.last_kernel_size, padding=0, stride=1,
                               bias=self.bias_before_bn)
        self.bn6 = nn.BatchNorm1d(num_features=self.bottleneck_size * self.params_to_learn, eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)

        if self.sample_treatment == 'iid':
            self.conv_mean = nn.Conv1d(in_channels=self.bottleneck_size*self.params_to_learn, out_channels=self.bottleneck_size,
                                       kernel_size=1, padding=0, stride=1, groups=1, bias=True)
            self.conv_logvar = nn.Conv1d(in_channels=self.bottleneck_size*self.params_to_learn, out_channels=self.bottleneck_size,
                                       kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        elif self.sample_treatment == 'group':
            self.conv_mean_group = nn.Conv1d(in_channels=self.bottleneck_size*self.params_to_learn, out_channels=self.bottleneck_size,
                                       kernel_size=1, padding=0, stride=1, groups=1, bias=True)
            #self.conv_mean_group = nn.Linear(in_features=self.bottleneck_size*self.params_to_learn, out_features=self.bottleneck_size)

            self.conv_mean_variations = nn.Conv1d(in_channels=self.bottleneck_size * self.params_to_learn,
                                             out_channels=self.bottleneck_size,
                                             kernel_size=1, padding=0, stride=1, groups=1, bias=True)
            #self.conv_mean_variations = nn.Linear(in_features=self.bottleneck_size * self.params_to_learn,
            #                                 out_features=self.bottleneck_size)

            # Learns logvars
            self.conv_logvar_group = nn.Conv1d(in_channels=self.bottleneck_size*self.params_to_learn, out_channels=self.bottleneck_size,
                                        kernel_size=1, padding=0, stride=1, groups=1, bias=True)
            #self.conv_logvar_group = nn.Linear(in_features=self.bottleneck_size * self.params_to_learn,
            #                                 out_features=self.bottleneck_size)

            self.conv_logvar_variations = nn.Conv1d(in_channels=self.bottleneck_size * self.params_to_learn,
                                              out_channels=self.bottleneck_size,
                                              kernel_size=1, padding=0, stride=1, groups=1, bias=True)
            #self.conv_logvar_variations = nn.Linear(in_features=self.bottleneck_size * self.params_to_learn,
            #                                 out_features=self.bottleneck_size)

        if self.model_type == 'ibp':
            self.conv_bernoulli = nn.Conv1d(in_channels=self.bottleneck_size*self.params_to_learn, out_channels=self.bottleneck_size,
                                            kernel_size=1, padding=0, stride=1, groups=1, bias=True)


        # Classifier ----------------------------
        self.full_conn1 = nn.Linear(in_features=200, out_features=100)
        self.dropout_clf = nn.Dropout(p=self.classifier_dropout)
        self.full_conn2 = nn.Linear(in_features=100, out_features=self.num_classes)

        # IBP -----------------------------------
        if self.model_type == 'ibp':
            a_val = np.log(np.exp(alpha0) - 1)  # inverse softplus
            b_val = np.log(np.exp(1.) - 1)
            self.beta_a = nn.Parameter(torch.Tensor(self.bottleneck_size).zero_() + a_val)
            self.beta_b = nn.Parameter(torch.Tensor(self.bottleneck_size).zero_() + b_val)

        # DECODER -----------------------------
        if (self.learning_type == 'supervised' or self.learning_type == 'baseline') and self.sample_treatment == 'iid':
            self.unconv1 = nn.ConvTranspose1d(in_channels=self.bottleneck_size + self.num_classes, out_channels=self.bottleneck_size*self.params_to_learn,
                                              kernel_size=self.last_kernel_size, padding=0, stride=1, bias=self.bias_before_bn)
            self.unbn1 = nn.BatchNorm1d(num_features=self.bottleneck_size * self.params_to_learn, eps=1e-05,
                                        momentum=0.1, affine=True, track_running_stats=True)
        elif (self.learning_type == 'supervised' or self.learning_type == 'baseline') and self.sample_treatment == 'group':
            self.unconv1 = nn.ConvTranspose1d(in_channels=self.bottleneck_size*self.params_to_learn + self.num_classes,
                                              out_channels=self.bottleneck_size * self.params_to_learn,
                                              kernel_size=self.last_kernel_size, padding=0, stride=1,
                                              bias=self.bias_before_bn)
            self.unbn1 = nn.BatchNorm1d(num_features=self.bottleneck_size * self.params_to_learn, eps=1e-05,
                momentum=0.1, affine=True,  track_running_stats=True)


        elif self.learning_type == 'unsupervised':
            if self.sample_treatment == 'group' or args.model_type == 'ibp':
                self.unconv1 = nn.ConvTranspose1d(in_channels=self.bottleneck_size*self.params_to_learn, out_channels=self.bottleneck_size*self.params_to_learn,
                                                      kernel_size=self.last_kernel_size, padding=0, stride=1, bias=self.bias_before_bn)

                self.unbn1 = nn.BatchNorm1d(num_features=self.bottleneck_size*self.params_to_learn, eps=1e-05, momentum=0.1, affine=True,
                                            track_running_stats=True)
            elif self.sample_treatment == 'iid':
                self.unconv1 = nn.ConvTranspose1d(in_channels=self.bottleneck_size,
                                                  out_channels=self.bottleneck_size * self.params_to_learn,
                                                  kernel_size=self.last_kernel_size, padding=0, stride=1,
                                                  bias=self.bias_before_bn)

                self.unbn1 = nn.BatchNorm1d(num_features=self.bottleneck_size * self.params_to_learn, eps=1e-05,
                                            momentum=0.1, affine=True,
                                            track_running_stats=True)

        self.unconv2 = nn.ConvTranspose1d(in_channels=self.bottleneck_size*self.params_to_learn,
                                          out_channels=self.bottleneck_size*self.params_to_learn,
                                          kernel_size=self.kernel_size, padding=self.pad, stride=self.stride,
                                          bias=self.bias_before_bn)
        self.unbn2 = nn.BatchNorm1d(num_features=self.bottleneck_size*self.params_to_learn, eps=1e-05, momentum=0.1, affine=True,
                                    track_running_stats=True)

        self.unconv3 = nn.ConvTranspose1d(in_channels=self.bottleneck_size * self.params_to_learn,
                                          out_channels=128,
                                          kernel_size=self.kernel_size, padding=self.pad, stride=self.stride,
                                          bias=self.bias_before_bn)
        self.unbn3 = nn.BatchNorm1d(num_features=128, eps=1e-05, momentum=0.1,
                                    affine=True,
                                    track_running_stats=True)

        self.unconv4 = nn.ConvTranspose1d(in_channels=128, out_channels=64,
                                          kernel_size=self.kernel_size, padding=self.pad, stride=self.stride,
                                          bias=self.bias_before_bn)
        self.unbn4 = nn.BatchNorm1d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.unconv5 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=self.kernel_size, padding=self.pad,
                                          stride=self.stride, bias=self.bias_before_bn)
        self.unbn5 = nn.BatchNorm1d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.unconv6 = nn.ConvTranspose1d(in_channels=32, out_channels=12, kernel_size=self.kernel_size, padding=self.pad,
                                          stride=self.stride, bias=self.bias_before_bn, groups=1)

    def adjust_size(self, out, newsize):
        if newsize > out.size()[-1]:
            # if out.size()[-1] % self.kernel_size != 0:
            pad_amount = self.kernel_size - (out.size()[-1] % self.kernel_size)
            pad = torch.zeros(out.size()[0], out.size()[1], pad_amount).double().to(self.device)
            out = torch.cat([out, pad], dim=-1)
        elif newsize < out.size()[-1]:
                out = out[:, :, :newsize - out.size()[-1]]
        return out

    def adjust_size_new(self, out):
        if out.size()[-1] % 2 != 0:
            pad = torch.zeros(out.size()[0], out.size()[1], 1).double().to(self.device)
            out = torch.cat([out, pad], dim=-1)
        return out

    def encode(self, x, zs):
        out = self.adjust_size_new(F.relu(self.bn1(self.conv1(x))))
        out = self.adjust_size_new(self.dropout_vae(F.relu(self.bn2(self.conv2(out)))))
        out = self.adjust_size_new(self.dropout_vae(F.relu(self.bn3(self.conv3(out)))))
        out = self.adjust_size_new(self.dropout_vae(F.relu(self.bn4(self.conv4(out)))))
        out = self.adjust_size_new(self.dropout_vae(F.relu(self.bn5(self.conv5(out)))))
        out = F.relu(self.conv6(out))
        encoding = out.squeeze(2)

        # Learn distribution params
        if self.sample_treatment == 'group':
            mu_group = self.conv_mean_group(out)
            logvar_group = self.conv_logvar_group(out)
            mu_variations = self.conv_mean_variations(out)
            logvar_variations = self.conv_logvar_variations(out)
        elif self.sample_treatment == 'iid':
            mu = self.conv_mean(out)
            logvar = self.conv_logvar(out)

        # Classifier
        out = F.relu(self.dropout_clf(self.full_conn1(encoding)))
        out = self.full_conn2(out)
        logit_y = F.softmax(out, dim=1)

        if self.model_type == 'ibp':
            z = self.conv_bernoulli(out)
            return mu_group, logvar_group, mu_variations, logvar_variations, z, logit_y, encoding
        else:
            if self.sample_treatment == 'group':
                return mu_group, logvar_group, mu_variations, logvar_variations, logit_y, encoding
            elif self.sample_treatment == 'iid':
                return mu, logvar, logit_y, encoding

    def adjust_size_decoding(self, out):
        out = out[:, :, :-1]
        return out

    def decode(self, z_discrete):
        out = F.relu(self.unbn1(self.unconv1(input=z_discrete)))

        out = self.dropout_vae(F.relu(self.unbn2(self.unconv2(out))))
        if self.dataset == 'clinical':
            out = self.adjust_size_decoding(out)

        out = self.dropout_vae(F.relu(self.unbn3(self.unconv3(out))))
        if self.dataset == 'clinical' or self.dataset == 'simulated':
            out = self.adjust_size_decoding(out)

        out = self.dropout_vae(F.relu(self.unbn4(self.unconv4(out))))
        if self.dataset == 'clinical' or self.dataset == 'simulated':
            out = self.adjust_size_decoding(out)

        out = self.dropout_vae(F.relu(self.unbn5(self.unconv5(out))))
        if self.dataset == 'clinical':
            out = self.adjust_size_decoding(out)

        out = self.unconv6(out)
        return out

    def reparametrize_gaussian(self, mu, logvar):
        # noise = Variable(mu.data.clone().normal_(0, 1), requires_grad=False)
        # return mu + (noise * logvar.exp())
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, requires_grad=False)
        return mu + eps * std

    def reparametrize_gaussian_group(self, mu, logvar, zs, training):
        if training:
            std = torch.exp(0.5 * logvar)
        else:
            std = torch.zeros_like(logvar)
        content_samples = []
        indexes = []
        sizes = []
        eps_dict = dict()
        batch_size = zs.size(0)
        groups_within_batch = np.unique(zs.detach().cpu().numpy())
        for i, group in enumerate(groups_within_batch):

            samples_group = zs.eq(group).nonzero()
            size_group = samples_group.numel()
            if size_group > 0:
                eps_dict[group] = torch.FloatTensor(1, logvar.size(1)).normal_().double().to(self.device)
                reparametrized = std[i][None, :] * eps_dict[group] + mu[i][None, :]
                group_content_sample = reparametrized.squeeze(dim=0).repeat((size_group, 1))
                content_samples.append(group_content_sample)
                if size_group == 1:
                    #samples_group = samples_group[None]
                    pass
                indexes.append(samples_group)
                size_group = torch.ones(size_group) * size_group
                sizes.append(size_group)

        content_samples = torch.cat(content_samples, dim=0)
        indexes = torch.cat(indexes)
        sizes = torch.cat(sizes)

        return content_samples, indexes, sizes


    def reparametrize_gaussian_variations(self, mu, logvar, zs, training):
        if training:
            std = torch.exp(0.5 * logvar)
        else:
            std = torch.zeros_like(logvar)
        eps = torch.randn_like(std, requires_grad=False)

        return mu + eps * std

    def accumulate_group_evidence(self, mu_group, logvar_group, zs):
        """
        As of now, not accumulating evidence as mentioned in ML-VAE paper. Using simple averaging, for the reasons
        mentioned in Group-based VAE paper.
        """
        content_mu = []
        content_logvar = []
        list_groups_labels = []
        sizes_group = []
        groups_within_batch = (zs).unique()
        for _, group in enumerate(groups_within_batch):
            group_label = group.item()
            samples_group = zs.eq(group_label).nonzero()
            if samples_group.numel() > 0:
                group_mu = mu_group[samples_group, :].mean(dim=0)
                group_logvar = logvar_group[samples_group, :].mean(dim=0)
            content_mu.append(group_mu)
            content_logvar.append(group_logvar)
            list_groups_labels.append(group_label)
            sizes_group.append(samples_group.numel())

        content_mu = torch.cat(content_mu, dim=0)
        content_logvar = torch.cat(content_logvar, dim=0)
        sizes_group = torch.FloatTensor(sizes_group)

        return content_mu, content_logvar, list_groups_labels, sizes_group


    # crucial - required for ibp
    def logit(self, x):
        return (x + SMALL).log() - (1. - x + SMALL).log()

    def forward(self, x, zs, learning_type, log_prior=None, training=False):
        if self.model_type == 'ibp':
            truncation = self.beta_a.size(0)
            beta_a = F.softplus(self.beta_a) + 0.01
            beta_b = F.softplus(self.beta_b) + 0.01
            batch_size = x.size(0)

            # log_prior might be passed in for IWAE
            if log_prior is None and self.model_type == 'ibp':
                log_prior = reparametrize(
                    beta_a.view(1, truncation).expand(batch_size, truncation),
                    beta_b.view(1, truncation).expand(batch_size, truncation),
                    ibp=True, log=True)
            mu_group, logvar_group, mu_variations, logvar_variations, z, y_logit, _ = self.encode(x)
            logit_post = z + self.logit(log_prior.exp())  # does logit() appear after importing lib???

            logsample = reparametrize_discrete(logit_post, self.temp)
            z_discrete = F.sigmoid(logsample)  # binary
            # zero-temperature rounding
            if not self.training:
                z_discrete = torch.round(z_discrete)
            z_continuous_group = self.reparametrize_gaussian_group(mu_group.squeeze(2), logvar_group.squeeze(2), zs, training)
            z_continuous_variations = self.reparametrize_gaussian_variations(mu_variations.squeeze(2), logvar_variations.squeeze(2), zs, training)
            z_continuous = torch.cat([z_continuous_group, z_continuous_variations])
            z_continuous = z_discrete * z_continuous
        else:
            if self.sample_treatment == 'group':
                mu_group, logvar_group, mu_variations, logvar_variations, y_logit, _ = self.encode(x, zs)
                mu_group = mu_group.squeeze(2)
                logvar_group = logvar_group.squeeze(2)
                mu_variations = mu_variations.squeeze(2)
                logvar_variations = logvar_variations.squeeze(2)
                variations_latent_embeddings = self.reparametrize_gaussian_variations(mu_group, logvar_group, zs,
                                                                                      training)
                variations_latent_embeddings = variations_latent_embeddings.squeeze(1)
                content_mu, content_logvar, list_g, sizes_group = self.accumulate_group_evidence(mu_group, logvar_group,
                                                                                                 zs)
                group_latent_embeddings, indexes, sizes = self.reparametrize_gaussian_group(content_mu, content_logvar,
                                                                                            zs, training)
                z_continuous = torch.cat([variations_latent_embeddings[indexes, :].squeeze(1), group_latent_embeddings],
                                         dim=1)
            elif self.sample_treatment == 'iid':
                mu, logvar, y_logit, _ = self.encode(x, zs)
                mu = mu.squeeze(2)
                logvar = logvar.squeeze(2)
                z_continuous = self.reparametrize_gaussian(mu, logvar)

        if learning_type == 'supervised':
            # y_logit = y_logit.unsqueeze(2)
            inputForDecoder = torch.cat([z_continuous, y_logit], 1)
        elif learning_type == 'unsupervised':
            inputForDecoder = z_continuous

        recon_batch = self.decode(inputForDecoder.unsqueeze(2))
        if self.model_type == 'ibp':
            return mu_group, logvar_group, mu_variations, logvar_variations, y_logit, recon_batch, logsample, logit_post, log_prior, z_discrete, z_continuous

        if self.sample_treatment == 'group':
            return mu_group, logvar_group, mu_variations, logvar_variations, content_mu, content_logvar, indexes, y_logit, recon_batch
        elif self.sample_treatment == 'iid':
            return mu, logvar, y_logit, recon_batch