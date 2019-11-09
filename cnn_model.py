import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from commonModels import reparametrize, reparametrize_discrete

SMALL = 1e-16


class VAE_Concrete_Simulated(nn.Module):
    def __init__(self, args, temp=1., alpha0=10., training_flag=True):
        super(VAE_Concrete_Simulated, self).__init__()
        self.temp = temp
        self.dataset = args.dataset_type
        self.bottleneck_size = args.bottleneck_size
        self.learning_type = args.learning_type
        self.num_classes = args.num_classes
        self.kernel_size = 7
        self.stride = 5
        self.pad = 1
        self.device = args.device
        self.model_type = args.model_type.lower()
        self.vae_dropout = 0.1
        print("IMPORTANT: NEED TO PASS proper training_flag for ibp execution ...")
        if self.model_type == 'ibp':
            self.params_to_learn = 3
        else:
            self.params_to_learn = 2
        if self.model_type == 'ibp':
            self.training = training_flag
            self.beta_a = torch.Tensor([10])
            self.beta_b = torch.Tensor([1])
            self.beta_a = F.softplus(self.beta_a) + 0.01
            self.beta_b = F.softplus(self.beta_b) + 0.01

        # ENCODER -----------------------------------------------
        # channels progression: 1 -> [32, 64, 128, 200, 200]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=self.kernel_size, padding=self.pad,
                               stride=self.stride, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, padding=self.pad,
                               stride=self.stride, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, padding=self.pad,
                               stride=self.stride, bias=True)
        self.bn3 = nn.BatchNorm1d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=self.bottleneck_size*self.params_to_learn,
                               kernel_size=self.kernel_size, padding=self.pad, stride=self.stride, bias=True)
        self.bn4 = nn.BatchNorm1d(num_features=self.bottleneck_size*self.params_to_learn, eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)

        self.conv5 = nn.Conv1d(in_channels=self.bottleneck_size*self.params_to_learn, out_channels=self.bottleneck_size*self.params_to_learn,
                               kernel_size=self.kernel_size, padding=self.pad, stride=1, bias=True)
        self.bn5 = nn.BatchNorm1d(num_features=self.bottleneck_size*self.params_to_learn)

        # Learns means
        self.conv_mean = nn.Conv1d(in_channels=self.bottleneck_size*self.params_to_learn, out_channels=self.bottleneck_size,
                                   kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # Learns logvar
        self.conv_logvar = nn.Conv1d(in_channels=self.bottleneck_size*self.params_to_learn, out_channels=self.bottleneck_size,
                                     kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        if self.model_type == 'ibp':
            self.conv_bernoulli = nn.Conv1d(in_channels=self.bottleneck_size*self.params_to_learn, out_channels=self.bottleneck_size,
                                            kernel_size=1, padding=0, stride=1, groups=1, bias=True)


        # Classifier ----------------------------
        self.full_conn1 = nn.Linear(in_features=self.bottleneck_size*self.params_to_learn, out_features=100)
        self.full_conn2 = nn.Linear(in_features=100, out_features=self.num_classes)
        
        # IBP -----------------------------------
        if self.model_type == 'ibp':
            a_val = np.log(np.exp(alpha0) - 1)  # inverse softplus
            b_val = np.log(np.exp(1.) - 1)
            self.beta_a = nn.Parameter(torch.Tensor(self.bottleneck_size).zero_() + a_val)
            self.beta_b = nn.Parameter(torch.Tensor(self.bottleneck_size).zero_() + b_val)

        # DECODER -----------------------------
        if self.learning_type == 'supervised' or self.learning_type == 'baseline':
            self.unconv1 = nn.ConvTranspose1d(in_channels=self.bottleneck_size+10, out_channels=self.bottleneck_size*self.params_to_learn,
                                              kernel_size=self.kernel_size, padding=self.pad, stride=self.stride,
                                              bias=True)
        elif self.learning_type == 'unsupervised':
            self.unconv1 = nn.ConvTranspose1d(in_channels=self.bottleneck_size, out_channels=self.bottleneck_size*self.params_to_learn,
                                              kernel_size=self.kernel_size, padding=self.pad, stride=self.stride, bias=True)
        self.unbn1 = nn.BatchNorm1d(num_features=self.bottleneck_size*self.params_to_learn, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.unconv2 = nn.ConvTranspose1d(in_channels=self.bottleneck_size*self.params_to_learn, out_channels=128,
                                          kernel_size=self.kernel_size, padding=self.pad, stride=self.stride, bias=True)
        self.unbn2 = nn.BatchNorm1d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.unconv3 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=self.kernel_size, padding=self.pad,
                                          stride=self.stride, bias=True)
        self.unbn3 = nn.BatchNorm1d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.unconv4 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=self.kernel_size, padding=self.pad,
                                          stride=self.stride, bias=True)
        self.unbn4 = nn.BatchNorm1d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.unconv5 = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=self.kernel_size, padding=self.pad,
                                          stride=self.stride, bias=True)

    def encode(self, x):
        # (N,C,H,W) -> (128, 1, 2950)
        x = x.unsqueeze(1)

        # 2950 -> 590
        out = F.dropout(F.relu(self.bn1(self.conv1(x))), p=self.vae_dropout)
        # 590 -> 117
        out = F.dropout(F.relu(self.bn2(self.conv2(out))), p=self.vae_dropout)
        # go from 117 to 120
        if out.size()[2] % 5 != 0:
            pad_amount = 5 - (out.size()[2] % 5)  # make size divisible by 5 - kernel size
            pad = torch.zeros(out.size()[0], out.size()[1], pad_amount).double().cuda()
            out = torch.cat([out, pad], dim=2)

        # 120 -> 24
        out = F.dropout(F.relu(self.bn3(self.conv3(out))), p=self.vae_dropout)
        # go from 24 to 25
        if out.size()[2] % 5 != 0:
            pad_amount = 5 - (out.size()[2] % 5)  # make size divisible by 5 - kernel size
            pad = torch.zeros(out.size()[0], out.size()[1], pad_amount).double().cuda()
            out = torch.cat([out, pad], dim=2)

        # 25 -> 5
        out = F.dropout(F.relu(self.bn4(self.conv4(out))), p=self.vae_dropout)
        # 5 -> 1
        out = F.relu(self.bn5(self.conv5(out)))
        # out = torch.sum(out, dim=2) / 6
        encoding = out.squeeze(2)
        # encoding = out
        # out = out.unsqueeze(2)

        # Learn distribution params
        mu = self.conv_mean(out)
        logvar = self.conv_logvar(out)
        if self.model_type == 'ibp':
            z = self.conv_bernoulli(out)

        # Classifier
        out = F.relu(self.full_conn1(encoding))
        out = self.full_conn2(out)
        logit_y = F.softmax(out, dim=1)

        if self.model_type == 'ibp':
            return mu, logvar, z, logit_y, encoding
        else:
            return mu, logvar, logit_y, encoding


    def decode(self, z_discrete):
        if self.learning_type == 'unsupervised':
            z_discrete = z_discrete.squeeze(2)
        # 1 -> 5
        out = self.unconv1(input=z_discrete)
        out = F.relu(self.unbn1(out))

        # 5 -> 25
        out = F.dropout(F.relu(self.unbn2(self.unconv2(input=out))), p=self.vae_dropout)
        # go from 25 to 24
        out = out[:, :, :-1]

        # 24 -> 120
        out = F.dropout(F.relu(self.unbn3(self.unconv3(out))), p=self.vae_dropout)
        # go from 120 to 117
        out = out[:, :, :-2]

        # 117 -> 590
        out = F.dropout(F.relu(self.unbn4(self.unconv4(out))), p=self.vae_dropout)
        
        # 590 -> 2950
        out = self.unconv5(out)
        return out

    def reparametrize_gaussian(self, mu, logvar):
        # noise = Variable(mu.data.clone().normal_(0, 1), requires_grad=False)
        # return mu + (noise * logvar.exp())
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (mu + eps * std).to(self.device)

    # crucial - required for ibp
    def logit(self, x):
        return (x + SMALL).log() - (1. - x + SMALL).log()

    def forward(self, x, learning_type, log_prior=None):
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

            mu, logvar, z, y_logit, _ = self.encode(x)
            logit_post = z.squeeze(2) + self.logit(log_prior.exp())  # does logit() appear after importing lib???

            logsample = reparametrize_discrete(logit_post, self.temp)
            z_discrete = F.sigmoid(logsample)  # binary
            # zero-temperature rounding
            if not self.training:
                z_discrete = torch.round(z_discrete)
            z_continuous = self.reparametrize_gaussian(mu, logvar)
            z_continuous = z_discrete * z_continuous.squeeze(2)
            z_continuous = z_continuous.unsqueeze(2)
        else:
            mu, logvar, y_logit, _ = self.encode(x)
            z_continuous = self.reparametrize_gaussian(mu, logvar)

        if self.learning_type == 'supervised':
            z_continuous = z_continuous.squeeze(2)
            inputForDecoder = torch.cat([z_continuous, y_logit], 1)
        elif self.learning_type == 'unsupervised':
            inputForDecoder = z_continuous

        inputForDecoder = inputForDecoder.unsqueeze(2)

        recon_batch = self.decode(inputForDecoder)

        if self.model_type == 'ibp':
            return mu, logvar, y_logit, recon_batch, logsample, logit_post, log_prior, z_discrete, z_continuous
        return mu, logvar, y_logit, recon_batch
