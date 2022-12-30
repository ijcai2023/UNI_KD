import torch
import torch.nn as nn
import numpy as np

from models.models import classifier, ReverseLayerF, Discriminator, RandomLayer, Discriminator_CDAN, \
    codats_classifier, Discriminator_fea, Adapter,Discriminator_t
from models.loss import MMD_loss, CORAL, ConditionalEntropyLoss, VAT, LMMD_loss, HoMM_loss
from utils import EMA

from torch.autograd import Variable


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class Lower_Upper_bounds(Algorithm):
    """
    Lower bound: train on source and test on target.
    Upper bound: train on target and test on target.
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(Lower_Upper_bounds, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, src_x, src_y):
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)

        loss = src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Src_cls_loss': src_cls_loss.item()}


class MMDA(Algorithm):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(MMDA, self).__init__(configs)

        self.mmd = MMD_loss()
        self.coral = CORAL()
        self.cond_ent = ConditionalEntropyLoss()

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, src_x, src_y, trg_x):
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)

        trg_feat = self.feature_extractor(trg_x)

        coral_loss = self.coral(src_feat, trg_feat)
        mmd_loss = self.mmd(src_feat, trg_feat)
        cond_ent_loss = self.cond_ent(trg_feat)

        loss = self.hparams["coral_wt"] * coral_loss + \
               self.hparams["mmd_wt"] * mmd_loss + \
               self.hparams["cond_ent_wt"] * cond_ent_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Coral_loss': coral_loss.item(), 'MMD_loss': mmd_loss.item(),
                'cond_ent_wt': cond_ent_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class DANN(Algorithm):
    """
    DANN: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DANN, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.hparams = hparams
        self.device = device

    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        trg_feat = self.feature_extractor(trg_x)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # Domain classification loss
        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.domain_classifier(src_feat_reversed)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.domain_classifier(trg_feat_reversed)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        # Total domain loss
        domain_loss = src_domain_loss + trg_domain_loss

        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
               self.hparams["domain_loss_wt"] * domain_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class CDAN(Algorithm):
    """
    CDAN: https://arxiv.org/abs/1705.10667
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(CDAN, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator_CDAN(configs)
        self.random_layer = RandomLayer([configs.features_len * configs.final_out_channels, configs.num_classes],
                                        configs.features_len * configs.final_out_channels)

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # hparams
        self.hparams = hparams
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.device = device

    def update(self, src_x, src_y, trg_x):
        # prepare true domain labels
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

        # source features and predictions
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # target features and predictions
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # concatenate features and predictions
        feat_concat = torch.cat((src_feat, trg_feat), dim=0)
        pred_concat = torch.cat((src_pred, trg_pred), dim=0)

        # Domain classification loss
        feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
        disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # update Domain classification
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()

        # prepare fake domain labels for training the feature extractor
        domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

        # Repeat predictions after updating discriminator
        feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
        disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
        # loss of domain discriminator according to fake labels

        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # conditional entropy loss.
        loss_trg_cent = self.criterion_cond(trg_pred)

        # total loss
        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["cond_ent_wt"] * loss_trg_cent

        # update feature extractor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                'cond_ent_loss': loss_trg_cent.item()}


class DIRT(Algorithm):
    """
    DIRT-T: https://arxiv.org/abs/1802.08735
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DIRT, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # hparams
        self.hparams = hparams

        # criterion
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.vat_loss = VAT(self.network, device).to(device)

        # device for further usage
        self.device = device

        self.ema = EMA(0.998)
        self.ema.register(self.network)

    def update(self, src_x, src_y, trg_x):
        # prepare true domain labels
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # target features and predictions
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # concatenate features and predictions
        feat_concat = torch.cat((src_feat, trg_feat), dim=0)

        # Domain classification loss
        disc_prediction = self.domain_classifier(feat_concat.detach())
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # update Domain classification
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()

        # prepare fake domain labels for training the feature extractor
        domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

        # Repeat predictions after updating discriminator
        disc_prediction = self.domain_classifier(feat_concat)

        # loss of domain discriminator according to fake labels
        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # conditional entropy loss.
        loss_trg_cent = self.criterion_cond(trg_pred)

        # Virual advariarial training loss
        loss_src_vat = self.vat_loss(src_x, src_pred)
        loss_trg_vat = self.vat_loss(trg_x, trg_pred)
        total_vat = loss_src_vat + loss_trg_vat
        # total loss
        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["cond_ent_wt"] * loss_trg_cent + self.hparams["vat_loss_wt"] * total_vat

        # update exponential moving average
        self.ema(self.network)

        # update feature extractor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                'cond_ent_loss': loss_trg_cent.item()}


class HoMM(Algorithm):
    """
    HoMM: https://arxiv.org/pdf/1912.11976.pdf
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(HoMM, self).__init__(configs)

        self.coral = CORAL()

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.HoMM_loss = HoMM_loss()

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # calculate lmmd loss
        domain_loss = self.HoMM_loss(src_feat, trg_feat)

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'HoMM_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class DDC(Algorithm):
    """
    DDC: https://arxiv.org/abs/1412.3474
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DDC, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.mmd_loss = MMD_loss()

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # calculate mmd loss
        domain_loss = self.mmd_loss(src_feat, trg_feat)

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class CoDATS(Algorithm):
    """
    CoDATS: https://arxiv.org/pdf/2005.10996.pdf
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(CoDATS, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        # we replace the original classifier with codats the classifier
        # remember to use same name of self.classifier, as we use it for the model evaluation
        self.classifier = codats_classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.hparams = hparams
        self.device = device

    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        trg_feat = self.feature_extractor(trg_x)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # Domain classification loss
        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.domain_classifier(src_feat_reversed)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.domain_classifier(trg_feat_reversed)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        # Total domain loss
        domain_loss = src_domain_loss + trg_domain_loss

        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
               self.hparams["domain_loss_wt"] * domain_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class UDA_KD(Algorithm):
    """
    AdvCDKD
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(UDA_KD, self).__init__(configs)
        from models import models
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.data_domain_classifier_t = Discriminator_t(configs)

        self.data_domain_classifier = Discriminator(configs)
        self.feature_domain_classifier = Discriminator_fea(configs)
        self.adapter = Adapter(configs)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.adapter.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.optimizer_disc = torch.optim.Adam(
            self.data_domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.optimizer_feat = torch.optim.Adam(
            self.feature_domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]


    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        self.network_t.eval()

        real_label = 1
        fake_label = 0

        ########################################################
        # (1) update D network: maximize log(D(fea_t)) + log(1-D(G(x))
        # fea_t: feature from teacher network
        # x: input data
        # G(x): feature from student network
        ########################################################
        self.optimizer_feat.zero_grad()

        # Format Batch
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_feat_t = Variable(trg_feat_t, requires_grad=False)

        f_domain_label = torch.full((src_x.shape[0]+trg_x.shape[0],), real_label, dtype=torch.float, device=self.device)

        # Forward pass real batch through D
        output = self.feature_domain_classifier(torch.concat((src_feat_t,trg_feat_t),dim=0)).view(-1)
        # Calculate loss on all-real batch
        errD_real = nn.BCELoss()(output, f_domain_label)
        # Calculate gradients for D in backward pass
        errD_real.backward()

        # Train with all-fake batch, Generate fake features with G
        # Student Forward
        src_feat = self.feature_extractor(src_x)
        src_feat_hint = self.adapter(src_feat)

        trg_feat = self.feature_extractor(trg_x)
        trg_feat_hint = self.adapter(trg_feat)

        f_domain_label.fill_(fake_label)
        # Classify all fake batch with D

        fea_hint = torch.concat((src_feat_hint,trg_feat_hint),dim=0).detach()

        output = self.feature_domain_classifier(fea_hint).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = nn.BCELoss()(output, f_domain_label)
        # Calculate the gradients for this batch
        errD_fake.backward()

        # Add the gradients from all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizer_feat.step()

        ########################################################
        # (2) update G network: maximize log(D(G(x))
        ########################################################

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        src_pred_t = self.t_classifier(src_feat_t)
        src_pred = self.classifier(src_feat)

        trg_pred_t = self.t_classifier(src_feat_t)
        trg_pred = self.classifier(src_feat)

        # fake labels are real for generator cost
        f_domain_label.fill_(real_label)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.feature_domain_classifier(fea_hint).view(-1)

        # Calculate G's loss based on this output
        errG = nn.BCELoss()(output, f_domain_label)
        errL1 = nn.L1Loss()(src_feat_hint, src_feat_t) + nn.L1Loss()(trg_feat_hint,trg_feat_t)
        errG = errG + errL1

        # Domain classification loss
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        trg_feat = self.feature_extractor(trg_x)

        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.data_domain_classifier(src_feat_reversed)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        src_dis_pred_t = torch.nn.functional.softmax(src_domain_pred, dim=1)
        weights_src = 1 - torch.abs(src_dis_pred_t[:, 0] - src_dis_pred_t[:, 1])

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.data_domain_classifier(trg_feat_reversed)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        trg_dis_pred_t = torch.nn.functional.softmax(trg_domain_pred,dim=1)
        weights_trg = 1 - torch.abs(trg_dis_pred_t[:, 0] - trg_dis_pred_t[:, 1])

        # Total domain loss
        domain_loss = src_domain_loss + trg_domain_loss

        # Add KD loss
        # Normal Knowledge
        # soft_loss = torch.nn.functional.kl_div(src_pred_s_soften, src_pred_t_soften, reduction='batchmean', log_target=True)\
        #             + torch.nn.functional.kl_div(trg_pred_s_soften, trg_pred_t_soften, reduction='batchmean', log_target=True)

        # Disentangled Knowledge
        soft_loss_skd = torch.nn.functional.softmax(src_pred_t/self.temperature, dim=1) * \
                     (torch.log(torch.nn.functional.softmax(src_pred_t/self.temperature, dim=1))
                      - torch.nn.functional.log_softmax(src_pred/self.temperature, dim=1))
        soft_loss_skd = (soft_loss_skd.sum(dim=1) * weights_src).sum(dim=0) / src_pred.size(0)
        soft_loss_tkd = torch.nn.functional.softmax(trg_pred_t/self.temperature, dim=1) * \
                     (torch.log(torch.nn.functional.softmax(trg_pred_t/self.temperature, dim=1))
                      - torch.nn.functional.log_softmax(trg_pred/self.temperature, dim=1))
        soft_loss_tkd = (soft_loss_tkd.sum(dim=1) * weights_trg).sum(dim=0) / trg_pred.size(0)
        soft_loss = soft_loss_skd + soft_loss_tkd

        kd_loss = soft_loss * self.temperature ** 2

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        import math
        g = math.log10(0.9/0.1) / self.hparams["num_epochs"]
        beta = 0.1 * math.exp(g*epoch)

        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + (1-beta)* self.hparams["domain_loss_wt"] * domain_loss \
               + beta * kd_loss + errG

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                'KD_loss':kd_loss.item(), 'errD': errD.item(), 'errG':errG.item()}


class JointUKD(Algorithm):
    """
    JointUKD
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(JointUKD, self).__init__(configs)
        from models import models
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.network_t.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]


    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.network_t.train()

        # Teacher inference on Source and Target
        src_feat_t = self.t_feature_extractor(src_x)
        src_pred_t = self.t_classifier(src_feat_t)
        src_pred_t_soften = torch.nn.functional.log_softmax(src_pred_t/self.temperature,dim=1)

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)

        # Student inference on Source and Target
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)
        src_pred_s_soften = torch.nn.functional.log_softmax(src_pred / self.temperature, dim=1)

        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)
        trg_pred_s_soft = torch.nn.functional.log_softmax(trg_pred / self.temperature, dim=1)

        from mmd import MMD_loss
        mmd_loss = MMD_loss()(src_feat_t,trg_feat_t)
        loss_ce_t = self.cross_entropy(src_pred_t, src_y)
        loss_tda = mmd_loss + 0.8 * loss_ce_t

        loss_tkd = torch.nn.functional.kl_div(trg_pred_s_soft, trg_pred_t_soften, reduction='batchmean', log_target=True)

        loss_kd_src = torch.nn.functional.kl_div(src_pred_s_soften, src_pred_t_soften, reduction='batchmean', log_target=True)
        loss_ce_s = self.cross_entropy(src_pred, src_y)

        loss_skd = loss_kd_src + 0.8 * loss_ce_s
        import math
        g = math.log10(0.9/0.1) / self.hparams["num_epochs"]
        beta = 0.1 * math.exp(g*epoch)
        loss = (1-beta) * loss_tda + beta*(loss_skd+loss_tkd)
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'loss_tda': loss_tda.item(), 'loss_skd': loss_skd.item(), 'loss_tkd':loss_tkd.item()}


class AAD(Algorithm):
    """
    AAD
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(AAD, self).__init__(configs)
        from models import models
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.data_domain_classifier = Discriminator(configs)
        self.feature_domain_classifier = Discriminator_fea(configs)
        self.adapter = Adapter(configs)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.adapter.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.optimizer_disc = torch.optim.Adam(
            self.data_domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.optimizer_feat = torch.optim.Adam(
            self.feature_domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]


    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        self.network_t.eval()

        real_label = 1
        fake_label = 0

        ########################################################
        # (1) update D network: maximize log(D(fea_t)) + log(1-D(G(x))
        # fea_t: feature from teacher network
        # x: input data
        # G(x): feature from student network
        ########################################################
        self.optimizer_feat.zero_grad()

        # Format Batch
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        f_domain_label = torch.full((src_x.shape[0],), real_label, dtype=torch.float, device=self.device)

        # Forward pass real batch through D
        output = self.feature_domain_classifier(src_feat_t).view(-1)
        # Calculate loss on all-real batch
        errD_real = nn.BCELoss()(output, f_domain_label)
        # Calculate gradients for D in backward pass
        errD_real.backward()

        # Train with all-fake batch, Generate fake features with G
        # Student Forward
        src_feat = self.feature_extractor(src_x)
        src_feat_hint = self.adapter(src_feat)

        f_domain_label.fill_(fake_label)
        # Classify all fake batch with D

        output = self.feature_domain_classifier(src_feat_hint.detach()).view(-1)
        # output = self.feature_domain_classifier(src_feat.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = nn.BCELoss()(output, f_domain_label)
        # Calculate the gradients for this batch
        errD_fake.backward()

        # Add the gradients from all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizer_feat.step()

        ########################################################
        # (2) update G network: maximize log(D(G(x))
        ########################################################

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        src_pred_t = self.t_classifier(src_feat_t)
        src_pred_t_soften = torch.nn.functional.log_softmax(src_pred_t / self.temperature, dim=1)


        src_pred = self.classifier(src_feat)
        src_pred_s_soften = torch.nn.functional.log_softmax(src_pred / self.temperature, dim=1)

        # fake labels are real for generator cost
        f_domain_label.fill_(real_label)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.feature_domain_classifier(src_feat_hint).view(-1)

        # Calculate G's loss based on this output
        errG = nn.BCELoss()(output, f_domain_label)

        # Add KD loss
        soft_loss = torch.nn.functional.kl_div(src_pred_s_soften, src_pred_t_soften, reduction='batchmean', log_target=True)
        kd_loss = soft_loss * self.temperature ** 2

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)


        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["soft_loss_wt"] * kd_loss + self.hparams ['errG'] * errG

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        return {'Total_loss': loss.item(), 'Src_cls_loss': src_cls_loss.item(), 'KD_loss':kd_loss.item(), 'errD': errD.item(), 'errG':errG.item() }


class MobileDA(Algorithm):
    """
    MobileDA
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(MobileDA, self).__init__(configs)
        from models import models
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        self.coral = CORAL()

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]


    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.network_t.eval()

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)

        # Student inference on Source and Target
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)
        trg_pred_s_soft = torch.nn.functional.log_softmax(trg_pred / self.temperature, dim=1)

        loss_ce_s = self.cross_entropy(src_pred, src_y)
        loss_soft = torch.nn.functional.kl_div(trg_pred_s_soft, trg_pred_t_soften, reduction='batchmean',
                                              log_target=True)

        loss_dc = self.coral(src_feat, trg_feat)

        loss = loss_ce_s + 0.7* loss_soft + 0.3 * loss_dc
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'loss_ce': loss_ce_s.item(), 'loss_soft': loss_soft.item(), 'loss_dc':loss_dc.item()}


