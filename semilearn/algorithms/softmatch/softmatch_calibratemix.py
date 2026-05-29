# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from .utils import SoftMatchWeightingHook
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
import torch.nn.functional as F
import numpy as np



@ALGORITHMS.register('softmatch_calibratemix')
class SoftMatchCalibrateMix(AlgorithmBase):
    """
        SoftMatch algorithm (https://openreview.net/forum?id=ymt1zQXBDiF&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2023%2FConference%2FAuthors%23your-submissions)).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ema_p (`float`):
                exponential moving average of probability update
        """
    
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

        dataset_size_lb = getattr(args, 'lb_dest_len', 0)
        if dataset_size_lb <= 0:
            dataset_size_lb = 50000
        dataset_size_ulb = getattr(args, 'ulb_dest_len', 0)
        if dataset_size_ulb <= 0:
            dataset_size_ulb = 50000

        device = next(self.model.parameters()).device
        self.apm_delta = getattr(args, 'apm_delta', 0.997)
        self.apm_warmup_iter = getattr(args, 'apm_warmup_iter', 150000)

        self.vector_lb_sum = torch.zeros(dataset_size_lb, device=device)
        self.vector_lb_count = torch.zeros(dataset_size_lb, device=device)
        self.vector_ulb_sum = torch.zeros(
            dataset_size_ulb,
            self.num_classes,
            device=device
        )
        self.vector_ulb_count = torch.zeros(
            dataset_size_ulb,
            self.num_classes,
            device=device
        )

        self.init(T=args.T, hard_label=args.hard_label, dist_align=args.dist_align, dist_uniform=args.dist_uniform, ema_p=args.ema_p, n_sigma=args.n_sigma, per_class=args.per_class)

    def init(self, T, hard_label=True, dist_align=True, dist_uniform=True, ema_p=0.999, n_sigma=2, per_class=False):
        self.T = T
        self.use_hard_label = hard_label
        self.dist_align = dist_align
        self.dist_uniform = dist_uniform
        self.ema_p = ema_p
        self.n_sigma = n_sigma
        self.per_class = per_class

    def update_aum(self, logits_x_lb, y_lb, idx_lb):
        idx = idx_lb.long().to(self.vector_lb_sum.device)
        logits = logits_x_lb.detach()
        y_lb = y_lb.long().to(logits.device)

        top2_vals, top2_idx = torch.topk(logits, k=2, dim=1)
        top1_vals = top2_vals[:, 0]
        top2_vals_ = top2_vals[:, 1]
        top1_idx = top2_idx[:, 0]
        true_logits = logits[torch.arange(logits.size(0), device=logits.device), y_lb]
        margin = torch.where(top1_idx == y_lb, top1_vals - top2_vals_, true_logits - top1_vals)

        self.vector_lb_sum[idx] += margin.to(self.vector_lb_sum.device)
        self.vector_lb_count[idx] += 1

    def get_aum_values(self, idx_lb):
        idx = idx_lb.long().to(self.vector_lb_sum.device)
        return self.vector_lb_sum[idx] / (self.vector_lb_count[idx] + 1e-8)

    def update_apm(self, logits_x_ulb_w, idx_ulb):
        idx = idx_ulb.long().to(self.vector_ulb_sum.device)
        logits = logits_x_ulb_w.detach()

        top2_vals, top2_idx = torch.topk(logits, k=2, dim=1)
        top1_vals = top2_vals[:, 0]
        top2_vals_ = top2_vals[:, 1]
        top1_idx = top2_idx[:, 0]

        pseudo_margin = logits - top1_vals.unsqueeze(1)
        pseudo_margin[torch.arange(logits.size(0), device=logits.device), top1_idx] = top1_vals - top2_vals_
        pseudo_margin = pseudo_margin.to(self.vector_ulb_sum.device)

        alpha = self.apm_delta / (1.0 + self.vector_ulb_count[idx])
        self.vector_ulb_sum[idx] = alpha * pseudo_margin + (1.0 - alpha) * self.vector_ulb_sum[idx]
        self.vector_ulb_count[idx] += 1

    def get_apm_values(self, idx_ulb):
        idx = idx_ulb.long().to(self.vector_ulb_sum.device)
        return self.vector_ulb_sum[idx]

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='uniform' if self.args.dist_uniform else 'model'), 
            "DistAlignHook")
        self.register_hook(SoftMatchWeightingHook(num_classes=self.num_classes, n_sigma=self.args.n_sigma, momentum=self.args.ema_p, per_class=self.args.per_class), "MaskingHook")
        super().set_hooks()    
        
    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):

        def mixup_data(x_easy, y_easy, x_hard, y_hard, mixup_alpha, use_cuda=True):
            
            batch_size_easy = x_easy.size(0)
            batch_size_hard = x_hard.size(0)
            
            mixed_x = torch.empty_like(x_easy)
            y_a = y_easy
            y_b = torch.empty_like(y_easy)
            
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            
            for i in range(batch_size_easy):

                x_easy_flat = x_easy[i].view(1, -1)
                x_hard_flat = x_hard.view(batch_size_hard, -1)
                
                cosine_sim = F.cosine_similarity(x_easy_flat, x_hard_flat, dim=1)
                
                # Get the indices of the top 15 most dissimilar samples (lowest cosine similarity)
                # if you want top 5% dissimilar use 15, since total pseudo labeled data is 224, so 5% of it is roughly 15
                # similarly if you want top 10% use 30, if you want top 15% use 45.

                _, top_15_dissimilar_indices = torch.topk(cosine_sim, 15, largest=False)
                
                random_idx = torch.randint(0, 15, (1,)).item()
                selected_idx = top_15_dissimilar_indices[random_idx]
                
                selected_hard_sample = x_hard[selected_idx]
                
                # Perform mixup
                mixed_x[i] = lam * x_easy[i] + (1 - lam) * selected_hard_sample
                y_b[i] = y_hard[selected_idx]
                    
            return mixed_x, y_a, y_b, lam

        
        def mixup_criterion(logits1, y_a_one_hot, y_b_one_hot, lam):  
            y_a_one_hot = y_a_one_hot.to(torch.float32)  # Ensure correct data type
            y_b_one_hot = y_b_one_hot.to(torch.float32)
            return lam * F.binary_cross_entropy_with_logits(logits1, y_a_one_hot) + (1 - lam) * F.binary_cross_entropy_with_logits(logits1, y_b_one_hot) 
        
        
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)

                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']

                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']

            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}
            sup_loss_partial = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

            # uniform distribution alignment
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                        # make sure this is logits, not dist aligned probs
                                        # uniform alignment in softmatch do not use aligned probs for generating pseudo labels
                                        logits=logits_x_ulb_w,
                                        use_hard_label=self.use_hard_label,
                                        T=self.T)

            # calculate loss
            unsup_loss_partial = self.consistency_loss(logits_x_ulb_s,
                                        pseudo_label,
                                        'ce',
                                        mask=mask)

            loss_1 = sup_loss_partial + self.lambda_u * unsup_loss_partial
            self.update_aum(logits_x_lb.detach(), y_lb, idx_lb)

            if pseudo_label.dim() == 2:
                predicted_classes = pseudo_label.argmax(dim=1)
            else:
                predicted_classes = pseudo_label
            y_ulb_w = predicted_classes

            if self.it <= self.apm_warmup_iter:
                total_loss = loss_1

                out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
                log_dict = self.process_log_dict(sup_loss=sup_loss_partial.item(),
                                                unsup_loss=unsup_loss_partial.item(),
                                                total_loss=total_loss.item(),
                                                util_ratio=mask.float().mean().item())
                return out_dict, log_dict

            num_classes = self.num_classes
            aum_values = self.get_aum_values(idx_lb)
            sorted_lb_vals, sorted_lb_idx = torch.sort(aum_values)
            half_lb = len(sorted_lb_idx) // 2
            hard_indices_tensor_lb = sorted_lb_idx[:half_lb]
            easy_indices_tensor_lb = sorted_lb_idx[half_lb:]

            x_lb_easy = x_lb[easy_indices_tensor_lb]
            y_lb_easy = y_lb[easy_indices_tensor_lb]
            x_lb_hard = x_lb[hard_indices_tensor_lb]
            y_lb_hard = y_lb[hard_indices_tensor_lb]

            self.update_apm(logits_x_ulb_w.detach(), idx_ulb)
            apm_values_all = self.get_apm_values(idx_ulb)
            apm_values = apm_values_all[
                torch.arange(idx_ulb.size(0), device=apm_values_all.device),
                predicted_classes.long().to(apm_values_all.device)
            ]
            sorted_ulb_vals, sorted_ulb_idx = torch.sort(apm_values)
            half_ulb = len(sorted_ulb_idx) // 2
            hard_indices_tensor_ulb = sorted_ulb_idx[:half_ulb]
            easy_indices_tensor_ulb = sorted_ulb_idx[half_ulb:]

            x_ulb_easy = x_ulb_w[easy_indices_tensor_ulb]
            y_ulb_easy = y_ulb_w[easy_indices_tensor_ulb]
            x_ulb_hard = x_ulb_w[hard_indices_tensor_ulb]
            y_ulb_hard = y_ulb_w[hard_indices_tensor_ulb]

            x_1, y_1_1, y_1_2, lam1 = mixup_data(x_lb_easy, y_lb_easy, x_ulb_hard, y_ulb_hard, mixup_alpha=0.4)
            x_2, y_2_1, y_2_2, lam2 = mixup_data(x_lb_hard, y_lb_hard, x_ulb_easy, y_ulb_easy, mixup_alpha=0.4)

            inputs = torch.cat((x_lb, x_1, x_2, x_ulb_w, x_ulb_s))
            outputs = self.model(inputs)
            logits_x_lb = outputs['logits'][:num_lb]
            logits_1 = outputs['logits'][num_lb:(num_lb+len(x_1))]
            logits_2 = outputs['logits'][(num_lb+len(x_1)):(num_lb+len(x_1)+len(x_2))]

            logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][(num_lb+len(x_1)+len(x_2)):].chunk(2)
            feats_x_lb = outputs['feat'][:num_lb]
            feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][(num_lb+len(x_1)+len(x_2)):].chunk(2)

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

            # uniform distribution alignment
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                        # make sure this is logits, not dist aligned probs
                                        # uniform alignment in softmatch do not use aligned probs for generating pseudo labels
                                        logits=logits_x_ulb_w,
                                        use_hard_label=self.use_hard_label,
                                        T=self.T)

            # calculate loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                        pseudo_label,
                                        'ce',
                                        mask=mask)

            #consider a = easy, b = hard. so first mixup = lb easy + ulb hard = a_lb + b_ulb
            y_a_one_hot_lb = F.one_hot(y_1_1, num_classes=num_classes)
            y_b_one_hot_ulb = F.one_hot(y_1_2, num_classes=num_classes)

            y_b_one_hot_lb = F.one_hot(y_2_1, num_classes=num_classes)
            y_a_one_hot_ulb = F.one_hot(y_2_2, num_classes=num_classes)

            mixup_loss_1 = mixup_criterion(logits_1, y_a_one_hot_lb, y_b_one_hot_ulb, lam1)
            mixup_loss_2 = mixup_criterion(logits_2, y_b_one_hot_lb, y_a_one_hot_ulb, lam2)

            loss_2 = mixup_loss_1 + mixup_loss_2  + sup_loss + self.lambda_u * unsup_loss

            total_loss = loss_1 + loss_2

            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            log_dict = self.process_log_dict(sup_loss_partial=sup_loss_partial.item(),
                                            unsup_loss_partial=unsup_loss_partial.item(),
                                            loss_1=loss_1.item(),
                                            sup_loss=sup_loss.item(),
                                            unsup_loss=unsup_loss.item(),
                                            mixup_loss_1 =  mixup_loss_1.item(),
                                            mixup_loss_2 =  mixup_loss_2.item(),
                                            loss_2=loss_2.item(),
                                            total_loss=total_loss.item(),
                                            util_ratio=mask.float().mean().item())

            return out_dict, log_dict


    # TODO: change these
    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        save_dict['prob_max_mu_t'] = self.hooks_dict['MaskingHook'].prob_max_mu_t.cpu()
        save_dict['prob_max_var_t'] = self.hooks_dict['MaskingHook'].prob_max_var_t.cpu()
        save_dict['vector_lb_sum'] = self.vector_lb_sum.cpu()
        save_dict['vector_lb_count'] = self.vector_lb_count.cpu()
        save_dict['vector_ulb_sum'] = self.vector_ulb_sum.cpu()
        save_dict['vector_ulb_count'] = self.vector_ulb_count.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_mu_t = checkpoint['prob_max_mu_t'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_var_t = checkpoint['prob_max_var_t'].cuda(self.args.gpu)
        if 'vector_lb_sum' in checkpoint and 'vector_lb_count' in checkpoint:
            ckpt_sum = checkpoint['vector_lb_sum']
            ckpt_count = checkpoint['vector_lb_count']
            if ckpt_sum.numel() == self.vector_lb_sum.numel() and ckpt_count.numel() == self.vector_lb_count.numel():
                device = self.vector_lb_sum.device
                self.vector_lb_sum = ckpt_sum.to(device)
                self.vector_lb_count = ckpt_count.to(device)
            else:
                self.print_fn('[AUM] checkpoint shape mismatch; keeping initialized AUM vectors')
        if 'vector_ulb_sum' in checkpoint and 'vector_ulb_count' in checkpoint:
            ckpt_ulb_sum = checkpoint['vector_ulb_sum']
            ckpt_ulb_count = checkpoint['vector_ulb_count']
            if ckpt_ulb_sum.numel() == self.vector_ulb_sum.numel() and ckpt_ulb_count.numel() == self.vector_ulb_count.numel():
                device = self.vector_ulb_sum.device
                self.vector_ulb_sum = ckpt_ulb_sum.to(device)
                self.vector_ulb_count = ckpt_ulb_count.to(device)
            else:
                self.print_fn('[APM] checkpoint shape mismatch; keeping initialized APM vectors')
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--dist_uniform', str2bool, True),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--n_sigma', int, 2),
            SSL_Argument('--per_class', str2bool, False),
            SSL_Argument(
                '--apm_delta',
                float,
                0.997,
                'smoothing parameter for EMA update of unlabeled pseudo-margins'
            ),
            SSL_Argument(
                '--apm_warmup_iter',
                int,
                150000,
                'number of iterations before APM-based mixup starts'
            ),
        ]
