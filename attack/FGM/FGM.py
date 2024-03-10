"""Implementation of gradient based attack methods, FGM, I-FGM, MI-FGM, PGD, etc.
Related paper: CVPR'20 GvG-P,
    https://openaccess.thecvf.com/content_CVPR_2020/papers/Dong_Self-Robust_3D_Point_Recognition_via_Gather-Vector_Guidance_CVPR_2020_paper.pdf
"""

import torch
import numpy as np
import time

class FGM:
    """Class for FGM attack.
    """

    def __init__(self, model, adv_func, budget,
                 dist_metric='l2',threshold = 0.05):
        """FGM attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            budget (float): \epsilon ball for FGM attack
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """

        self.model = model.cuda()
        self.model.eval()

        self.adv_func = adv_func
        self.budget = budget
        self.dist_metric = dist_metric.lower()
        self.threshold = threshold

    def get_norm(self, x):
        """Calculate the norm of a given data x.

        Args:
            x (torch.FloatTensor): [B, 3, K]
        """
        # use global l2 norm here!
        norm = torch.sum(x ** 2, dim=[1, 2]) ** 0.5
        return norm

    def get_gradient(self, data, identity, gt, normalize=True):
        """Generate one step gradient.

        Args:
            data (torch.FloatTensor): batch pc, [B, 3, K]
            target (torch.FloatTensor): target label, [B]
            normalize (bool, optional): whether l2 normalize grad. Defaults to True.
        """
        data = data.float().cuda()
        data.requires_grad_()
        gt = gt.float().cuda()

        # forward pass
        start = time.time()
        transferred = self.model(data, identity)
        end = time.time()
        print('ho')
        print(end-start)

        # backward pass
        loss = self.adv_func(transferred, gt).mean()
        loss.backward()
        with torch.no_grad():
            grad = data.grad.detach()  # [B, 3, K]
            if normalize:
                norm = self.get_norm(grad)
                # print(norm)
                grad = grad / (norm[:, None, None] + 1e-9)
        return grad, transferred,loss.item()

    def attack(self, pose, identity, gt):
        """One step FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.FloatTensor): target label, [B]
        """
        B, _, K = pose.shape
        data = pose.float().cuda().detach()
        # print(data)
        pc = data.clone().detach()
        target = gt.float().cuda().detach()

        # gradient
        normalized_grad, _, loss = self.get_gradient(pc, identity,target)  # [B, 3, K]
        perturbation = normalized_grad * self.budget
        # print(normalized_grad)
        # print(data)
        with torch.no_grad():
            #perturbation = perturbation#.transpose(1, 2).contiguous()
            data = data - perturbation  # no need to clip
            # print(data)
            # test attack performance
            transferred = self.model(data,identity)
            pose_dis = torch.mean((transferred - gt)**2, [1,2])
            success_num = (pose_dis>self.threshold).sum().item()



        print('Successfully attack {}/{}'.format(success_num, data.shape[0]))
        torch.cuda.empty_cache()
        # print(data.detach().cpu().numpy())
        return 0, data.detach().cpu().numpy(), success_num,B,loss,0#normalized_grad.item()

class IFGM(FGM):
    """Class for I-FGM attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter,
                 dist_metric='l2',threshold = 0.05):
        """Iterative FGM attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(IFGM, self).__init__(model, adv_func, budget, dist_metric)
        self.clip_func = clip_func
        self.step_size = step_size
        self.num_iter = num_iter
        self.threshold = threshold

    def attack(self, pose, identity, gt):
        """Iterative FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """
        B, _, K = pose.shape
        data = pose.float().cuda().detach()
        # print(data)
        pc = data.clone().detach()
        target = gt.float().cuda().detach()

        # gradient
        normalized_grad, _, loss = self.get_gradient(pc, identity,target)  # [B, 3, K]

        data = pose.float().cuda().detach()
        pc = data.clone().detach().transpose(1, 2).contiguous()
        pc = pc + torch.randn((B, 3, K)).cuda() * 1e-8
        ori_pc = pc.clone().detach()
        target = gt.float().cuda()
        identity = identity.float().cuda()

        # start iteration
        for iteration in range(self.num_iter):
            # gradient
            normalized_grad, pred = self.get_gradient(pc, target)
            success_num = (pred == target).sum().item()
            if iteration % (self.num_iter // 5) == 0:
                print('iter {}/{}, success: {}/{}'.
                      format(iteration, self.num_iter,
                             success_num, B))
                torch.cuda.empty_cache()
            perturbation = self.step_size * normalized_grad

            # add perturbation and clip
            with torch.no_grad():
                pc = pc - perturbation
                pc = self.clip_func(pc, ori_pc)

        # end of iteration
        with torch.no_grad():
            logits = self.model(pc)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred == target).sum().item()
        print('Final success: {}/{}'.format(success_num, B))
        return pc.transpose(1, 2).contiguous().detach().cpu().numpy(), \
            success_num


class MIFGM(FGM):
    """Class for MI-FGM attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter, mu=1.,
                 dist_metric='l2',threshold = 0.05):
        """Momentum enhanced iterative FGM attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            mu (float): momentum factor
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(MIFGM, self).__init__(model, adv_func,
                                    budget, dist_metric)
        self.clip_func = clip_func
        self.step_size = step_size
        self.num_iter = num_iter
        self.mu = mu
        self.threshold = threshold

    def attack(self, pose, identity, gt):
        """Momentum enhanced iterative FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        pc = data.clone().detach().transpose(1, 2).contiguous()
        pc = pc + torch.randn((B, 3, K)).cuda() * 1e-7
        ori_pc = pc.clone().detach()
        target = target.long().cuda()
        momentum_g = torch.tensor(0.).cuda()

        # start iteration
        for iteration in range(self.num_iter):
            # gradient
            grad, pred = self.get_gradient(pc, target, normalize=False)
            success_num = (pred == target).sum().item()
            if iteration % (self.num_iter // 5) == 0:
                print('iter {}/{}, success: {}/{}'.
                      format(iteration, self.num_iter,
                             success_num, B))
                torch.cuda.empty_cache()

            # grad is [B, 3, K]
            # normalized by l1 norm
            grad_l1_norm = torch.sum(torch.abs(grad), dim=[1, 2])  # [B]
            normalized_grad = grad / (grad_l1_norm[:, None, None] + 1e-9)
            momentum_g = self.mu * momentum_g + normalized_grad
            g_norm = self.get_norm(momentum_g)
            normalized_g = momentum_g / (g_norm[:, None, None] + 1e-9)
            perturbation = self.step_size * normalized_g

            # add perturbation and clip
            with torch.no_grad():
                pc = pc - perturbation
                pc = self.clip_func(pc, ori_pc)

        # end of iteration
        with torch.no_grad():
            logits = self.model(pc)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred == target).sum().item()
        print('Final success: {}/{}'.format(success_num, B))
        return pc.transpose(1, 2).contiguous().detach().cpu().numpy(), \
            success_num


class PGD(IFGM):
    """Class for PGD attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter,
                 dist_metric='l2',threshold = 0.05):
        """PGD attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(PGD, self).__init__(model, adv_func, clip_func,
                                  budget, step_size, num_iter,
                                  dist_metric,threshold)

    def attack(self, pose, identity, gt):
        """PGD attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """
        # the only difference between IFGM and PGD is
        # the initialization of noise
        epsilon = self.budget / \
            ((data.shape[1] * data.shape[2]) ** 0.5)
        init_perturbation = \
            torch.empty_like(data).uniform_(-epsilon, epsilon)
        with torch.no_grad():
            init_data = data + init_perturbation
        return super(PGD, self).attack(init_data, target)
