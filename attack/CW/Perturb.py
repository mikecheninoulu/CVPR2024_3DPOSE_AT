"""Implementation of optimization based attack,
    CW Attack for point perturbation.
Based on CVPR'19: Generating 3D Adversarial Point Clouds.
"""

import pdb
import time
import torch
import torch.optim as optim
import numpy as np


class CWPerturb:
    """Class for CW attack.
    """

    def __init__(self, model, adv_func, dist_func, attack_lr=1e-2,
                 init_weight=10., max_weight=80., binary_step=10, num_iter=500, threshold = 0.005,kappa=0.1):
        """CW attack by perturbing points.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            dist_func (function): distance metric
            attack_lr (float, optional): lr for optimization. Defaults to 1e-2.
            init_weight (float, optional): weight factor init. Defaults to 10.
            max_weight (float, optional): max weight factor. Defaults to 80.
            binary_step (int, optional): binary search step. Defaults to 10.
            num_iter (int, optional): max iter num in every search step. Defaults to 500.
        """

        self.model = model.cuda()
        self.model.eval()

        self.adv_func = adv_func
        self.dist_func = dist_func
        self.attack_lr = attack_lr
        self.init_weight = init_weight
        self.max_weight = max_weight
        self.binary_step = binary_step
        self.num_iter = num_iter
        self.threshold = threshold
        self.kappa = kappa

    def attack(self, pose, identity, gt):
        """Attack on given data to target.

        Args:
            data (torch.FloatTensor): victim data, [B, num_points, 3]
            target (torch.FloatTensor): target output, [B]
        """
        B, _, K = pose.shape
        # print(B,K)
        pose = pose.float().cuda().detach()
        #pose = pose.transpose(1, 2).contiguous()
        ori_data = pose.clone().detach()
        ori_data.requires_grad = False
        gt = gt.float().cuda().detach()
        gt_val = gt.detach().cpu().numpy()  # [B]

        # weight factor for budget regularization
        lower_bound = np.zeros((B,))
        upper_bound = np.ones((B,)) * self.max_weight
        current_weight = np.ones((B,)) * self.init_weight

        # record best results in binary search
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, 3, K))

        # perform binary search
        for binary_step in range(self.binary_step):
            # init variables with small perturbation
            adv_data = ori_data.clone().detach() + \
                torch.randn((B, 3, K)).cuda() * 1e-7
            adv_data.requires_grad_()
            bestdist = np.array([1e10] * B)
            bestscore = np.array([-1] * B)
            opt = optim.Adam([adv_data], lr=self.attack_lr, weight_decay=0.)

            adv_loss = torch.tensor(0.).cuda()
            dist_loss = torch.tensor(0.).cuda()

            total_time = 0.
            forward_time = 0.
            backward_time = 0.
            update_time = 0.

            # one step in binary search
            for iteration in range(self.num_iter):
                t1 = time.time()

                # forward passing
                transferred = self.model(adv_data,identity)  # [B, num_classes]

                # transferred = transferred.detach()

                t2 = time.time()
                forward_time += t2 - t1

                # print
                # print((transferred - gt).size())
                # print(((transferred - gt)**2).size())
                # print(torch.mean((transferred - gt)**2, [1,2]).size())
                # print(abc)
                success_num = (torch.mean((transferred - gt)**2, [1,2])>self.threshold).sum().item()

                # print(success_num)
                # if iteration % (self.num_iter // 5) == 0:
                #     print('Step {}, iteration {}, success {}/{}\n'
                #           'adv_loss: {:.4f}, dist_loss: {:.4f}'.
                #           format(binary_step, iteration, success_num, B,
                #                  adv_loss.item(), dist_loss.item()))

                # record values!
                dist_val = torch.sqrt(torch.sum(
                    (adv_data - ori_data) ** 2, dim=[1, 2])).\
                    detach().cpu().numpy()  # [B]
                transferred_val = transferred.detach().cpu().numpy()  # [B]
                input_val = adv_data.detach().cpu().numpy()  # [B, 3, K]


                # update
                for e, (dist, transferred_, gt_, ii) in \
                        enumerate(zip(dist_val, transferred_val, gt_val, input_val)):
                    pose_dis = np.mean((transferred_ - gt_)**2)
                    pose_flag = pose_dis>self.threshold
                    if dist < bestdist[e] and pose_flag:
                        bestdist[e] = dist
                        bestscore[e] = pose_dis
                    if dist < o_bestdist[e] and pose_flag:
                        o_bestdist[e] = dist
                        o_bestscore[e] = pose_dis
                        o_bestattack[e] = ii

                t3 = time.time()
                update_time += t3 - t2

                # compute loss and backward
                adv_loss = self.adv_func(transferred, gt).mean()
                dist_loss = self.dist_func(adv_data, ori_data,
                                           torch.from_numpy(
                                               current_weight)).mean()
                loss = adv_loss + self.kappa*dist_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

                t4 = time.time()
                backward_time += t4 - t3
                total_time += t4 - t1

                if iteration % 100 == 0:
                    # print('total time: {:.2f}, for: {:.2f}, '
                    #       'back: {:.2f}, update: {:.2f}'.
                    #       format(total_time, forward_time,
                    #              backward_time, update_time))
                    total_time = 0.
                    forward_time = 0.
                    backward_time = 0.
                    update_time = 0.
                    torch.cuda.empty_cache()

            # adjust weight factor
            for e, gt_ in enumerate(gt_val):
                if bestscore[e] >0.5 and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e]:
                    # success
                    lower_bound[e] = max(lower_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
                else:
                    # failure
                    upper_bound[e] = min(upper_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.

            torch.cuda.empty_cache()

        # end of CW attack
        # fail to attack some examples
        # just assign them with last time attack data
        fail_idx = (lower_bound == 0.)
        o_bestattack[fail_idx] = input_val[fail_idx]

        # return final results
        success_num = (lower_bound > 0.).sum()
        # print('Successfully attack {}/{}'.format(success_num, B))
        print('Success {}/{}, adv_loss: {:.4f}, dist_loss: {:.4f}'.
              format(success_num, B, adv_loss.item(), dist_loss.item()))
        return o_bestdist, o_bestattack.transpose((0, 2, 1)), success_num,B,adv_loss.item(),dist_loss.item()
