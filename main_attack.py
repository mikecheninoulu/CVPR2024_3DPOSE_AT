"""Targeted point perturbation attack."""

import os


import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import sys
sys.path.append('./')

# import open3d as o3d
import cv2,copy,trimesh
# from open3d import *
from data import SMPL_DATA,SMG_DATA,SMAL_DATA,SMPLadv_DATA
from util.utils import str2bool, set_seed
from attack import CWPerturb
from attack import CrossEntropyAdvLoss, LogitsAdvLoss,PoseEntropyAdvLoss
from attack import FGM, IFGM, MIFGM, PGD
from attack import L2Dist
from tqdm import tqdm
import argparse
import numpy as np
import trimesh
import cv2
import time
import loss_utils as loss_utils

# Training settings
parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--data_root', type=str,
                    default='data/attack_data.npz')
parser.add_argument('--feature_transform', type=str2bool, default=False,
                    help='whether to use STN on features in PointNet')
parser.add_argument('--dataset_name', type=str,default='SMPL-NPT-adv',help='training data set')
parser.add_argument('--batch_size', type=int, default=8, metavar='BS',
                    help='Size of batch')
parser.add_argument('--num_points', type=int, default=1024,
                    help='num of points to use')
parser.add_argument('--adv_func', type=str, default='pose_entropy',
                    choices=['logits', 'cross_entropy','pose_entropy'],
                    help='Adversarial loss function to use')
parser.add_argument('--kappa', type=float, default=1,
                    help='min margin in logits adv loss')
parser.add_argument('--shuffle', type=bool, default=False, help='shuffle mesh points')
parser.add_argument('--attack_lr', type=float, default=1e-3,
                    help='lr in CW optimization')
parser.add_argument('--binary_step', type=int, default=10, metavar='BIS',
                    help='Binary search step')
parser.add_argument('--num_iter', type=int, default=20, metavar='N',
                    help='Number of iterations in each search step')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--train_epoch', type=int,default=40,help='training epoch')
parser.add_argument('--train_size', type=int,default=8,help='training data size')
parser.add_argument('--model_type', type=str, default='model_3D_MAE', metavar='N',
                    choices=['NPT', 'model_3D_MAE','model_3D_MAE_full','max_pool'],
                    help='Model to use, for transferring')
parser.add_argument('--threshold', type=float, default=0.001,
                    help='min margin in to determine a success attack')
parser.add_argument('--adv_type', type=str, default='ifgm',
                    choices=['CWPerturb', 'FGM','IFGM', 'MIFGM','PGD'],
                    help='ad sample to use')
parser.add_argument('--server', type=str, default='local',choices=['local', 'CSC'],
                    help='server to use')
parser.add_argument('--budget', type=float, default=0.0008,help='FGM attack budget')
parser.add_argument('--FGM_num_iter', type=int, default=50,help='IFGM iterate step')
parser.add_argument('--mu', type=float, default=1., help='momentum factor for MIFGM attack')
args = parser.parse_args()

# build model
model_type = args.model_type
if model_type == 'original':
    from model.model import NPT
elif model_type == 'max_pool':
    from model.model_maxpool import NPT
elif model_type == 'max_pool_CGP':
    from model.model_maxpool_CGP import NPT
elif model_type == 'CGP':
    from model.model_CGP import NPT
elif model_type == 'model_3D_MAE':
    from model.model_3D_MAE import NPT
else:
    print('wrong model')
    exit(-1)

model=NPT(num_points=6890)

batch_size = args.batch_size
BEST_WEIGHTS = './pretrain/NPT_maxpool/maxpool.model'
print(args)

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)
cudnn.benchmark = True


# load model weight
state_dict = torch.load(
    BEST_WEIGHTS, map_location='cpu')
print('Loading weight {}'.format(model_type))
try:
    model.load_state_dict(state_dict)
except RuntimeError:
    # eliminate 'module.' in keys
    state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

# distributed mode on multiple GPUs!
# much faster than nn.DataParallel
model = DistributedDataParallel(
    model.cuda(), device_ids=[args.local_rank])

# setup attack settings
if args.adv_func == 'logits':
    adv_func = LogitsAdvLoss(kappa=args.kappa)
elif args.adv_func == 'cross_entropy':
    adv_func = CrossEntropyAdvLoss()
else:
    adv_func = PoseEntropyAdvLoss()
dist_func = L2Dist()

# setup attack type
if args.adv_type == 'CWPerturb':
    # hyper-parameters from their official tensorflow code
    attacker = CWPerturb(model, adv_func, dist_func,
                         attack_lr=args.attack_lr,
                         init_weight=10., max_weight=80.,
                         binary_step=args.binary_step,
                         num_iter=args.num_iter,threshold = args.threshold,kappa=args.kappa)
else:
    delta = args.budget
    args.budget = args.budget * \
        np.sqrt(args.num_points * 3)  # \delta * \sqrt(N * d)
    num_iter = args.FGM_num_iter
    args.step_size = args.budget / float(num_iter)
    if args.adv_type.lower() == 'fgm':
        attacker = FGM(model, adv_func=adv_func,
                       budget=args.budget, dist_metric='l2',threshold = args.threshold)
    elif args.adv_type.lower() == 'ifgm':
        attacker = IFGM(model, adv_func=adv_func,
                        clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                        num_iter=num_iter, dist_metric='l2',threshold = args.threshold)
    elif args.adv_type.lower() == 'mifgm':
        attacker = MIFGM(model, adv_func=adv_func,
                         clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                         num_iter=num_iter, mu=args.mu, dist_metric='l2',threshold = args.threshold)
    elif args.adv_type.lower() == 'pgd':
        attacker = PGD(model, adv_func=adv_func,
                       clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                       num_iter=num_iter, dist_metric='l2',threshold = args.threshold)


# load dataset
dataset_name = args.dataset_name
if dataset_name =='SMPL-NPT':
    dataset = SMPL_DATA(train=True, shuffle_point = args.shuffle, training_size = args.train_size)
elif dataset_name =='SMPL-NPT-adv':
    dataset = SMPLadv_DATA(train=True, shuffle_point = args.shuffle, training_size = args.train_size)
elif dataset_name =='SMG-3D':
    dataset = SMG_DATA(train=True, shuffle_point = args.shuffle, training_size = args.train_size)
elif dataset_name =='COMA':
    dataset = COMA_DATA(train=True, shuffle_point = args.shuffle, training_size = args.train_size)
elif dataset_name =='SMAL':
    dataset = SMAL_DATA(train=True, shuffle_point = args.shuffle, training_size = args.train_size)
else:
    print('wrong dataset')
test_sampler = DistributedSampler(dataset, shuffle=False)
test_loader = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=4,
                         pin_memory=True, drop_last=False,
                         sampler=test_sampler)
lrate=0.00005
lamda=0.0005
optimizer_G = optim.Adam(model.parameters(), lr=lrate)
scheduler = MultiStepLR(optimizer_G, milestones=[300,500,700], gamma=0.1)


# vis = o3d.visualization.Visualizer()
# vis.create_window(visible=True)
# run attack
model.eval()
for j,data in enumerate(test_loader,0):

    pose_points, random_sample, gt_points, identity_points, new_face, pose_face,pose_mesh_name=data



    with torch.no_grad():
        pose_points=pose_points.transpose(2,1)
        pose_points=pose_points.float().cuda(non_blocking=True)

        identity_points=identity_points.transpose(2,1)
        identity_points=identity_points.float().cuda(non_blocking=True)

        gt_points=gt_points.float().cuda(non_blocking=True)
        # vertices4=np.swapaxes(vertices4,0,1)
        # #np.random.shuffle(vertices)
        # #vertices4 = vertices4[0::100, :]
        # random_index4 = np.random.choice(vertices4.shape[0],size=vertices4.shape[0],replace=False)
        # #print(vertices4.shape)
        # #faces = np.array(obj_info1.faces)
        # points_pcd4 = o3d.geometry.PointCloud()
        # points_pcd4.points = o3d.utility.Vector3dVector(vertices4)
        # #mesh_in1.vertex_colors = o3d.utility.Vector3dVector(color)
        # #o3d.visualization.draw_geometries([mesh_in1])
        # vis.add_geometry(points_pcd4)
        # vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
        # #opt.point_size = 1 # Point cloud size
        # vis.update_geometry(points_pcd4)
        # vis.poll_events()
        # vis.update_renderer()

    # attack!

    _,best_pc, _,_,_,_ = attacker.attack(pose_points,identity_points, gt_points)



    for i in range(len(data)):

        mesh = trimesh.Trimesh(vertices=best_pc[i].transpose(1,0),
                       faces=pose_face[i].transpose(1,0),)
        save_path ='./attack_results/'+str(args.adv_type)+'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        mesh.export(save_path+pose_mesh_name[i])

        mesh = trimesh.Trimesh(vertices=pose_points[i].transpose(1,0).detach().cpu().numpy(),
                       faces=pose_face[i].transpose(1,0),)
        save_path ='./attack_results/'+str(args.adv_type)+'/original/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        mesh.export(save_path+pose_mesh_name[i])
    # save results
    # save_path = './attack_results/'+str(args.adv_type)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # save_name = 'Perturb.npz'
    # np.savez(os.path.join(save_path, save_name),
    #          test_pc=best_pc.astype(np.float32))
