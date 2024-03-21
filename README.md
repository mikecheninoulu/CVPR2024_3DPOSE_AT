# CVPR2024_3DPOSE_AT
3DPOSE adversarial training source code for CVPR 2024
"[Towards Robust 3D Pose Transfer with Adversarial Learning]"

# 3DPOSE adversarial training
This is the PyTorch implementation of our CVPR 2024 paper Towards Robust 3D Pose Transfer with Adversarial Learning.
[Haoyu Chen](https://scholar.google.com/citations?user=QgbraMIAAAAJ&hl=en), [Hao Tang](https://github.com/Ha0Tang), [Ehsan Adeli](https://scholar.google.com/citations?user=7NX_J_cAAAAJ&hl=en), [Guoying Zhao](https://scholar.google.com/citations?user=hzywrFMAAAAJ&hl=en). <br>

#### Citation

If you use our code or paper, please consider citing:
```
@inproceedings{chen2024robustposetransfer,
  title={Towards Robust 3D Pose Transfer with Adversarial Learning},
  author={Chen, Haoyu and Tang, Hao and Adeli, Ehsan and Zhao, Guoying},
  booktitle={CVPR},
  year={2024}
}
```

## Dependencies

Requirements:
- python3.6
- numpy
- pytorch==1.1.0 and above
- [trimesh](https://github.com/mikedh/trimesh)

## Dataset preparation
We use the SMPL-NPT dataset provided by NPT, please download data from this link http://www.sdspeople.fudan.edu.cn/fuyanwei/download/NeuralPoseTransfer/data/, 

## Run (to be done, still cleaning up the code)
run the training:
```
python main_attack.py
```

Attacks are in "attack" folder, to check different attacks.



## Acknowledgement
Part of our code is based on 

3D transfer: [NPT](https://github.com/jiashunwang/Neural-Pose-Transfer)，

Transformer framework: (https://github.com/lucidrains/vit-pytorch) 

Meshattack framework: (https://github.com/cuge1995/Mesh-Attack) 

Many thanks!

## License
MIT-2.0 License
