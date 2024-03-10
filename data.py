import torch.utils.data as data
import torch
import numpy as np
import trimesh
from objLoader_trimesh import trimesh_load_obj
from objLoader_trimesh_animal import trimesh_load_obj_animal
import random
import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)




class SMPLadv_DATA(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False, training_size = 400, server='local'):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.length = training_size
        self.path = './datasets/NPT/npt-data/'
        self.test_label_path = '/home/haoyu/Documents/15_3dattack/CVPR2023/baselines/datasets/NPT/list/supervised_list_obj.txt'
        self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):
        pairs = self.test_list[index]
        id_mesh_name, pose_mesh_name, gt_mesh_name = pairs.split(' ')
        identity_mesh_path = self.path+id_mesh_name
        # print(identity_mesh_path)
        pose_mesh_path =self.path+pose_mesh_name
        gt_mesh_path = self.path+gt_mesh_name


        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)
        gt_mesh=trimesh_load_obj(gt_mesh_path)

        pose_points = pose_mesh.vertices

        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        gt_points = gt_mesh.vertices

        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        gt_points=gt_points-(gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces

        if self.shuffle_point:
            pose_points = pose_points[random_sample2]
            #pose_points = pose_points[:self.pc_sampling,:]
            identity_points=identity_points[random_sample]
            gt_points=gt_points[random_sample]

            face_dict={}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]]=i

            print(len(face_dict))
            new_f=[]

            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_face=np.array(new_f)

        return pose_points, random_sample, gt_points, identity_points, new_face, pose_mesh.faces,pose_mesh_name


    def __len__(self):
        return len(self.test_list)





class ANIMAL_DATA(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='/home/haoyu/Documents/12_NIPS/datasets/animal/'
        self.length = training_size

    def __getitem__(self, index):
        if self.train:
            identity_mesh_path =self.path+'/camel-gallop/camel-gallop-26.obj'
            pose_mesh_path =self.path+'elephant-gallop/elephant-gallop-01.obj'
            gt_mesh_path = self.path+'/camel-gallop/camel-gallop-26.obj'

        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)
        gt_mesh=trimesh_load_obj(gt_mesh_path)

        pose_points = pose_mesh.vertices
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        gt_points = gt_mesh.vertices

        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        gt_points=gt_points-(gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces


        return pose_points, random_sample, gt_points, identity_points, new_face


    def __len__(self):
        if self.train:
            return self.length
        else:
            return len(self.test_list)


class SMPL_DATA(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False, training_size = 400, server='local'):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='./datasets/NPT/npt-data/'
        self.length = training_size
        self.test_label_path = './datasets/NPT/list/supervised_list_obj.txt'
        self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):
        if self.train:
            identity_mesh_i=np.random.randint(0,16)
            identity_mesh_p=np.random.randint(200,600)

            pose_mesh_i=np.random.randint(0,16)
            pose_mesh_p=np.random.randint(200,600)

            identity_mesh_path = self.path+'id'+str(identity_mesh_i)+'_'+str(identity_mesh_p)+'.obj'
            # print(identity_mesh_path)
            pose_mesh_path =self.path+'id'+str(pose_mesh_i)+'_'+str(pose_mesh_p)+'.obj'
            gt_mesh_path = self.path+'id'+str(identity_mesh_i)+'_'+str(pose_mesh_p)+'.obj'



        else:
            pairs = self.test_list[index]
            id_mesh_name, pose_mesh_name, gt_mesh_name = pairs.split(' ')
            identity_mesh_path = self.path+id_mesh_name
            # print(identity_mesh_path)
            pose_mesh_path =self.path+pose_mesh_name
            gt_mesh_path = self.path+gt_mesh_name


        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)
        gt_mesh=trimesh_load_obj(gt_mesh_path)

        pose_points = pose_mesh.vertices
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        gt_points = gt_mesh.vertices

        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        gt_points=gt_points-(gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces

        if self.shuffle_point:
            pose_points = pose_points[random_sample2]
            identity_points=identity_points[random_sample]
            gt_points=gt_points[random_sample]

            face_dict={}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]]=i
            new_f=[]
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_face=np.array(new_f)

        return pose_points, random_sample, gt_points, identity_points, new_face


    def __len__(self):
        if self.train:
            return self.length
        else:
            return len(self.test_list)

class DFAUST_DATA(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='/home/haoyu/Documents/7_3D_animation/BMVC/datasets/DFAUST_sequence_data_all/'
        self.length = training_size

    def __getitem__(self, index):
        if self.train:
            identity_mesh_i=4#np.random.randint(0,12)#1-12train. 13-16 test,
            identity_mesh_p=7#np.random.randint(0,41)#1-14train. rest test,
            identity_mesh_fr=5#np.random.randint(0,30)

            pose_mesh_i=1#np.random.randint(13,16)
            pose_mesh_p=2#np.random.randint(0,8)
            pose_mesh_fr=9#np.random.randint(0,30)

            identity_mesh_path = self.path+str(identity_mesh_i)+'_'+str(identity_mesh_p)+'_'+str(identity_mesh_fr)+'.obj'
            #print(identity_mesh_path)
            pose_mesh_path =self.path+str(pose_mesh_i)+'_'+str(pose_mesh_p)+'_'+str(pose_mesh_fr)+'.obj'
            gt_mesh_path = self.path+str(identity_mesh_i)+'_'+str(pose_mesh_p)+'_'+str(pose_mesh_fr)+'.obj'


        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)
        gt_mesh=trimesh_load_obj(gt_mesh_path)

        pose_points = pose_mesh.vertices
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        gt_points = gt_mesh.vertices

        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        gt_points=gt_points-(gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces

        return pose_points, random_sample, gt_points, identity_points, new_face


    def __len__(self):
        if self.train:
            return self.length
        else:
            return len(self.test_list)


class AMASS_DATA(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False, training_size = 400, video_len = 1):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='/media/haoyu/TITAN/DATASET/AMASS/welldone/AMASS_data_oneshot/'
        self.length = training_size
        self.video_len = video_len

    def __getitem__(self, index):
        if self.train:

            identity_mesh_i=np.random.randint(0,3)#1-12train. 13-16 test,
            identity_mesh_p=4#np.random.randint(0,8)#1-14train. rest test,
            identity_mesh_fr=10#np.random.randint(0,60-self.video_len)

            pose_mesh_i=5#np.random.randint(1,12)
            pose_mesh_p=2#np.random.randint(0,8)
            pose_mesh_fr=np.random.randint(0,60-self.video_len)

        '''load target mesh'''

        identity_mesh_path = self.path+'action/'+str(identity_mesh_i)+'_'+str(identity_mesh_p)+'_'+str(identity_mesh_fr)+'.obj'
        # print('target mesh')
        # print(identity_mesh_path)
        identity_mesh=trimesh_load_obj(identity_mesh_path)
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        #pose_mesh_sequence
        '''load pose and gt sequence meshes'''
        # print('length')
        # print(self.video_len)
        pose_mesh_sequence = np.zeros((self.video_len, self.npoints, 3))
        gt_mesh_sequence = np.zeros((self.video_len, self.npoints,3 ))
        idx = 0
        for frame in range(pose_mesh_fr,pose_mesh_fr + self.video_len):
            pose_mesh_path =self.path+'action/'+str(pose_mesh_i)+'_'+str(pose_mesh_p)+'_'+str(frame)+'.obj'
            gt_mesh_path = self.path+'action/'+str(identity_mesh_i)+'_'+str(pose_mesh_p)+'_'+str(frame)+'.obj'
            # print('source mesh')
            # print(pose_mesh_path)
            # print(gt_mesh_path)

            pose_mesh=trimesh_load_obj(pose_mesh_path)
            gt_mesh=trimesh_load_obj(gt_mesh_path)

            pose_points = pose_mesh.vertices
            gt_points = gt_mesh.vertices

            pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
            gt_points = gt_points - (gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
            # print(pose_points.shape)
            pose_mesh_sequence[idx,:,:] = pose_points
            gt_mesh_sequence[idx,:,:]  = gt_points
            idx +=1

        pose_mesh_sequence = torch.from_numpy(pose_mesh_sequence.astype(np.float32))
        gt_mesh_sequence = torch.from_numpy(gt_mesh_sequence.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces

        if self.shuffle_point:
            #for f in range()
            pose_points = pose_points[random_sample2]


            identity_points=identity_points[random_sample]
            for f in range(self.video_len):
                gt_mesh_sequence[f,:,:]=gt_mesh_sequence[f,:,:][random_sample]

            face_dict={}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]]=i
            new_f=[]
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_face=np.array(new_f)

        return pose_mesh_sequence, random_sample, gt_mesh_sequence, identity_points, new_face


    def __len__(self):
        if self.train:
            return self.length
        else:
            return len(self.test_list)


class AMASS_static_DATA(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='/media/haoyu/TITAN/DATASET/AMASS/welldone/AMASS_data/'
        self.length = training_size

    def __getitem__(self, index):
        if self.train:

            identity_mesh_i=0#np.random.randint(0,4)#1-12train. 13-16 test,
            identity_mesh_p=3#np.random.randint(0,400)#1-14train. rest test,
            identity_mesh_fr=6#np.random.randint(0,400)#1-14train. rest test,

            pose_mesh_i=7#np.random.randint(1,12)
            pose_mesh_p=9#np.random.randint(0,8)
            pose_mesh_fr=28#np.random.randint(0,60-self.video_len)

        '''load target mesh'''

        identity_mesh_path = self.path+'action/'+str(identity_mesh_i)+'_'+str(identity_mesh_p)+'_'+str(identity_mesh_fr)+'.obj'
        # print('target mesh')
        # print(identity_mesh_path)
        identity_mesh=trimesh_load_obj(identity_mesh_path)
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        #pose_mesh_sequence
        '''load pose and gt sequence meshes'''
        # print('length')
        # print(self.video_len)

        pose_mesh_path =self.path+'action/'+str(pose_mesh_i)+'_'+str(pose_mesh_p)+'_'+str(pose_mesh_fr)+'.obj'
        gt_mesh_path = self.path+'action/'+str(identity_mesh_i)+'_'+str(pose_mesh_p)+'_'+str(pose_mesh_fr)+'.obj'
        # print('source mesh')
        # print(pose_mesh_path)
        # print(gt_mesh_path)

        pose_mesh=trimesh_load_obj(pose_mesh_path)
        gt_mesh=trimesh_load_obj(gt_mesh_path)

        pose_points = pose_mesh.vertices
        gt_points = gt_mesh.vertices

        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        gt_points = gt_points - (gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        # print(pose_points.shape)

        pose_points = torch.from_numpy(pose_points.astype(np.float32))
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces


        return pose_points, random_sample, gt_points, identity_points, new_face


    def __len__(self):
        if self.train:
            return self.length
        else:
            return len(self.test_list)




class SMPL_sequence(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False, training_size = 400, video_len = 1):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='/home/haoyu/Documents/7_3D_animation/BMVC/datasets/DFAUST_sequence_data_all/'
        self.length = training_size
        self.video_len = video_len

    def __getitem__(self, index):
        if self.train:

            identity_mesh_i=np.random.randint(0,12)#1-12train. 13-16 test,
            identity_mesh_p=np.random.randint(0,41)#1-14train. rest test,
            identity_mesh_fr=np.random.randint(0,30)

            pose_mesh_i=np.random.randint(13,16)
            pose_mesh_p=np.random.randint(0,8)
            pose_mesh_fr=np.random.randint(0,30-self.video_len)

        '''load target mesh'''

        identity_mesh_path = self.path+str(identity_mesh_i)+'_'+str(identity_mesh_p)+'_'+str(identity_mesh_fr)+'.obj'
        #print('target mesh')
        #print(identity_mesh_path)
        identity_mesh=trimesh_load_obj(identity_mesh_path)
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        #pose_mesh_sequence
        '''load pose and gt sequence meshes'''
        # print('length')
        # print(self.video_len)
        pose_mesh_sequence = np.zeros((self.video_len, self.npoints, 3))
        gt_mesh_sequence = np.zeros((self.video_len, self.npoints,3 ))
        idx = 0
        for frame in range(pose_mesh_fr,pose_mesh_fr + self.video_len):
            pose_mesh_path =self.path+str(pose_mesh_i)+'_'+str(pose_mesh_p)+'_'+str(frame)+'.obj'
            gt_mesh_path = self.path+str(identity_mesh_i)+'_'+str(pose_mesh_p)+'_'+str(frame)+'.obj'
            #print('source mesh')
            #print(pose_mesh_path)
            #print(gt_mesh_path)

            pose_mesh=trimesh_load_obj(pose_mesh_path)
            gt_mesh=trimesh_load_obj(gt_mesh_path)

            pose_points = pose_mesh.vertices
            gt_points = gt_mesh.vertices

            pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
            gt_points = gt_points - (gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
            # print(pose_points.shape)
            pose_mesh_sequence[idx,:,:] = pose_points
            gt_mesh_sequence[idx,:,:]  = gt_points
            idx +=1

        pose_mesh_sequence = torch.from_numpy(pose_mesh_sequence.astype(np.float32))
        gt_mesh_sequence = torch.from_numpy(gt_mesh_sequence.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces

        if self.shuffle_point:
            #for f in range()
            pose_points = pose_points[random_sample2]


            identity_points=identity_points[random_sample]
            gt_points=gt_points[random_sample]

            face_dict={}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]]=i
            new_f=[]
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_face=np.array(new_f)

        return pose_mesh_sequence, random_sample, gt_mesh_sequence, identity_points, new_face


    def __len__(self):
        if self.train:
            return self.length
        else:
            return len(self.test_list)


class MANO_DATA(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='/media/haoyu/TITAN/4_NIPS/mano/handMANOset/handMANOtest/'
        self.length = training_size

    def __getitem__(self, index):
        if self.train:
            identity_mesh_path =self.path+'02.obj'
            pose_mesh_path =self.path+'02.obj'
            gt_mesh_path = self.path+'02.obj'

        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)
        gt_mesh=trimesh_load_obj(gt_mesh_path)

        pose_points = pose_mesh.vertices
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        gt_points = gt_mesh.vertices

        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        gt_points=gt_points-(gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces


        return pose_points, random_sample, gt_points, identity_points, new_face


    def __len__(self):
        if self.train:
            return self.length
        else:
            return len(self.test_list)









class SMG_DATA(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='./datasets/SMG/SMG_3d_rotated/'
        self.length = training_size
        self.test_label_path = './datasets/SMG/MG_list/seen_pose_list.txt'
        self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):
        if self.train:
            identity_mesh_i=np.random.randint(0,35)
            identity_mesh_p=np.random.randint(0,180)

            pose_mesh_i=np.random.randint(0,35)
            pose_mesh_p=np.random.randint(0,180)

            identity_mesh_path = self.path+'id_'+str(identity_mesh_i)+'_pose_'+str(identity_mesh_p)+'.obj'
            # print(identity_mesh_path)
            pose_mesh_path =self.path+'id_'+str(pose_mesh_i)+'_pose_'+str(pose_mesh_p)+'.obj'
            gt_mesh_path = self.path+'id_'+str(identity_mesh_i)+'_pose_'+str(pose_mesh_p)+'.obj'

        else:
            pairs = self.test_list[index]
            id_mesh_name, pose_mesh_name, gt_mesh_name = pairs.split(' ')
            identity_mesh_path = self.path+id_mesh_name
            # print(identity_mesh_path)
            pose_mesh_path =self.path+pose_mesh_name
            gt_mesh_path = self.path+gt_mesh_name


        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)
        gt_mesh=trimesh_load_obj(gt_mesh_path)

        pose_points = pose_mesh.vertices
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        gt_points = gt_mesh.vertices

        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        gt_points=gt_points-(gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces

        if self.shuffle_point:
            pose_points = pose_points[random_sample2]
            identity_points=identity_points[random_sample]
            gt_points=gt_points[random_sample]

            face_dict={}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]]=i
            new_f=[]
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_face=np.array(new_f)

        return pose_points, random_sample, gt_points, identity_points, new_face


    def __len__(self):
        if self.train:
            return self.length
        else:
            return len(self.test_list)



class SMAL_DATA(data.Dataset):
    def __init__(self, train,  npoints=3889, shuffle_point = False, training_size = 16):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='./datasets/SMAL/processed/'
        self.length = training_size
        self.train_label_path = './datasets/SMAL/processed/train_list_lion_cow.txt'
        self.train_list = open(self.train_label_path,'r').read().splitlines()

        self.test_label_path = './datasets/SMG/MG_list/seen_pose_list.txt'
        self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):
        if self.train:
            pairs = self.train_list[index]
            id_mesh_name, pose_mesh_name, gt_mesh_name = pairs.split(' ')

            identity_mesh_path = self.path+id_mesh_name
            # print(identity_mesh_path)
            pose_mesh_path =self.path+pose_mesh_name
            gt_mesh_path = self.path+gt_mesh_name
        else:
            pairs = self.test_list[index]
            id_mesh_name, pose_mesh_name, gt_mesh_name = pairs.split(' ')
            identity_mesh_path = self.path+id_mesh_name
            # print(identity_mesh_path)
            pose_mesh_path =self.path+pose_mesh_name
            gt_mesh_path = self.path+gt_mesh_name


        identity_mesh=trimesh_load_obj_animal(identity_mesh_path)
        pose_mesh=trimesh_load_obj_animal(pose_mesh_path)
        gt_mesh=trimesh_load_obj_animal(gt_mesh_path)

        pose_points = pose_mesh.vertices
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        gt_points = gt_mesh.vertices

        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        gt_points=gt_points-(gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces

        if self.shuffle_point:
            pose_points = pose_points[random_sample2]
            identity_points=identity_points[random_sample]
            gt_points=gt_points[random_sample]

            face_dict={}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]]=i
            new_f=[]
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_face=np.array(new_f)

        return pose_points, random_sample, gt_points, identity_points, new_face


    def __len__(self):
        if self.train:
            return len(self.train_list)
        else:
            return len(self.test_list)


#FAUST
class FAUST_DATA(data.Dataset):
    def __init__(self, train,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='./datasets/FAUST/MPI-FAUST/processed_FAUST/FAUST_body/'
        self.length = training_size
        #self.test_label_path = './datasets/FAUST/FAUST_list/seen_pose_list.txt'
        #self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):
        if self.train:
            identity_mesh_i=np.random.randint(0,5)
            identity_mesh_p=np.random.randint(0,10)

            gt_mesh_i=identity_mesh_i
            gt_mesh_p=0

            identity_mesh_path = self.path+'id'+str(identity_mesh_i)+'_'+str(identity_mesh_p)+'.obj'
            # print(identity_mesh_path)

            gt_mesh_path = self.path+'id'+str(identity_mesh_i)+'_'+str(gt_mesh_p)+'.obj'
        else:
            pairs = self.test_list[index]
            id_mesh_name, pose_mesh_name, gt_mesh_name = pairs.split(' ')
            identity_mesh_path = self.path+id_mesh_name
            # print(identity_mesh_path)
            pose_mesh_path =self.path+pose_mesh_name
            gt_mesh_path = self.path+gt_mesh_name

        #print(identity_mesh_path)
        #print(gt_mesh_path)
        identity_mesh=trimesh_load_obj(identity_mesh_path)
        gt_mesh=trimesh_load_obj(gt_mesh_path)

        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        gt_points = gt_mesh.vertices

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        gt_points=gt_points-(gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        random_sample = np.random.choice(self.npoints,size=self.npoints,replace=False)
        random_sample2 = np.random.choice(self.npoints,size=self.npoints,replace=False)

        new_face=identity_faces

        if self.shuffle_point:
            identity_points=identity_points[random_sample]
            gt_points=gt_points[random_sample]

            face_dict={}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]]=i
            new_f=[]
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_face=np.array(new_f)

        return random_sample, gt_points, identity_points, new_face


    def __len__(self):
        if self.train:
            return self.length
        else:
            return len(self.test_list)
