# from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import torch
import os
from torch.nn import functional as F
import pickle
import numpy as np

class MeshDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        '''
        root: a folder where the dataset should be stored. This folder is split into raw_dir (raw by default) and processed_dir (processed by default).
        '''
        super().__init__(root, transform, pre_transform)
        # self.raw_files_path = os.path.join(root, "raw")
        # self.build()

    @property
    def raw_file_names(self):
        '''
        return the datasets in raw_dir
        '''
        # path = os.getcwd()
        file_nums = 0
        for root, dirs, files in os.walk(self.raw_dir):
            for each in files:
                file_nums += 1
        return [f'pp{i+1}_labeled.pkl' for i in range(file_nums)]

    @property
    def processed_file_names(self):
        '''
        return a list of files in the processed_dir which needs to be found in order to skip the processing
        '''
        index = 0
        process_list = []
        for raw_path in self.raw_paths:
            process_list.append(f'pp_{index}.pt')
            index += 1

        return process_list

    def process(self):
        '''
        (1) process the raw data and save into processed_dir
        '''
        index = 0
        for raw_path in self.raw_paths:
            file = open(raw_path, 'rb')
            data = pickle.load(file)
            file.close()

            # read features
            one_ring_index = data['one_ring_neighbors']
            f1_gc = data['one_ring_gaussian_curvatures']
            f2_fn = data['one_ring_face_normals']
            f3_vn = data['one_ring_points_normals']
            f4_ang = data['one_ring_angle']
            feature_mat = np.hstack((f1_gc, f2_fn, f3_vn, f4_ang))  # complete features
            label = data['label']

            label = torch.tensor(label).long()
            feature_mat = torch.tensor(feature_mat)

            # read edge index information
            edge_ids = []
            for i in range(one_ring_index.shape[0]):
                for j in range(one_ring_index.shape[1]):
                    if (one_ring_index[i, j] > i):
                        tmp = [i, one_ring_index[i, j]]
                        edge_ids.append(tmp)

            edge_index = torch.tensor(np.asarray(edge_ids).transpose()).long()
            data = Data(x=feature_mat, edge_index=edge_index, y=label)


            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'pp_{index}.pt'))
            index += 1


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'pp_{idx}.pt'))
        return data


if __name__ == '__main__':
    print('this is main code')
    data_path = os.path.join(os.getcwd(), "data")
    train_set = MeshDataset(root=data_path)
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    print('dataset finished')
    # print(train_set[4])