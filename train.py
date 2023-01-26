from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from model_1 import MDC_GCN
from dataset_build import MeshDataset
import os
import torch
import torch.nn.functional as F

import open3d as o3d
import trimesh as tm

import copy
import numpy as np

####################################### initialization ######################################################
LEARNING_RATE = 0.001
NUM_All_HEAD = 5
NUM_EPOCHS = 10
TEST_INDEX  = [1,2,5]

train_index = list(set(range(NUM_All_HEAD)) - (set(TEST_INDEX)))

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

data_path = os.path.join(os.getcwd(), "data")
dataset = MeshDataset(root=data_path)
# dataset = dataset.shuffle()

train_set = dataset[train_index]
test_set = dataset[TEST_INDEX]
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False)


model = MDC_GCN(num_features=57, hidden_channels=1024, num_classes=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

######################################### train and test functions ##############################################
torch.cuda.empty_cache()

def train():
    model.train()
    loss_epoch = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index)
        loss = criterion(out, data.y.squeeze())
        loss.backward()
        loss_epoch += loss
        optimizer.step()
    return loss_epoch


def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x.float(), data.edge_index)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y.squeeze()).sum())
    return correct / (len(test_loader.dataset)*data.y.size(0))

#######################################################################################################
num_epoch = NUM_EPOCHS
for epoch in range(num_epoch):
    loss_train = train()
    if (epoch+1)%10==0 or epoch==0:
        accuracy = test()
        print(f'Epoch {epoch+1:d}: loss of train stage is {loss_train:.2f}, accuracy of test stage is {accuracy*100:.2f}%\n')

#################################### display the result #######################################
torch.cuda.empty_cache()
model.eval()
test_path = os.path.join(os.getcwd(), "data_raw")
test_ply_path = [os.path.join(test_path, file_name) for file_name in os.listdir(test_path)]

for count, data in enumerate(test_loader):
    data = data.to(device)
    out = model(data.x.float(), data.edge_index)
    test_labels = out.argmax(dim=1)
    test_head_label = (test_labels == 0)
    test_right_ear_label = (test_labels == 1)
    test_left_ear_label = (test_labels == 2)

    # for test_ply_path in test_plys_path:
        # test_path = os.path.join(os.getcwd(), "data_raw\\pp5_labeled.ply")
    mesh = o3d.io.read_triangle_mesh(test_ply_path[TEST_INDEX[count]])
    mesh.compute_vertex_normals()

    test_mesh1 = copy.deepcopy(mesh)
    test_mesh1.triangles = o3d.utility.Vector3iVector(np.asarray(test_mesh1.triangles)[test_head_label.cpu(), :])
    test_mesh1.triangle_normals = o3d.utility.Vector3dVector(np.asarray(test_mesh1.triangle_normals)[test_head_label.cpu(), :])

    test_mesh2 = copy.deepcopy(mesh)
    test_mesh2.triangles = o3d.utility.Vector3iVector(np.asarray(test_mesh2.triangles)[test_right_ear_label.cpu(), :])
    test_mesh2.triangle_normals = o3d.utility.Vector3dVector(np.asarray(test_mesh2.triangle_normals)[test_right_ear_label.cpu(), :])

    test_mesh3 = copy.deepcopy(mesh)
    test_mesh3.triangles = o3d.utility.Vector3iVector(np.asarray(test_mesh3.triangles)[test_left_ear_label.cpu(), :])
    test_mesh3.triangle_normals = o3d.utility.Vector3dVector(np.asarray(test_mesh3.triangle_normals)[test_left_ear_label.cpu(), :])

    test_mesh1.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([test_mesh1])

    test_mesh2.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([test_mesh2])

    test_mesh3.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([test_mesh3])
