import numpy as np
import open3d as o3d
import trimesh as tm
import os
import pickle

################################------Useful functions----##############################################
def find_1_ring_neighbor_face(Adjacency_faces,NumberOfFaces):
    matrix = np.ones((NumberOfFaces, 3),dtype=int)
    for number in range(NumberOfFaces):
        index = np.array(np.where(Adjacency_faces == number))
        row = index[0][:]
        col = index[1][:]
        inves_col = abs(1-col)
        a = Adjacency_faces[row[0]][inves_col[0]]
        b = Adjacency_faces[row[1]][inves_col[1]]
        c = Adjacency_faces[row[2]][inves_col[2]]
        matrix[number,:] = [a , b, c]
        # print(matrix[number,:])
    return matrix

def find_one_ring_vertex_index(triangles,neighbor_index_list,NumberOfFaces):
    matrix = np.ones((NumberOfFaces, 6),dtype=int)
    for center_index in range(NumberOfFaces):
        neighbor_1_index = neighbor_index_list[center_index][0]
        neighbor_2_index = neighbor_index_list[center_index][1]
        neighbor_3_index = neighbor_index_list[center_index][2]
        points_center = triangles[center_index,:]
        points_neighbor_1 = triangles[neighbor_1_index,:]
        points_neighbor_2 = triangles[neighbor_2_index, :]
        points_neighbor_3 = triangles[neighbor_3_index, :]
        tem1 = np.concatenate( (np.setdiff1d(points_center,points_neighbor_1) , np.setdiff1d(points_neighbor_1,points_center)) ,axis = 0  )
        tem2 = np.concatenate( (np.setdiff1d(points_center,points_neighbor_2) , np.setdiff1d(points_neighbor_2,points_center)) ,axis = 0  )
        tem3 = np.concatenate( (np.setdiff1d(points_center,points_neighbor_3) , np.setdiff1d(points_neighbor_3,points_center)) ,axis = 0  )
        tem = np.concatenate((tem1,tem2,tem3),axis=0)
        matrix[center_index,:] = tem
    return matrix

def calculate_face_normal_matrix(Face_normals_list,neighbor_index_list):
    Number_Face = Face_normals_list.shape[0]
    matrix = np.ones((Number_Face, 12))
    for i in range(Number_Face):
        neighbor_1_index = neighbor_index_list[i][0]
        neighbor_2_index = neighbor_index_list[i][1]
        neighbor_3_index = neighbor_index_list[i][2]
        neighbor_1_normals = Face_normals_list[neighbor_1_index, :]
        neighbor_2_normals = Face_normals_list[neighbor_2_index, :]
        neighbor_3_normals = Face_normals_list[neighbor_3_index, :]
        center_normals = Face_normals_list[i, :]
        matrix[i, :] = np.concatenate((center_normals,neighbor_1_normals,neighbor_2_normals,neighbor_3_normals))
    return matrix

def calculate_One_ring_gaussian_curvature_matrix(One_ring_Neighbor_points,NumberOfFaces,curvature_vertices):
    matrix = np.ones((NumberOfFaces, 6))
    for centerface in range(NumberOfFaces):
        matrix[centerface][0] = curvature_vertices[One_ring_Neighbor_points[centerface][0]]
        matrix[centerface][1] = curvature_vertices[One_ring_Neighbor_points[centerface][1]]
        matrix[centerface][2] = curvature_vertices[One_ring_Neighbor_points[centerface][2]]
        matrix[centerface][3] = curvature_vertices[One_ring_Neighbor_points[centerface][3]]
        matrix[centerface][4] = curvature_vertices[One_ring_Neighbor_points[centerface][4]]
        matrix[centerface][5]= curvature_vertices[One_ring_Neighbor_points[centerface][5]]
    return matrix

def calculate_One_ring_points_coor(Vertices,One_ring_Neighbor_points,NumberOfFaces):
    matrix = np.ones((NumberOfFaces, 18))
    for centerface in range(NumberOfFaces):
        point_index = One_ring_Neighbor_points[centerface,:]
        matrix[centerface, :] = np.concatenate(
            (Vertices[point_index[0]], Vertices[point_index[1]], Vertices[point_index[2]],
             Vertices[point_index[3]], Vertices[point_index[4]], Vertices[point_index[5]]))
    return matrix

def calculate_One_ring_points_normals(One_ring_Neighbor_points,NumberOfFaces,vertices_normals):
    matrix = np.ones((NumberOfFaces, 18))
    for centerface in range(NumberOfFaces):
        point_index = One_ring_Neighbor_points[centerface, :]
        matrix[centerface, :] = np.concatenate(
            (vertices_normals[point_index[0]], vertices_normals[point_index[1]], vertices_normals[point_index[2]],
             vertices_normals[point_index[3]], vertices_normals[point_index[4]], vertices_normals[point_index[5]]))
    return matrix

def calculate_label(NumberOfFaces,vertex_color_matrix,triangles):
    #matrix = np.zeros((NumberOfFaces, 1))
    matrix = np.full((NumberOfFaces, 1),0b00)
    ear_points_index_right = np.where(vertex_color_matrix[:,0] == 0.67)
    ear_points_index_right = np.asarray(ear_points_index_right)
    ear_points_index_left = np.where(vertex_color_matrix[:,2] == 1)
    ear_points_index_left = np.asarray(ear_points_index_left)
    for points in range(len(ear_points_index_right[0][:])):
        ear_right = ear_points_index_right[0,points]
        triangle_index = np.array(np.where(triangles == ear_right))
        matrix[triangle_index[0,:],0] = 0b01
    for points in range(len(ear_points_index_left[0][:])):
        ear_left = ear_points_index_left[0,points]
        triangle_index = np.array(np.where(triangles == ear_left))
        matrix[triangle_index[0, :], 0] = 0b10
    return matrix

def calculate_angle(vector_1,vector_2):
    vector_1 = np.array(vector_1)
    vector_2 = np.array(vector_2)
    norm_1 = np.sqrt(vector_1.dot(vector_1))
    norm_2 = np.sqrt(vector_2.dot(vector_2))
    cross = vector_1.dot(vector_2)
    cos_value = cross/(norm_1*norm_2)
    angle = np.arccos(cos_value)
    return angle

def calculate_angle_matrix(Face_normals_list,neighbor_index_list):
    Number_Face = Face_normals_list.shape[0]
    matrix = np.ones((Number_Face,3))
    for i in range(Number_Face):
        neighbor_1_index = neighbor_index_list[i][0]
        neighbor_2_index = neighbor_index_list[i][1]
        neighbor_3_index = neighbor_index_list[i][2]
        center_face_normals = Face_normals_list[i,:]
        neighbor_1_normals = Face_normals_list[neighbor_1_index,:]
        neighbor_2_normals = Face_normals_list[neighbor_2_index, :]
        neighbor_3_normals = Face_normals_list[neighbor_3_index, :]
        angle_1 = calculate_angle(center_face_normals,neighbor_1_normals)
        angle_2 = calculate_angle(center_face_normals, neighbor_2_normals)
        angle_3 = calculate_angle(center_face_normals, neighbor_3_normals)
        matrix[i,:] = [angle_1,angle_2,angle_3]
    return matrix

#####################################-------Data Object------###########################################

class data_set:
    def __init__(self,mesh_tm,mesh):
        self.mesh = mesh
        self.mesh_tm = mesh_tm
#        self.pcd = pcd
        self.Number_Faces = 0  #default
        self.One_ring_neighbor_points = np.ones(((self.Number_Faces, 6))) #default
        self.One_ring_neighbor_face_index = np.ones((self.Number_Faces,3)) #default
        self.One_ring_feature_matrix = np.ones((self.Number_Faces,57)) #default  feature
        self.One_ring_Angle_matrix = np.ones((self.Number_Faces,3)) #default   feature
        self.One_ring_Face_normal_matrix = np.ones((self.Number_Faces,12)) #default         feature
        self.One_ring_gaussian_curvature_matrix = np.ones((self.Number_Faces,6))  #default      feature
        self.One_ring_points_coordinate_matrix = np.ones((self.Number_Faces,18))  #default      feature
        self.One_ring_points_normal_matrix = np.ones((self.Number_Faces,18))  #default      feature
        self.label = np.ones((self.Number_Faces,1))
        self.feature_dict = {}
    def copmute_features(self):
        # Get 1 Ring Neighborhood Faces Index_Matrix
        adjacency_faces = np.asarray(self.mesh_tm.face_adjacency)
        Faces = np.asarray(mesh_tm.faces)
        self.Number_Faces = Faces.shape[0]
        self.One_ring_neighbor_face_index = find_1_ring_neighbor_face(adjacency_faces, self.Number_Faces)
        # Get Triangle 3_points
        triangles = np.asarray(self.mesh.triangles)
        # Get One_ring_Neighbor_Point's index
        self.One_ring_neighbor_points = find_one_ring_vertex_index(triangles, self.One_ring_neighbor_face_index,
                                                                   self.Number_Faces)
        # Get Face Normals list
        mesh.compute_triangle_normals()
        Face_normals_matrix = np.asarray(mesh.triangle_normals)
        # Get Vertex Normals list
        mesh.compute_vertex_normals()
        vertices_normals_matrix = np.asarray(mesh.vertex_normals)
        #Get Vertex Colors
        Colors = np.asarray((mesh.vertex_colors))
        # Get Gaussian Curvature vectors
        Vertices = np.asarray(self.mesh.vertices)
        curvature_vertices = tm.curvature.discrete_gaussian_curvature_measure(mesh_tm, Vertices, 0)

        # Get One_Ring_Angle Feature Matrix
        self.One_ring_Angle_matrix = calculate_angle_matrix(Face_normals_matrix,self.One_ring_neighbor_face_index)  #3 dimension
        # Get One_Ring_Face Normal Feature Matrix
        self.One_ring_Face_normal_matrix = calculate_face_normal_matrix(Face_normals_matrix,self.One_ring_neighbor_face_index)  #12 dimension
        #Get One_Ring gaussian curvature Matrix
        self.One_ring_gaussian_curvature_matrix = calculate_One_ring_gaussian_curvature_matrix(self.One_ring_neighbor_points, self.Number_Faces, curvature_vertices) #6 dimension
        #Get One_Ring_Point_coordinate
        self.One_ring_points_coordinate_matrix = calculate_One_ring_points_coor(Vertices, self.One_ring_neighbor_points, self.Number_Faces)
        #Get One_Ring_Point_normals
        self.One_ring_points_normal_matrix = calculate_One_ring_points_normals(self.One_ring_neighbor_points, self.Number_Faces, vertices_normals_matrix)

        #Get full One_Ring_features
        self.One_ring_feature_matrix = np.concatenate((self.One_ring_Angle_matrix,self.One_ring_Face_normal_matrix,self.One_ring_gaussian_curvature_matrix,
                                                       self.One_ring_points_coordinate_matrix,self.One_ring_points_normal_matrix), axis=1)

        #Get Label
        self.label = calculate_label(self.Number_Faces,np.round(Colors,2),triangles)  #00:head 1:right ear red aa0000 2:left ear blue 0000ff

        #Get dictionary
        self.feature_dict = {'one_ring_neighbors': self.One_ring_neighbor_face_index ,
                             'one_ring_face_normals': self.One_ring_Face_normal_matrix,
                             'one_ring_gaussian_curvatures': self.One_ring_gaussian_curvature_matrix,
                             'one_ring_points_coordinate': self.One_ring_points_coordinate_matrix,
                             'one_ring_points_normals':self.One_ring_points_normal_matrix,
                             'one_ring_angle':self.One_ring_Angle_matrix,
                             'label': self.label}

# if __name__ == "__main__":
#     mesh = o3d.io.read_triangle_mesh('pp1_down.ply')
#     mesh_tm = tm.load_mesh('pp1_down.ply')
#     head = data_set(mesh_tm,mesh)
#     head.copmute_features()
#     # Final DataSet
#     DataSet = head.feature_dict

if __name__ == "__main__":
    # dataset_path_upper = os.getcwd()
    folder_out = os.getcwd() + '\\data\\raw'
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    dataset_path = os.path.join(os.getcwd(), "data_raw_downsampled_ply")

    datasets_path = [os.path.join(dataset_path, file_name) for file_name in os.listdir(dataset_path)]
    for pp in datasets_path:
        mesh = o3d.io.read_triangle_mesh(pp)
        mesh_tm = tm.load_mesh(pp)
        color_matrix = np.asarray(mesh.vertex_colors)
        head = data_set(mesh_tm,mesh)
        head.copmute_features()
        # Final DataSet
        DataSet = head.feature_dict
        # save the dictionary
        _, full_file_name = os.path.split(pp)
        name,_ = os.path.splitext(full_file_name)
        f_save = open(folder_out + '\\' + name + '.pkl', 'wb')
        pickle.dump(DataSet, f_save)
        f_save.close()
    print('finished')