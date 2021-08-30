import open3d as o3d
import numpy as np
import struct


### Software for creating a mesh from a point cloud and visualizing it ###

## This software uses single chip mmWaveRadar point clouds from the ColoRadar dataset(https://arpg.github.io/coloradar//)

# Define point cloud file path and default file
inp = ".../sing_lechip/pointclouds/data"
dataname = "radar_pointcloud_1.bin"

# Loads a point cloud from a binary file and returns it as an o3d point cloud object
def getPointcloud(d_name=dataname):

    filename = inp + d_name
    with open(filename, mode='rb') as file:
        cloud_bytes = file.read()

    # convert point cloud to appropriate format
    cloud_vals = struct.unpack(str(len(cloud_bytes) // 4) + 'f', cloud_bytes)
    cloud_vals = np.array(cloud_vals)

    # point cloud matrix with one point per row (x,y,z,intensity,doppler)
    cloud = cloud_vals.reshape((-1, 5))

    # take only xyz
    xyz = cloud[:,0:3]

    # create an open3d point cloud data object and define the points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # normal estimation (needed when creating Poisson mesh)
    # changing radius and max_nn affects the resulting mesh
    # normals can be shown by pressing N in the point cloud visualizer
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))

    return pcd


## Function for constructing a mesh ##

# Takes point cloud data and returns a mesh
# Parameters:
# pcd: o3d PointCloud object
# depth, width, scale: Parameters for tuning the mesh
# smooth(Bool): True if smoother mesh needed
# crop(Bool): True if cropping needed
# sharp(Bool): for sharpening the mesh(change parameters if needed)
# simplify(Bool): for simplifying the mesh(change parameter if needed)
# meshtype(poisson/ball): define function used for creating the mesh (Poisson works the best)

# return: open3d.geometry.LineSet

def createMesh(pcd, depth=10,width=1,scale=1.1, smooth=False,
               crop=False, sharp=False, simplify=False, meshtype='poisson') -> o3d.geometry.LineSet:

    if meshtype == 'poisson':
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, width=width, scale=scale,
                                                                         linear_fit=False)[0]

    if meshtype == 'ball':
        mesh = ballMesh(pcd)


    if crop == True:
        bb = createBoundinBox()
        mesh = mesh.crop(bb)

    if smooth == True:
        mesh = smoothMesh(mesh)

    if sharp == True:
        mesh = mesh.filter_sharpen(number_of_iterations=2,strength=2)

    if simplify == True:
        mesh = mesh.simplify_vertex_clustering(1.2)

    # Change triangle mesh to a line set (shows only the lines of the mesh)
    mesh = meshToLineset(mesh)

    return mesh


# Creates a ball mesh
# changing the radius affects the resulting mesh
def ballMesh(pcd):
    radius = 0.5
    return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2,radius * 4, radius * 8]))

# Takes a mesh as a parameter and returns a smoothened mesh
def smoothMesh(mesh_in):
    mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=5)
    mesh_out.compute_vertex_normals()
    return mesh_out

# Changes a mesh to a line set
def meshToLineset(mesh):
    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh).paint_uniform_color([0,1,0])

# Bounding box for cropping a mesh
def createBoundinBox():
    points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0],
                                                  [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]))
    bb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points)
    bb = bb.scale(10,[0,0,0])
    return bb


## Functions for visualization ##

# Shows point clouds / meshes frame by frame
# type: combined_mesh(default) / mesh / point_cloud
# start and stop indices defines the first and last pcd/mesh showed
def animation(type="combined_mesh", start_index=1, stop_index=20):

    # creates a visualizer object and a window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Animation",width=1000)

    for i in range(start_index, stop_index+1):

        dname = "radar_pointcloud_" + str(i) + ".bin"
        # remove the former pcd/mesh from the visualizer
        vis.clear_geometries()

        # add the new mesh/pcd to the visualizer
        if type == "mesh":
            vis.add_geometry(createMesh(getPointcloud(dname), smooth=False, crop=False))

        if type == "point_cloud":
            vis.add_geometry(getPointcloud(dname))

        if type == "combined_mesh":
            vis.add_geometry(createMesh(combined_point_cloud(i)))

        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

# Creates a custom visualizer and shows the mesh
def customVisualizer(meshIn, win_name="mesh",wireframe=True,showBackFace=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=win_name, width=1000)
    opt = vis.get_render_option()
    opt.mesh_show_wireframe = wireframe
    opt.mesh_show_back_face = showBackFace
    opt.line_width = 100
    vis.add_geometry(meshIn)
    vis.run()


## Functions related to point cloud merging ##

# Loads num_of_pcs point clouds beginning with index and returns them in a list.
# If down_sample is true down sampling is done to all point clouds before adding them to the list.
def load_point_clouds(num_of_pcs, index, down_sample=False):
    pcds = []
    x = range(index, index+num_of_pcs+1)
    for i in x:
        dname = "radar_pointcloud_" + str(i) + ".bin"
        pcd = getPointcloud(dname)
        if down_sample == True:
            pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=2)
        pcds.append(pcd)
    return pcds

def pairwise_registration(source, target, max_correspondence_distance_coarse=2,max_correspondence_distance_fine=1):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=True))
    return pose_graph


# Combines point clouds from index to index+num_point_clouds into one point cloud
# Returns an o3d.geometry.PointCloud object
# The output can be modified by changing num_point_cloud, ds(True/False) and optimization(True/False)

def combined_point_cloud(index=0, num_point_clouds=5, ds=True, optimization=True):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pcds_down = load_point_clouds(index=index,down_sample=ds,num_of_pcs=num_point_clouds)

    pose_graph = full_registration(pcds_down)

    # Optimization
    if optimization == True:
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=2,
            edge_prune_threshold=0.25,
            reference_node=0)
        o3d.pipelines.registration.global_optimization(
            pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)

    # Transform points
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)

    # Make a combined point cloud
    pcds = load_point_clouds(index=index, num_of_pcs=num_point_clouds, down_sample=ds)
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    return pcd_combined



### Creating and visualizing point clouds ###

# get a point cloud and draw it
pcd = getPointcloud(dataname)  # dataname defined on the top of the script ("radar_pointcloud_'index'.bin")
customVisualizer(pcd)

# create a Poisson mesh and draw it
pm = createMesh(pcd)
customVisualizer(pm)

# create a combined point cloud and draw it
pcd_c = combined_point_cloud()
customVisualizer(pcd_c)
# create a Poisson mesh and draw it
pm_c = createMesh(pcd_c)
customVisualizer(pm_c)

# create an animation of meshes made from combined point clouds
animation()
