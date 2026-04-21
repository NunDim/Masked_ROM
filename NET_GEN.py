#=========================================================
#=========================================================
#Create a complex voronoi 1D network mesh starting from 
#point clouds
#=========================================================
#=========================================================



import numpy as np
from dolfin import *
import numpy as np
import ufl
import scipy as sp
import scipy.spatial as sptl
import random
import math


##########################################################################

def box2sph(point):
    #transform a box into a circle WORKS ONLY from (-1,1)**3 cubes
    x, y, z = point
    xx = x * np.sqrt(1-(y**2/2)-(z**2/2) + (y**2)*(z**2)/3)
    yy = y * np.sqrt(1-(z**2/2)-(x**2/2) + (x**2)*(z**2)/3)
    zz = z * np.sqrt(1-(x**2/2)-(y**2/2) + (y**2)*(x**2)/3)
    return np.array([xx, yy, zz])


#--------------------------------------------------------------------
#--------------------------------------------------------------------


def find_min_max_points(point_cloud):
    x_min, y_min, z_min = float('inf'), float('inf'), float('inf')
    x_max, y_max, z_max = float('-inf'), float('-inf'), float('-inf')

    for point in point_cloud:
        x, y, z = point

        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x

        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y

        if z < z_min:
            z_min = z
        if z > z_max:
            z_max = z


    return (x_min, x_max), (y_min, y_max), (z_min, z_max)


#--------------------------------------------------------------------
#--------------------------------------------------------------------


def is_point_outside_domain(point, domain):
    x, y, z = point
    
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    z_min, z_max = domain[2]
    
    if x < x_min or x > x_max:
        return True
    if y < y_min or y > y_max:
        return True
    if z < z_min or z > z_max:
        return True
    else:  
        return False


#--------------------------------------------------------------------
#--------------------------------------------------------------------


def is_out_of_shape(vertices, edge, shape_box):
    if is_point_outside_domain( vertices[edge[0]] , shape_box) : 
       return True 
    if is_point_outside_domain( vertices[edge[1]] , shape_box) :
       return True  
  
    else:    
       return False
#--------------------------------------------------------------------
#--------------------------------------------------------------------


def is_inside_elli(vertices, edge, elli): 
    #rule out all points inside ellipse 
    vv0 = np.dot(elli, vertices[edge[0]])
    vv1 = np.dot(elli, vertices[edge[1]])

    if (np.linalg.norm(vv0)) <= 1. or (np.linalg.norm(vv1))<= 1.:
        return True
    else:
        return False
    
    


#--------------------------------------------------------------------
#--------------------------------------------------------------------


def create_voronoi_mesh(vert_coord, vert_list, edges):

 #Given list of vertex coordinate tuples, build and return a mesh of intervals."

  # Get dimensions
  gdim = len(vert_coord[0])
  tdim = 1

  # Choice of cellname for simplices
  cellname = "interval"

  # Indirect error checking and determination of tdim via ufl
  ufl_cell = ufl.Cell(cellname, gdim)
  assert tdim == ufl_cell.topological_dimension()

  # Create mesh to return
  mesh = Mesh()

  # Open mesh in editor
  me = MeshEditor()
  me.open(mesh, cellname, tdim, gdim)

  # Add vert_coord to mesh
  nv = max(max(vert_list), len(vert_list))
  nv = nv+1
  me.init_vertices(nv)
  for i in range(nv): 
      if i in vert_list:
         me.add_vertex(i, vert_coord[i])
      else:
         me.add_vertex(i, [-666, -666, -666])
  # Add cells to mesh
  edg_len = len(edges) 
  me.init_cells(edg_len)
  
  for i , edge in enumerate(edges):
      c = (edge[0], edge[1])
      me.add_cell(i, c)
  me.close()

  return mesh


#--------------------------------------------------------------------
#--------------------------------------------------------------------


def find_nearest_points(mesh, plane_coeffs, n):
    # NOTE: tagged out-of-shape vertices as [-666, -666, -666]: 
    vertices = mesh.coordinates()
    # Extract plane coefficients
    A, B, C, D = plane_coeffs  # plane: Ax + By + Cz = D
    
    # Calculate distances between points and plane
    distances = np.abs(np.dot(vertices, plane_coeffs[:3]) - D) / np.linalg.norm(plane_coeffs[:3]) 
    # Sort distances and get indices of closest points
    closest_indices = np.argsort(distances)[:n]
    # Retrieve the closest points from the verteces
    closest_points = vertices[closest_indices]
    
    return (closest_points, closest_indices)


#--------------------------------------------------------------------
#--------------------------------------------------------------------


def get_3D_vector(arr):
    arr = np.array(arr)
    if not arr.shape == (3,1):
       arr = arr.reshape(3,1) #trasnpose array
       return arr
    else:
       return arr


#--------------------------------------------------------------------
#--------------------------------------------------------------------


def again_np_array(arr):
    arr = arr.reshape(1,3)
    return arr[0]

#--------------------------------------------------------------------
#--------------------------------------------------------------------


def get_homo_vector(arr):
    arr = np.array(arr)
    if arr.shape == (1,3):
       arr = np.append(arr, 1)
       arr = arr.reshape(4,1)
       return arr
    else:
       arr = np.hstack((arr,[1]))
       return arr


#--------------------------------------------------------------------
#--------------------------------------------------------------------


def point_plane_proj(point_coords, plane):
    #build projection matrix
    A, B, C, D = plane
    point_coords = get_homo_vector(point_coords)

    #GETTING PARAMETRIZED REPRESENTATION OF THE PLANE
    if not A == 0:
       u = np.array([-C/A, 0, 1])
       v = np.array([-B/A, 1, 0])
       normal = np.cross(u, v)
       normal = normal/np.linalg.norm(normal)
       normal = get_3D_vector(normal)
       P = np.identity(3)-np.matmul(normal , np.transpose(normal) )
       homo = np.array([0, 0, 0])
       P = np.vstack((P, homo))
       homo = np.array([0, 0, 0, 1])
       P = np.hstack((P, homo.reshape(-1,1)))
       #matrici di antitraslazione e traslazione
       ant_trasl = np.array([[1., 0., 0, -D/A], [0., 1., 0., 0.], [0., 0., 1., 0.],  [0., 0., .0, 1.]])                   

       trasl = np.array([[1., 0., 0, D/A], [0., 1., 0., 0.], [0., 0., 1., 0.],  [0., 0., .0, 1.]])
       P = np.matmul(P, ant_trasl) 
       P = np.matmul(trasl, P)                  
       point_hom = np.matmul(P, point_coords) 
       point = point_hom[:3]
       return point
       
    if not B == 0:
       u = np.array([1, -A/B, 0])
       v = np.array([0, -C/B, 1])
       normal = np.cross(u, v)
       normal = normal/np.linalg.norm(normal)
       normal = get_3D_vector(normal)
       P = np.identity(3)-np.matmul(normal , np.transpose(normal) )
       homo = np.array([0, 0, 0])
       P = np.vstack((P, homo))
       homo = np.array([0, 0, 0, 1])
       P = np.hstack((P, homo.reshape(-1,1)))
       #matrici di antitraslazione e traslazione
       ant_trasl = np.array([[1., 0., 0, 0], [0., 1., 0., -D/B], [0., 0., 1., 0.],  [0., 0., .0, 1.]])                   

       trasl = np.array([[1., 0., 0, 0.], [0., 1., 0., D/B], [0., 0., 1., 0.],  [0., 0., .0, 1.]])

       P = np.matmul(P, ant_trasl) 
       P = np.matmul(trasl, P)                  
       point_hom = np.matmul(P, point_coords) 
       point = point_hom[:3]
       return point 

    if not C == 0:
       u = np.array([1, 0, -A/C])
       v = np.array([0, 1 -B/C])
       normal = np.cross(u, v)
       normal = normal/np.linalg.norm(normal)
       normal = get_3D_vector(normal)
       P = np.identity(3) - np.matmul(normal , np.transpose(normal) )
       homo = np.array([0, 0, 0])
       P = np.vstack((P, homo))
       homo = np.array([0, 0, 0, 1])
       P = np.hstack((P, homo.reshape(-1,1)))
       #matrici di antitraslazione e traslazione
       ant_trasl = np.array([[1., 0., 0, 0.], [0., 1., 0., 0.], [0., 0., 1., -D/C],  [0., 0., .0, 1.]])                   

       trasl = np.array([[1., 0., 0, 0.], [0., 1., 0., 0.], [0., 0., 1., D/C],  [0., 0., .0, 1.]])

       P = np.matmul(P, ant_trasl) 
       P = np.matmul(trasl, P)                  
       point_hom = np.matmul(P, point_coords) 
       point = point_hom[:3]
       return point
   
    else: 
      print("plane equation error")
      return 


#--------------------------------------------------------------------
#--------------------------------------------------------------------


def plane_new_verts(mesh, planes_in, n_in, planes_out, n_out): 
    #import pdb; pdb.set_trace() 
    #get n nearest point to plane and projec onto it, creating new vertexes and cells
    #nearest := inflow vertices
    #furthest:= outflow vetices
    li = len(planes_in)
    lo = len(planes_out)

    projected_ver_in = [[] for _ in range(li)]
    projected_ver_out = [[] for _ in range(lo)]

    for pl, (plane_in, plane_out) in enumerate(zip(planes_in, planes_out)):

        nearest = find_nearest_points(mesh, plane_in, n_in)
        furthest = find_nearest_points(mesh, plane_out, n_out)
        vertices = mesh.coordinates()  
        nv = len(vertices)
        last_ver_idx = nv-1  

        #inflow vertices
        projected_ver_in[pl] = [ [], [] ]
        for idx, coord in zip(nearest[1], nearest[0]):
            projected_ver_in[pl][0].append(idx)
            projected_ver_in[pl][1].append(point_plane_proj(coord, plane_in))
        #outflow_vertices
         
        #outflow vertices
        projected_ver_out[pl] = [ [], [] ]
        for idx, coord in zip(furthest[1], furthest[0]):
            projected_ver_out[pl][0].append(idx)
            projected_ver_out[pl][1].append(point_plane_proj(coord, plane_out))
    '''
    #stack all together
    projected_ver_in_tot  = [ [], [] ]
    projected_ver_out_tot = [ [], [] ]

    for pl in range(li):

        projected_ver_in_tot[0] += projected_ver_in[pl][0]
        projected_ver_in_tot[1] += projected_ver_in[pl][1]

    for pl in range(lo):

        projected_ver_out_tot[0] += projected_ver_out[pl][0]
        projected_ver_out_tot[1] += projected_ver_out[pl][1]
    '''





    '''
    # merge all the data relative to the three planes into one array             
    nearest_tot = [nearest[0][0], nearest[0][1]]
    for n in nearest[1:]:
           nearest_tot[0] = np.concatenate((nearest_tot[0], n[0]), axis=0)
           nearest_tot[1] = np.concatenate((nearest_tot[1], n[1]), axis=0)
    nearest_tot = tuple(nearest_tot)    
 
    furthest_tot = [furthest[0][0], furthest[0][1]]
    for f in furthest[1:]:
           furthest_tot[0] = np.concatenate((furthest_tot[0], f[0]), axis=0)
           furthest_tot[1] = np.concatenate((furthest_tot[1], f[1]), axis=0)
    furthest_tot = tuple(furthest_tot) 
    '''

    # CREATING MESH----------------------------------------------------------------


    gdim = len(vertices[0])
    tdim = 1
    # Choice of cellname for simplices
    cellname = "interval"
    # Indirect error checking and determination of tdim via ufl
    ufl_cell = ufl.Cell(cellname, gdim)
    assert tdim == ufl_cell.topological_dimension()  

    #initializin the new mesh containig "boundary" elements
    mesh1 = Mesh()
    me = MeshEditor()
    me.open(mesh1, cellname, tdim, gdim)
   


    #VERTEXES

    #inizializzo con il numero totale di vertici
    nv_old = len(vertices)
    nv_new_in  = []
    nv_new_out = []

    for pl in range(li):
        nv_new_in.append(len(projected_ver_in[pl][1]))
        nv_new_out.append(len(projected_ver_out[pl][1]))

    #[tot_vert_in  += v_in, for v_in in nv_new_in]
    #[tot_vert_out += v_out, for v_out in nv_new_out]


    me.init_vertices(nv_old + sum(nv_new_in) + sum(nv_new_out))

    #reload old (bulk) vertexes
    for i, coords in enumerate(vertices):
        #coords = box2sph(coords)# <--- eventualmente inserire qui il cambio di coordinate!
        
        me.add_vertex(i, coords)

    #load new vertices plane to plane  

    # load inflow vertes
    for pl in range(li):
            for i, coords in enumerate(projected_ver_in[pl][1]):
        
                #coords = box2sph(coords)# <--- eventualmente inserire qui il cambio di coordinate! 
                
                me.add_vertex(i + nv_old + sum(nv_new_in[0:pl]), again_np_array(coords))
        
    # load outflow vertex
    for pl in range(lo):
            for i, coords in enumerate(projected_ver_out[pl][1]):
        
                #coords = box2sph(coords)# <--- eventualmente inserire qui il cambio di coordinate! 
                
                me.add_vertex(i + nv_old + sum(nv_new_in) + sum(nv_new_out[0:pl]), again_np_array(coords))


    #EDGES

    #inizializzo con il numero totale di edge
    ne_old = mesh.num_cells()
    ne_new_in  = nv_new_in
    ne_new_out = nv_new_out
    me.init_cells(ne_old + sum(ne_new_in) + sum(ne_new_out))

    #reload old (bulk) edges
    for i, edge in enumerate(cells(mesh)):
        
        me.add_cell(edge.index(), edge.entities(0))

    #new inflow ones
    for pl in range(li):
        for i, idx in enumerate(projected_ver_in[pl][0]):
            v0 = i + nv_old + sum(nv_new_in[0:pl])
            v1 = idx
            c = (v0, v1)
            me.add_cell(i + ne_old + sum(ne_new_in[0:pl]), c)

    #new outflow ones
    for pl in range(lo):
        for i, idx in enumerate(projected_ver_out[pl][0]):
            v0 = i + nv_old + sum(nv_new_in) + sum(nv_new_out[0:pl])
            v1 = idx
            c = (v0, v1)
            me.add_cell(i + ne_old + sum(ne_new_in) + sum(nv_new_out[0:pl]), c)
    me.close()

    #create marked mesh-----------------------------------------------------------
    vertex_markers = MeshFunction("size_t", mesh1, 0, 0)
    eps = 0.0001
    
    #import pdb; pdb.set_trace() 
    for i, coords in enumerate(vertices):
       if not (min(coords) <= -666+eps) and (min(coords)>= -666-eps):
          vertex_markers[i] = 555 #<---marking bulk
  
    tot_in=[]
    for pl in range(li):
    #for v_i in projected_ver_in[:][1]:
        tot_in += projected_ver_in[pl][1]
    
    tot_out=[]
    for pl in range(lo):
    #for v_i in projected_ver_in[:][1]:
        tot_out += projected_ver_out[pl][1]


    for i, coords in enumerate(tot_in):
       if not (min(coords) <= -666+eps) and (min(coords)>= -666-eps):
          ii = i + nv_old
          vertex_markers[ii] = 111 #<---marking inflow

    for i, coords in enumerate(tot_out):
       if not (min(coords) <= -666+eps) and (min(coords)>= -666-eps):
          iii = i + nv_old + sum(nv_new_in)
          vertex_markers[iii] = 999 #<---marking inflow
 

    #----------------------------------------------------------------------------
    
    return mesh1, vertex_markers


#--------------------------------------------------------------------
#--------------------------------------------------------------------


def get_vor_edges(vor, vertices, shape_box, elli):
   #----------------CREATING EDGES--------------------------------------------
   edges = [[], []]
   
   for facet in vor.ridge_vertices:
       edges[0].extend(facet[:-1]+[facet[-1]])
       edges[1].extend(facet[1:]+[facet[0]])
   
   edges = np.vstack(edges).T  # Convert to scipy-friendly format
   mask = np.any(edges == -1, axis=1)  # Identify edges at infinity
   edges = edges[~mask]  # Remove edges at infinity
   
   
   #making mask
   mask_box = []
   for i, edge in enumerate(edges):
        if is_out_of_shape(vertices, edge, shape_box):
           #t = "True"
           mask_box.append(True)
        else:
           #f = "False"
           mask_box.append(False)
   
   mask_box=np.array(mask_box)
   assert len(mask_box)==len(edges)
    
   #----------------------------------------------------------------------------
   
   edges = edges[~mask_box] #remove out of box

   #making mask
   mask_elli = []
   for i, edge in enumerate(edges):
        if is_inside_elli(vertices, edge, elli):
           #t = "True"
           mask_elli.append(True)
        else:
           #f = "False"
           mask_elli.append(False)
   
   mask_elli=np.array(mask_elli)
   assert len(mask_elli)==len(edges)

   #edges = edges[~mask_elli] #remove inside ellipse COMMENT IF NO OBSTACLE
   edges = np.sort(edges, axis=1)  # Move all points to upper triangle
   
   # Remove duplicate pairs
   edges = edges[:, 0] + 1j*edges[:, 1]  # Convert to imaginary
   edges = np.unique(edges)  # Remove duplicates
   edges = np.vstack((np.real(edges), np.imag(edges))).T  # Back to real
   edges = np.array(edges, dtype=int)
   return edges
 #-------------------------------------------------------------------------


#--------------------------------------------------------------------
#--------------------------------------------------------------------


def random_rectangle(min_b, max_b, h, num_points):
    x_min, y_min = min_b

    x_max, y_max = max_b

    h_min, h_max = h

    x = np.random.uniform(x_min, x_max, num_points)
    y = np.random.uniform(y_min, y_max, num_points)
    z = np.random.uniform(h_min, h_max, num_points)
    points = np.column_stack((x, y, z))
    return points


#--------------------------------------------------------------------
#--------------------------------------------------------------------


def dist(mesh, i0, i1):
   vertices = mesh.coordinates()
   p0 = vertices[i0]
   p1 = vertices[i1]
   d = np.linalg.norm(p0-p1)
   return d


#--------------------------------------------------------------------
#--------------------------------------------------------------------



def fun(mesh, n_departures, n_arrivals, markers):
    #A random curve from the edges
    import networkx as nx
    import random
    from xii import EmbeddedMesh, transfer_markers   
    #edge marking initialization
    facet_f = MeshFunction('size_t', mesh, 1, 0)    
 
    mesh.init(1, 0) 
 
    # Init the graph
    G = nx.Graph()
    edges = mesh.cells()
    # Iterate over cells and get the cell index
    edges_topol = [[], [], []]
    for cell in cells(mesh):
       edges_topol[0].append(cell.index()) #<---index
       edges_topol[1].append(cell.entities(0))#<--- connectivity
       edges_topol[2].append(dist(mesh, cell.entities(0)[0], cell.entities(0)[1])) #<-- tool for weight             


    w_edges_topol = [[], []]
    for i  in range(len(edges_topol[0])):
       w_edges_topol[0].append(edges_topol[0][i])
       listed_edge = edges_topol[1][i].tolist()
       listed_edge.append(edges_topol[2][i])
       w_edges_topol[1].append(listed_edge) 

    edge_indices = {tuple(edge.tolist()): f_index for f_index, edge in zip(edges_topol[0], edges_topol[1])}    
    w_edge_indices = {tuple(w_edge): f_index for f_index, w_edge in zip(w_edges_topol[0], w_edges_topol[1])}
    G.add_weighted_edges_from(iter(w_edge_indices.keys()))
    #---------------------------------------------------------
    '''
    # Assuming G is your graph
    for u, v, data in G.edges(data=True):
          print("PESI", data['weight'], type(data['weight']))
          #data['weight'] = float(data['weight'])
    '''
     
    # creating departures and arrivals
   
    # Access the vertex markers
    vertex_markers = markers.array()
    
    #making departures
    vertices_departures = []
    for idx in range(mesh.num_vertices()):
        if vertex_markers[idx] == 111:
           vertices_departures.append(idx)
    
    #making arrivals
    vertices_arrivals = []
    for idx in range(mesh.num_vertices()):
        if vertex_markers[idx] == 999:
           vertices_arrivals.append(idx)
  
    #---------------------------------------------------------
    

    # concatenazione del for per ogni vertex departures
    for _ in range(n_departures):
        v0 = random.sample(vertices_departures, 1)
        v0 = v0[0]
        for _ in range(n_arrivals):        
            v1 = random.sample(vertices_arrivals, 1)
            v1 = v1[0]
            # The path is a shortest path between 2 random vertices
            path = nx.shortest_path(G, source=v0, target=v1, weight='weight')
            for v00, v11 in zip(path[:-1], path[1:]):
                edge = (v00, v11) if v00 < v11 else (v11, v00)
                facet_f[edge_indices[edge]] = 1
                path = []
    vaso = EmbeddedMesh(facet_f, 1)
    vaso_markers = transfer_markers(vaso, markers)  #trasferisco i markers dalla parent mesh alla child(embedded) mesh
    print("vaso market", vaso_markers.array())
    return vaso, vaso_markers

#----------------------------------------------------------------------


#___________________________________________________________________________________________________
#___________________________________________________________________________________________________
#___________________________________________________________________________________________________
#___________________________________________________________________________________________________
#___________________________________________________________________________________________________
#___________________________________________________________________________________________________
#___________________________________________________________________________________________________
#___________________________________________________________________________________________________
#___________________________________________________________________________________________________
'''
---------- plane out

+---------+
|         |
|   n3    |  h3
|         |
+---------+
|         |
|   n2    |  h2
|         |
+---------+
|         |
|   n1    |  h1
|         |
+---------+

----------  plane in
'''



#--------------PARAMETERS----------------------------------
folder_path = "./nets"

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-name', type=str, help='name of the network mesh')
parser.add_argument('-test', type=str, help='outles of the network mesh')
parser.add_argument('-inlet', type=int, help='inlets of the network mesh')
parser.add_argument('-outlet', type=int, help='outles of the network mesh')
args, _ = parser.parse_known_args()

name = f"{folder_path}/{args.test}/{args.name}_"

nmin    = -1
nmax    = 1

p_min    = [nmin, nmin, nmin]
p_max    = [nmax, nmax, nmax]
n_layers = 3 # for now fixed!


# dati di riferimento-----------------#
#                                     #
#                                     #
#n1 1690 / 4000,/ 7800                #
#num_points_in 560 / 1000 / 1560      #
#-inlet 420 / 1000 / 1950             #
#-outlet 1 (?)                        #
#for the cases 3x3x3, 4x4x4, 5x5x5    #
#                                     #
#                                     #
#-------------------------------------#


n1 = 1690
n2 = 1690         #<--number of point for each zone
n3 = 1690

delta = (nmax-nmin)/n_layers
h1 = [p_min[0], p_min[0] + delta]
h2 = [p_min[0] + delta, p_min[0] + 2*delta]  #<-- zone extension
h3 = [p_min[0] + 2*delta, p_max[0]]

#definig number of inlet and outlet
num_points_in  = 560
num_points_out = 560

#definig number of vase and number of their ramifications
#n_vasi = np.random.randint(1, 10)
#n_ramifications =np.random.randint (1, 200)
n_vasi = args.inlet
n_ramifications = args.outlet

#checks
if num_points_in >= n1:
   print("WARNING: possbily unnatural straight inflow channels")
if num_points_out >= n3:
   print("WARNING: possbily unnatural straight outflow channels")
if n_vasi >= num_points_in:
   print("WARNING: number of vasi greater than number of inflows")
if n_ramifications*n_vasi >= num_points_out:
   print("WARNING: number of total vasi outflow greater than number of outflows")


#----------------------------------------------------------
base_pts_1 = random_rectangle([p_min[0], p_min[1]] ,[p_max[0] , p_max[1]], h1, n1 )
base_pts_2 = random_rectangle([p_min[0], p_min[1]] ,[p_max[0] , p_max[1]], h2, n2 )
base_pts_3 = random_rectangle([p_min[0], p_min[1]] ,[p_max[0] , p_max[1]], h3, n3 )

#merges all points clouds
base_pts = np.vstack((base_pts_1,base_pts_2,base_pts_3))



#define the obstacle--------------------------------------------------- 

theta = np.random.uniform(0., 2*np.pi, 1) #<--rotation angle
theta = theta[0]
#print("theta", theta, np.cos(theta))
ax, ay, az = np.random.uniform(0.6, 0.9, 3)

scal_mat = np.array([[ax, 0., 0.],
                     [0., ay, 0.], 
                     [0., 0., az]], dtype=np.float64)

#ay = np.random(0:1)
#az = np.random(0:1)
rot_axis = np.random.randint(0, 2)

# TODO: maybe there is an error in the rotation matrix

if rot_axis == 0:
                rot_mat = np.array([[1., 0., 0.],
                           [0., np.cos(theta), -np.sin(theta)],
                           [0., np.sin(theta),  np.cos(theta)] ], dtype=np.float64) 
                          

if rot_axis == 1:
                rot_mat = np.array([[np.cos(theta), 0., np.sin(theta)],
                           [0., 1., 0.],
                           [-np.sin(theta), 1., np.cos(theta)] ], dtype=np.float64) 
                            

if rot_axis == 2:
                rot_mat =np.array([[np.cos(theta), -np.sin(theta), 0.],
                           [np.sin(theta), np.cos(theta), 0.], 
                           [0., 0., 1.] ], dtype=np.float64)

#making the inverse matrix
A = np.dot(scal_mat, rot_mat)
A_inv = np.linalg.inv(A) 

#----------------------------------------------------------------------

#create starting voronoi meshes
vor = sptl.Voronoi(points=base_pts)
vertices = vor.vertices

#getting box containing points-------------------------------------------------

shape_box = find_min_max_points(base_pts)

#viualization of the box
nx, ny, nz = 1, 1, 1

# Create the box mesh
xmin, xmax = shape_box[0]
ymin, ymax = shape_box[1]
zmin, zmax = shape_box[2]
box_mesh = BoxMesh(Point(xmin, ymin, zmin), Point(xmax, ymax, zmax), nx, ny, nz)

#defining inflow and outflow planes Ax + By + Cz = D 
planes_in = [[0, 0, 1, zmin], [1, 0, 0, xmin], [0, 1, 0, ymin]] #inflow planes; this can be a list of planes
#planes_in = [[0, 0, 1, zmin], [0, 0, 1, zmin]] #inflow planes; this can be a list of planes 
planes_out =[[0, 0, 1, zmax], [1, 0, 0, xmax], [0, 1, 0, ymax]] #outflow planes; this can be a list of planes
#planes_out =[[0, 0, 1, zmax], [0, 1, 0, ymax]] #outflow planes; this can be a list of planes


#vtu box mesh mesh export---------------------------------------
print("\n...printing box vtk mesh")
file = File(f'{name}_box.pvd')
file << box_mesh
#---------------------------------------------------------------

# Creating edges s.t. they are in box---------------------------
edges = get_vor_edges(vor, vertices, shape_box, A_inv)

# Extract vertex from edges without repetitions
vert_list = list(set(vert_list for sublist in edges for vert_list in sublist))

# Create voronoi fenics mesh
mesh1D = create_voronoi_mesh(vertices, vert_list, edges)   


#points, indices = find_nearest_points(mesh1D,  plane_out, num_points_out)

mesh1D, markers = plane_new_verts(mesh1D, planes_in, num_points_in, planes_out, num_points_out)
 
#vtu mesh export------------------------------------------------

print("\n...printing base voronoi vtk mesh\n")
file = File(f'{name}_reticolo.pvd')
file << mesh1D
#---------------------------------------------------------------

#create final network mesh with inflow, outflow and bulk markers

vaso, vaso_markers = fun(mesh1D, n_vasi, n_ramifications, markers)
print("N DOF vaso", vaso.num_edges())
NDOFvaso =  vaso.num_edges()


#vtu mesh export------------------------------------------------

print("\n...printing ultimate vaso vtk  mesh\n")
#file = File(f'output_complex_network/net_{NDOFvaso}/vaso.pvd')
file = File(f'{name}_vaso.pvd')
file << vaso
#---------------------------------------------------------------

#export mesh in xdmf--------------------------------------------

# Define XDMFFile and open it for writing mesh
file = XDMFFile(MPI.comm_world, f"{name}_marked_mesh.xdmf")
file.parameters["flush_output"] = True
# Write the mesh and mesh function to the XDMF file
file.write(vaso)
# Close the XDMFFile
file.close

# Define XDMFFile and open it for writing mesh
file = XDMFFile(MPI.comm_world, f"{name}_markers.xdmf")
file.parameters["flush_output"] = True
# Write the mesh and mesh function to the XDMF file
file.write(vaso_markers)
# Close the XDMFFile
file.close()


# CODE BELOW IS NOT ESSENTIAL FOR NET GENERATION


#visualize ellipse
# Create a sphere mesh

import gmsh
import meshio


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points, cells={
                           cell_type: cells}, cell_data={"name_to_read": [cell_data]})
    if prune_z:
        out_mesh.prune_z_0()
    return out_mesh

# Initialize Gmsh ------------------------------------------------------------------
gmsh.initialize()

# Load the Gmsh script
gmsh.merge("../sphere.geo")

# Generate the mesh
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)

# Save the mesh to a file
gmsh_mesh = f"sphere.msh" 
gmsh.write(gmsh_mesh)

# Close Gmsh
gmsh.finalize()
#----------------------------------------------------------------------------------


msh = meshio.read(gmsh_mesh) #reading mesh.msh

triangle_mesh = create_mesh(msh, "triangle") #creating 2D mesh.msh

xdmf_mesh = f"sphere.xdmf"

meshio.write(xdmf_mesh, triangle_mesh) # convert msh to xdmf

#loading mesh
mesh1 = Mesh()
with XDMFFile(xdmf_mesh) as infile:
     infile.read(mesh1)

coord = mesh1.coordinates()
coords_new = np.zeros(coord.shape)
for i, v in enumerate(coord):
    coords_new[i] = np.dot(A, v)
mesh1.coordinates()[:] = coords_new
#-----------print mesh to pvd file------------------
file = File(f"{name}_ellipse.pvd")
file << mesh1
#--------------------------------------------------


