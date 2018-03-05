import bpy
from bpy.props import *
import bmesh
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve,factorized
import os
import time
import pyopencl as cl
from math import sqrt

os.environ['PYOPENCL_CTX'] = '0'
#             LAMBDA     APLHA     BETA   M A  MU B   B MAX
fandisk_03  = [0.01    , 0.00346 , 0.001, 0.5, 1.414, 1000.0]
fandisk_07  = [0.01    , 1.0     , 0.001, 0.9, 1.414, 1000.0]
julius      = [0.000003, 0.00107 , 0.001, 0.5, 1.414, 1000.0]
sphere      = [0.1     , 0.00206 , 0.001, 0.5, 1.414, 1000.0]
bunny       = [0.000004, 1.0     , 0.001, 0.8, 1.414, 1000.0]
nicolo      = [0.6     , 0.00149 , 0.001, 0.5, 1.414, 1000.0]
block       = [0.55    , 0.00389 , 0.001, 0.5, 1.414, 1000.0]
twelve      = [5       , 0.00351 , 0.001, 0.5, 1.2  , 1000.0]
angel       = [0.0005  , 0.000924, 0.001, 0.5, 1.414, 1000.0]
rabbit      = [0.0035  , 0.000809, 0.001, 0.5, 1.414, 1000.0]
iron        = [0.0001  , 0.000912, 0.001, 0.5, 1.414, 1000.0]

# CHANGE THIS LINE AND RUN THE SCRIPT AGAIN TO USE ANOTHER PARAMETERS
# FROM THE LIST ABOVE
[d_lambda,d_alpha,d_beta,d_mu_alpha,d_mu_beta,d_beta_max] = fandisk_07

def initDenoisingProperties(scn):
    bpy.types.Scene.mu_beta = FloatProperty(
        name = 'Mu beta',
        description = 'Mu beta',
        default = d_mu_beta,
        min = 0,
        max = 2)
    scn['mu_beta'] = d_mu_beta
 
    bpy.types.Scene.beta = FloatProperty(
        name = 'Beta', 
        description = 'Beta',
        default = d_beta,
        min = 0,
        max = 0.5)
    scn['beta'] = d_beta
    
    bpy.types.Scene.beta_max = FloatProperty(
        name = 'Beta max', 
        description = 'Beta max',
        default = d_beta_max,
        min = 0,
        max = 2000)
    scn['beta_max'] = d_beta_max
    
    bpy.types.Scene.mu_alpha = FloatProperty(
        name = 'Mu alpha',
        description = 'Mu alpha',
        default = d_mu_alpha,
        min = 0,
        max = 1)
    scn['mu_alpha'] = d_mu_alpha
    
    bpy.types.Scene.alpha = FloatProperty(
        name = 'Alpha', 
        description = 'Alpha',
        default = d_alpha,
        min = 0,
        max = 1)
    scn['alpha'] = d_alpha
    
    bpy.types.Scene.lamb = FloatProperty(
        name = 'Lambda', 
        description = 'Lambda',
        default = d_lambda,
        min = 0,
        max = 2)
    scn['lamb'] = d_lambda

initDenoisingProperties(bpy.context.scene)
 
#
#    Menu in UI region
#
class DenoisingPanel(bpy.types.Panel):
    bl_label = "Mesh denoising via L0 minimization"
    bl_idname = "SCENE_PT_layout"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"
 
    def draw(self, context):
        layout = self.layout
        scn = context.scene
        layout.prop(scn, 'beta')
        layout.prop(scn, 'mu_beta')
        layout.prop(scn, 'beta_max')
        layout.prop(scn, 'alpha')
        layout.prop(scn, 'mu_alpha')
        layout.prop(scn, 'lamb')
        layout.operator('denoising.l0_minimization')
        layout.operator('denoising.l0_minimization_reload')
 
#
#    The button reload default parameters
#
class ReloadDenoisingParamOperator(bpy.types.Operator):
    bl_idname = 'denoising.l0_minimization_reload'
    bl_label = 'Reload parameters'
 
    def execute(self, context):
        scn = context.scene
        #scn['mu_beta']
        scn['beta']=d_beta
        #scn['beta_max'] = 1000
        scn['mu_alpha'] = d_mu_alpha
        scn['alpha'] = d_alpha
        #scn['lamb']
        return{'FINISHED'}

#
#    The button start the batch denoising process
#
class DenoisingOperator(bpy.types.Operator):
    bl_idname = 'denoising.l0_minimization'
    bl_label = 'Denoise'
 
    def execute(self, context):
        scn = context.scene
        denoise(scn['mu_beta'],
                scn['beta'],
                scn['beta_max'],
                scn['mu_alpha'],
                scn['alpha'],
                scn['lamb'],
                scn,
                True)
        return{'FINISHED'}

def denoise(mu_beta,beta,beta_max,mu_alpha,alpha,lamb,scn,batch):
    bpy.ops.object.mode_set(mode='OBJECT')
    # Get the active mesh
    me = bpy.context.object.data
    # Get a BMesh representation
    bm = bmesh.new()   # create an empty BMesh
    bm.from_mesh(me)   # fill it in from a Mesh

    vertex_c = []
    edges_c  = []
    faces_c  = []
    for v in bm.verts:
        vertex_c.append([v.co.x,
                         v.co.y,
                         v.co.z])
    
    for f in bm.faces:
        #print(f.index,f.calc_area())
        faces_c.append([f.verts[0].index,
                        f.verts[1].index,
                        f.verts[2].index])
    
    for e in bm.edges:
        if len(e.link_faces) == 2:
            edges_c.append([e.verts[0].index,
                            e.verts[1].index,
                            e.link_faces[0].index,
                            e.link_faces[1].index])
    
    #vertex_c = np.asarray(vertex_c,dtype=np.float32)
    if not batch:
        [b,a]=step_denoise_l0_minimization( vertex_c,
                                            edges_c,
                                            faces_c,
                                            mu_beta,
                                            beta,
                                            #beta_max,
                                            mu_alpha,
                                            alpha,
                                            lamb)
        scn['beta'] = b
        scn['alpha'] = a
    else:
        [b,a]=denoise_l0_minimization(vertex_c,
                                    edges_c,
                                    faces_c,
                                    mu_beta,
                                    beta,
                                    beta_max,
                                    mu_alpha,
                                    alpha,
                                    lamb)
        scn['beta'] = b
        scn['alpha'] = a
    i = 0
    for v in bm.verts:
       v.co.x = vertex_c[i][0]
       v.co.y = vertex_c[i][1]
       v.co.z = vertex_c[i][2]
       i = i+1

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()  # free and prevent further access
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')

#    Registration
def register():
    bpy.utils.register_class(DenoisingOperator)
    bpy.utils.register_class(ReloadDenoisingParamOperator)
    bpy.utils.register_class(DenoisingPanel)

def unregister():
    bpy.utils.unregister_class(DenoisingOperator)
    bpy.utils.unregister_class(ReloadDenoisingParamOperator)
    bpy.utils.unregister_class(DenoisingPanel)
    
if __name__ == '__main__':
    register()

# | operator symetric version of dot
def dot(v1,v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

# % operator symetric version of cross
def cross(v1,v2):
    return   [v1[1]*v2[2]-v1[2]*v2[1],
              v1[2]*v2[0]-v1[0]*v2[2],
              v1[0]*v2[1]-v1[1]*v2[0]]

def substract(v1,v2):
    return [v1[0] - v2[0] , v1[1] - v2[1] , v1[2] - v2[2]]

def norm(point):
    return sqrt(point[0]*point[0]+point[1]*point[1]+point[2]*point[2])

def print_matrix(matrix):
    print('========')
    for r in range(len(matrix)):
        print(r,'->',matrix[r])
    print('\n========')

def getInitialVerticesMatrix(vertex,l_vertex):
    initial_vertices_matrix = []
    for index in range(l_vertex):
        p = vertex[index]
        initial_vertices_matrix.append([p[0],p[1],p[2]])
    return initial_vertices_matrix

def calculateEdgeVertexHandle(edges, l_edges, faces, l_faces):
    edge_vertex_handle = []
    for e in range(l_edges):
        #two faces ids
        f1 = edges[e][2]
        f2 = edges[e][3]
        #get four vertices correspond to edge e
        v1 = edges[e][0]
        v3 = edges[e][1]

        v4 = faces[f1][0]
        if v4 == v1 or v4 == v3:
            v4 = faces[f1][1]
        if v4 == v1 or v4 == v3:
            v4 = faces[f1][2]

        v2 = faces[f2][0]
        if v2 == v1 or v2 == v3 or v2 == v4:
            v2 = faces[f2][1]
        if v2 == v1 or v2 == v3 or v2 == v4:
            v2 = faces[f2][2]

        edge_vertex_handle.append([v1,v2,v3,v4])
    return edge_vertex_handle

def calculateEdgeHandle(l_vertex,l_edges,edge_vertex_handle):
    
    temp = []
    for o in edge_vertex_handle:
        for v in o:
            temp.append(v)

    edge_vertex_handle = np.array(temp,np.int32)
    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    edge_vertex_handle_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edge_vertex_handle)
    
    prg = cl.Program(ctx, '''
__kernel void prepare_data(
    int l_edges,
    __global const int *edge_vertex_handle,
    __global float *edge_handle){
    int v = get_global_id(0);
    int c_edges = 0;
    int i;

    for(i = 0;i<30;i++){
        edge_handle[v*30+i]=-1;
    }

    for (int e = 0; e < l_edges; e++) {
        if (edge_vertex_handle[e * 4] == v ||
            edge_vertex_handle[e * 4 + 1] == v ||
            edge_vertex_handle[e * 4 + 2] == v ||
            edge_vertex_handle[e * 4 + 3] == v) {
            edge_handle[v*30+c_edges] = e;
            c_edges++;
        }
    }
}
''').build()
    edge_handle_g = cl.Buffer(ctx,mf.WRITE_ONLY,l_vertex *4 * 30 )

    kernel = prg.prepare_data
    kernel.set_scalar_arg_dtypes([np.int32,None,None])

    kernel(queue, (l_vertex,), None, l_edges,edge_vertex_handle_g,edge_handle_g)

    edge_handle = np.empty((l_vertex*30,),np.float32)

    cl.enqueue_copy(queue, edge_handle, edge_handle_g)
    temp = []
    for v in range(l_vertex):
        t = []
        for e in range(30):
            t.append(int(edge_handle[v*30+e]))
        temp.append(t)
    return temp

def calculateEdgeHandle_back(l_vertex,l_edges,edge_vertex_handle,edge_handle):
    for v in range(l_vertex):
        temp = [-1]*30
        count = 0
        for e in range(l_edges):
           if edge_vertex_handle[e][0] == v or edge_vertex_handle[e][1] == v or edge_vertex_handle[e][2] == v or edge_vertex_handle[e][3] == v:
               temp[count] = e
               count +=1
        edge_handle.append(temp)

def getFaceArea(faces,l_faces,vertex,l_vertex):
    face_area=[]
    for i in range(l_faces):
        points = []
        #face.calc_area()
        points.append(vertex[faces[i][0]])
        points.append(vertex[faces[i][1]])
        points.append(vertex[faces[i][2]])
        edge1 = substract(points[1], points[0])
        edge2 = substract(points[1], points[2])
        face_area.append( 0.5 * norm( cross(edge1, edge2)))
    return face_area

def calculateAreaBasedEdgeOperator(vertex,l_vertex,edges,l_edges,faces,l_faces, area_based_edge_operator, edge_vertex_handle, coef):
    face_area = getFaceArea(faces,l_faces,vertex,l_vertex)

    for e in range(l_edges):
        
        temp_coef = [0.0,0.0,0.0,0.0]
        edge_length = norm(substract(vertex[edges[e][0]],vertex[edges[e][1]]))
        
        [v1,v2,v3,v4] = edge_vertex_handle[e]

        #two faces ids
        f1 = edges[e][2]
        f2 = edges[e][3]
        
        # the area of two faces correspond to edge *e_it
        area134   = face_area[f1]
        area123   = face_area[f2]
        totalArea = area123 + area134

        p1 =vertex[v1]
        p2 =vertex[v2]
        p3 =vertex[v3]
        p4 =vertex[v4]

        p12 = substract(p1, p2)
        p13 = substract(p1, p3)
        p14 = substract(p1, p4)
        p23 = substract(p2, p3)
        p34 = substract(p3, p4)

        # calc coefficient
        temp_coef[0] = (area123 * dot(p34, p13) - area134 * dot(p13, p23)) / (edge_length * edge_length * totalArea)
        temp_coef[1] = area134 / totalArea
        temp_coef[2] = (-area123 * dot(p13, p14) - area134 * dot(p12, p13)) / (edge_length * edge_length * totalArea)
        temp_coef[3] = area123 / totalArea
        coef[e] = temp_coef

        # calc area-based edge operator
        pt = [0.0,0.0,0.0]
        pt[0] = p1[0] * temp_coef[0] + p2[0] * temp_coef[1] + p3[0] * temp_coef[2] + p4[0] * temp_coef[3]
        pt[1] = p1[1] * temp_coef[0] + p2[1] * temp_coef[1] + p3[1] * temp_coef[2] + p4[1] * temp_coef[3]
        pt[2] = p1[2] * temp_coef[0] + p2[2] * temp_coef[1] + p3[2] * temp_coef[2] + p4[2] * temp_coef[3]
        #print 'pt',pt
        area_based_edge_operator[e] = pt

def solveDelta(area_based_edge_operator,l_edges, lamb, beta, delta):
    for i in range(l_edges):
        pt = area_based_edge_operator[i]
        if norm(pt)*norm(pt) >= lamb/beta:
            delta[i] = pt
        else:
            delta[i] = [0.0,0.0,0.0]

def solveVertices(vertex,l_vertex,edges,l_edges,faces,l_faces, initial_vertices_matrix, edge_vertex_handle,edge_handle, coef, delta, alpha, beta):
    right_term = []
    for v in initial_vertices_matrix:
        right_term.append([v[0],v[1],v[2]])

    coef_matrix = lil_matrix((l_vertex,l_vertex),dtype=float)

    triple =[]
    start = time.time()
    
    for v in range(l_vertex):
        edge_handle_t = edge_handle[v]
        vertex_coef = {}
        vertex_coef[v] = 1.0
    
        right = [0.0, 0.0, 0.0]

        for e in edge_handle_t:
            index = e
            if index == -1:
                break
            v1 = edge_vertex_handle[index][0]
            v2 = edge_vertex_handle[index][1]
            v3 = edge_vertex_handle[index][2]
            v4 = edge_vertex_handle[index][3]
            coe1 = coef[index][0]
            coe2 = coef[index][1]
            coe3 = coef[index][2]
            coe4 = coef[index][3]
            temp_delta = delta[index]

            if v1 not in vertex_coef:
                vertex_coef[v1] = 0.0
            if v2 not in vertex_coef:
                vertex_coef[v2] = 0.0
            if v3 not in vertex_coef:
                vertex_coef[v3] = 0.0
            if v4 not in vertex_coef:
                vertex_coef[v4] = 0.0

            if v1 == v:
                vertex_coef[v1] = vertex_coef[v1] + alpha + beta * coe1 * coe1
                vertex_coef[v2] = vertex_coef[v2] - alpha + beta * coe1 * coe2
                vertex_coef[v3] = vertex_coef[v3] + alpha + beta * coe1 * coe3
                vertex_coef[v4] = vertex_coef[v4] - alpha + beta * coe1 * coe4
                right[0] += temp_delta[0] * beta * coe1
                right[1] += temp_delta[1] * beta * coe1
                right[2] += temp_delta[2] * beta * coe1
            elif v2 == v:
                vertex_coef[v1] = vertex_coef[v1] - alpha + beta * coe2 * coe1
                vertex_coef[v2] = vertex_coef[v2] + alpha + beta * coe2 * coe2
                vertex_coef[v3] = vertex_coef[v3] - alpha + beta * coe2 * coe3
                vertex_coef[v4] = vertex_coef[v4] + alpha + beta * coe2 * coe4
                right[0] += temp_delta[0] * beta * coe2
                right[1] += temp_delta[1] * beta * coe2
                right[2] += temp_delta[2] * beta * coe2
            elif v3 == v:
                vertex_coef[v1] = vertex_coef[v1] + alpha + beta * coe3 * coe1
                vertex_coef[v2] = vertex_coef[v2] - alpha + beta * coe3 * coe2
                vertex_coef[v3] = vertex_coef[v3] + alpha + beta * coe3 * coe3
                vertex_coef[v4] = vertex_coef[v4] - alpha + beta * coe3 * coe4
                right[0] += temp_delta[0] * beta * coe3
                right[1] += temp_delta[1] * beta * coe3
                right[2] += temp_delta[2] * beta * coe3
            elif v4 == v:
                vertex_coef[v1] = vertex_coef[v1] - alpha + beta * coe4 * coe1
                vertex_coef[v2] = vertex_coef[v2] + alpha + beta * coe4 * coe2
                vertex_coef[v3] = vertex_coef[v3] - alpha + beta * coe4 * coe3
                vertex_coef[v4] = vertex_coef[v4] + alpha + beta * coe4 * coe4
                right[0] += temp_delta[0] * beta * coe4
                right[1] += temp_delta[1] * beta * coe4
                right[2] += temp_delta[2] * beta * coe4
        right_term[v][0] += right[0]
        right_term[v][1] += right[1]
        right_term[v][2] += right[2]
        for key,value in vertex_coef.items():
            triple.append([v,key,value])
    for c,r,v in triple:
        coef_matrix[c,r] = v

    end = time.time()
    print('--> Prepare data')
    print(end-start)
    coef_matrix = coef_matrix.tocsc()
    start = time.time()
    f_function = factorized(coef_matrix)
    end = time.time()
    print('--> Factorize')
    print(end-start)

    start = time.time()
    vertices_term = f_function(np.array(right_term))
    end = time.time()
    print('--> Solve')
    print(end-start)

    for v in range(l_vertex):
        vertex[v] = vertices_term[v]

def denoise_l0_minimization(vertex,edges,faces,mu_beta,beta,beta_max,mu_alpha,alpha,lamb):
    if len(vertex) == 0:
        return
    print('=======================================================')
    print('==============START PROCESS OF DENOISE=================')
    print('=======================================================')
    print('mu_beta',mu_beta)
    print('beta',beta)
    print('beta_max',beta_max)
    print('mu_alpha',mu_alpha)
    print('alpha',alpha)
    print('lambda',lamb)

    l_vertex = len(vertex)
    l_faces = len(faces)
    l_edges = len(edges)
    
    start_f = time.time()
    edge_vertex_handle = calculateEdgeVertexHandle(edges,l_edges,faces,l_faces)

    edge_handle = calculateEdgeHandle(l_vertex,l_edges,edge_vertex_handle)

    initial_vertices_matrix = getInitialVerticesMatrix(vertex,l_vertex)

    end = time.time()
    print('Precalc data')
    print(end-start_f)
    start_p = time.time()
    while beta < beta_max:
        print('\n==== Step in beta',beta,'to',beta_max)
        print('=== With alpha',alpha)
        area_based_edge_operator = [[0.0,0.0,0.0]] * l_edges
        coef                     = [[0.0,0.0,0.0]] * l_edges
        delta                    = [[0.0,0.0,0.0]] * l_edges

        # prepare area based edge operator values
        start = time.time()
        calculateAreaBasedEdgeOperator(vertex,l_vertex,edges,l_edges,faces,l_faces, area_based_edge_operator, edge_vertex_handle, coef)
        end = time.time()
        print('Time to calculateAreaBasedEdgeOperator')
        print(end-start)
        
        # update delta
        start = time.time()
        solveDelta(area_based_edge_operator,l_edges, lamb, beta,delta)
        end = time.time()
        print('Time to solveDelta')
        print(end-start)

        # solve and update vertices
        start = time.time()
        solveVertices(vertex,l_vertex,edges,l_edges,faces,l_faces,initial_vertices_matrix, edge_vertex_handle,edge_handle, coef, delta, alpha, beta)
        end = time.time()
        print('Time to solveVertices')
        print(end-start)
        
        beta *= mu_beta
        alpha *= mu_alpha
    end_p = time.time()
    print('Process Time')
    print(end_p - start_p)
    print('Full Time')
    print(end_p - start_f)
    return [beta,alpha]