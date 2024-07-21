import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.path as mpath
import tensorflow as tf
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis as cfv
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator, CubicSpline, interp1d, PchipInterpolator
from scipy.spatial import Delaunay
import sys
import os
import json
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape, Concatenate
from tensorflow.keras.models import Model
from tqdm import tqdm

class IsInDomain:
    def __init__(self, nodesCurves):
        self.loops = [mpath.Path(nodesCurves[i]) for i in range(len(nodesCurves))]
        
    def __call__(self, points):
        return np.logical_and(self.loops[0].contains_points(points),np.logical_not(np.array([loop.contains_points(points) for loop in self.loops[1:]]).any(0)))

class Interp2DInDomain:
    def __init__(self,points,vals,isInDomainF,kind='cubic'):
        if kind == 'cubic':
            self.interpF = CloughTocher2DInterpolator(points,vals)
        elif kind == 'linear':
            self.interpF = LinearNDInterpolator(points,vals)
        else:
            raise ValueError("kind must be either 'cubic' or 'linear'")
        self.inInDomainF = isInDomainF
        self.nDims = vals.shape[-1]
    
    def __call__(self,points):
        vals = self.interpF(points)
        vals[~self.inInDomainF(points)] = np.nan
        return vals
    
class Interp1DPeriodic:
    def __init__(self, x, y, kind = 'linear'):
        self.x = x
        self.y = y
        self.kind = kind
        self.spline = interp1d(self.x,self.y,kind=self.kind,axis=0)
    
    def __call__(self, x):
        return self.spline(x%self.x[-1])
    
class Interp1DPchipPeriodic:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.spline = PchipInterpolator(self.x,self.y,axis=0)
    
    def __call__(self, x):
        return self.spline(x%self.x[-1])
    
def addInterpF2Data(data,derL=False,includePressureBC=False):
    valSliceE = slice(0,3)
    valSliceD = slice(0,2)

    data['isInDomainF'] = IsInDomain(data['nodesCurves'])
    # data['interpSE'] = Interp2DInDomain(data['nodes'],data['solution'][:,valSliceE],data['isInDomainF'],kind='linear')
    # data['interpSD'] = Interp2DInDomain(data['nodes'],data['solution'][:,valSliceD],data['isInDomainF'],kind='linear')
    data['interpSE'] = Interp2DInDomain(data['nodes'],data['solution'][:,valSliceE],data['isInDomainF'],kind='cubic')
    data['interpSD'] = Interp2DInDomain(data['nodes'],data['solution'][:,valSliceD],data['isInDomainF'],kind='cubic')
    data['interpD2B'] = Interp2DInDomain(data['nodes'],data['distance2boundary'],data['isInDomainF'],kind='linear')
    data['interpELS'] = Interp2DInDomain(data['nodes'],data['elSizeNodes'],data['isInDomainF'],kind='linear')

    data['solutionB'] = []
    data['interpBS'] = []
    data['interpBN'] = []
    data['interpBC'] = []
    if derL:
        data['solutionBoffset'] = []
        
    for i in range(len(data['idxCurves'])):
        data['solutionB'].append(data['solution'][data['idxCurves'][i]])

        if derL:
            nodesCurvesOffset = data['nodesCurves'][i] - derL*data['normalCurves'][i]
            valsCurvesOffset = data['interpSE'](nodesCurvesOffset)
            data['solutionBoffset'].append(valsCurvesOffset)
            boundaryData = np.concatenate([data['solutionB'][i],valsCurvesOffset],-1)
            data['interpBS'].append(Interp1DPchipPeriodic(data['distanceCurves'][i],boundaryData))
        else:
            if includePressureBC:
                data['interpBS'].append(Interp1DPchipPeriodic(data['distanceCurves'][i],data['solutionB'][i]))
            else:
                data['interpBS'].append(Interp1DPchipPeriodic(data['distanceCurves'][i],data['solutionB'][i][:,:2]))
        
        data['interpBN'].append(Interp1DPchipPeriodic(data['distanceCurves'][i],data['normalCurves'][i]))
        data['interpBC'].append(Interp1DPchipPeriodic(data['distanceCurves'][i],data['nodesCurves'][i]))

def addElementSize2Data(data):
    nodes = data['nodes']
    elements = data['elements']
    areaElements = GetAreaTriang(nodes[elements])[:,0,0]
    areaNodes = np.zeros(nodes.shape[0])
    countNodes = np.zeros(nodes.shape[0])
    for i,el in enumerate(elements):
        areaNodes[el] += areaElements[i]
        countNodes[el] += 1

    countNodes[countNodes == 0] = np.inf
    areaNodes /= countNodes
    areaNodes = areaNodes.reshape(-1,1)
    elSizeNodes = 2/(3**0.25)*np.sqrt(areaNodes)
    data['elSizeNodes'] = elSizeNodes

def remove_elements(lst, indices):
    for index in sorted(indices, reverse=True):
        if index < len(lst):
            del lst[index]
    return lst

def loadData(filePath,derL=False,includePressureBC=False):
    with open(filePath, 'r') as f:
        data = json.load(f)
    
    # idxIgnore = [293, 369, 564, 1561, 1867, 1998, 2473, 2475, 2802, 3272, 3469, 3474, 3594, 3764, 3784, 3900, 3921]
    # data = remove_elements(data,idxIgnore)
    keys_single = ['nodes', 'elements', 'solution', 'elementsBoundaryNodesOnly','areaElementsBoundaryNodesOnly','distance2boundary','elSizeNodes']
    for i in tqdm(range(len(data))):
        for key in data[i].keys():
            if key in keys_single:
                data[i][key] = np.array(data[i][key])
            else:
                for j in range(len(data[i][key])):
                    if type(data[i][key][j]) == list:
                        data[i][key][j] = np.array(data[i][key][j])
        # try:
        #     addInterpF2Data(data[i],derL)
        # except:
        #     print('Error in data',i)
        #     idxIgnore.append(i)
        #     continue
        addInterpF2Data(data[i],derL,includePressureBC)
        data[i]['area'] = GetAreaTriang(data[i]['nodes'][data[i]['elements']])
    return data

def GenerateGeometry2(p):
    g = cfg.Geometry()
    for i in range(p.shape[0]):
        g.point(list(p[i]))
    
    for i in range(4):
        g.line([i,(i+1)%4],marker=1)
    g.spline(list(range(4,p.shape[0]))+[4],marker=2)
    g.surface([0,1,2,3],[[4]])
    return g

def MeshSurface(g,elSize):
    mesh = cfm.GmshMesh(g)
    mesh.elType = 2       # Degrees of freedom per node.
    mesh.dofsPerNode = 1     # Factor that changes element sizes.
    mesh.elSizeFactor = elSize # Element size Factor
    nodes, edof, dofs, bdofs, elementmarkers = mesh.create()

    elements = edof-1
    boundaryNodes = [np.array(bdofs[1])-1,np.array(bdofs[2])-1]
    internalNodes = np.setdiff1d(np.arange(nodes.shape[0]), np.concatenate(boundaryNodes))
    return nodes, elements, boundaryNodes, internalNodes

def computeLengthAlongCurve(nodesB):
    dl = np.sqrt(((nodesB[1:]-nodesB[:-1])**2).sum(1))
    l = np.cumsum(dl)
    l = np.concatenate([[0],l],0)
    return l

def RemeshData(data,elSize):
    nodes = data['nodes']
    idxSquare = data['idxCorner']
    idxSpline = data['idxCurves'][1][:-1]
    nodesB = np.concatenate([ nodes[idxSquare], nodes[idxSpline]],0)

    g = GenerateGeometry2(nodesB)
    nodes,elements,idxCurves,internalNodes = MeshSurface(g,elSize)
    for i in range(len(idxCurves)):
        idxCurves[i] = np.sort(idxCurves[i])

    nodesCurve00 = nodes[idxCurves[0][0]]
    alpha0 = np.arctan2(nodesCurve00[1],nodesCurve00[0])
    nodesCurves0 = nodes[idxCurves[0]]
    alpha = np.arctan2(nodesCurves0[:,1],nodesCurves0[:,0])-alpha0
    alpha[alpha<0] += 2*np.pi
    idxs_sorted = np.argsort(alpha)
    idxCurves[0] = idxCurves[0][idxs_sorted]
    for i in range(len(idxCurves)):
        idxCurves[i] = np.concatenate([idxCurves[i],idxCurves[i][0:1]],0)

    nodesCurves = [nodes[idxCurves[i]] for i in range(len(idxCurves))]
    distanceCurves = [computeLengthAlongCurve(nodesCurves[i]) for i in range(len(idxCurves))]
    lengthCurves = [distanceCurves[i][-1] for i in range(len(idxCurves))]

    solution = data['interpSE'](nodes)[:,:2]
    for i in range(len(idxCurves)):
        solution[idxCurves[i][:-1]] =  data['interpBS'][i](distanceCurves[i][:-1])[:,:2]
    curvesNormals = [data['interpBN'][i](distanceCurves[i]) for i in range(len(idxCurves))]


    data_remeshed = {'nodes':nodes, 'elements':elements, 'idxCurves':idxCurves, 'internalNodes':internalNodes, \
                     'nodesCurves':nodesCurves, 'distanceCurves':distanceCurves, 'lengthCurves':lengthCurves, \
                     'solution':solution, 'curvesNormals':curvesNormals, 'interpSE':data['interpSE'], 'interpSD':data['interpSD'], \
                     'interpBS':data['interpBS'], 'interpBN':data['interpBN'], 'interpBC':data['interpBC'], 'isInDomainF':data['isInDomainF']}

    return data_remeshed

def GetAreaTriang(nodes_els):
    # Calculate the area of a triangles in a mesh
    #   nodes_els - [N,3,2] tensor which containes the x,y positions of nodes of N triangles
    b = (np.roll(nodes_els[:,:,1],1,axis=1) - np.roll(nodes_els[:,:,1],2,axis=1)).reshape(-1,3,1)
    c = (np.roll(nodes_els[:,:,0],2,axis=1) - np.roll(nodes_els[:,:,0],1,axis=1)).reshape(-1,3,1)
    Area = np.abs(np.matmul(nodes_els[:,:,0].reshape(-1,1,3),b))/2
    return Area

def GenerateTriangTrainMeshInter(sRand=0.1):
    # Generates random variation of a mesh made out of 6 triangular elements in a hexagon configuration
    # The mesh is normalized so that the longest edge length is 1
    #   nodes - [7,2] tensor which containes the x,y positions of nodes of the mesh
    #   elements - [6,3] tensor which containes the indices of the nodes of the mesh

    # constructu a hexagon with a point in the middle
    alphas = np.linspace(0,2*np.pi,7)[:-1]-2*np.pi/3 + np.random.uniform(-np.pi/15,np.pi/15,6)
    r = np.random.uniform(1-sRand,1+sRand,6)
    bNodes = np.stack([np.cos(alphas)*r,np.sin(alphas)*r],axis=1)
    nodes = np.concatenate([[[0,0]],bNodes],axis=0)

    # construct element matrix
    elements = np.arange(3).reshape(1,-1)+np.arange(6).reshape(-1,1)
    elements[:,0] = 0
    elements[elements>6]=1

    # randomly rotate the mesh
    rot = np.random.uniform(0,np.pi/3)
    rot_mat = np.array([[np.cos(rot),np.sin(rot)],[-np.sin(rot),np.cos(rot)]])
    nodes = (rot_mat@nodes.T).T
    return nodes,elements

def RandomTriangSample(nodes):
    # Sample a random points inside a trinagle defined by [3,2] matrix: nodes
    y = np.random.uniform()
    x = np.abs(np.random.uniform()-np.random.uniform())
    p = nodes[1] + (nodes[0]-nodes[1])*x - (nodes[1]-nodes[2])*(1-x)*y
    return p

def RandomMeshSample(nodes,elements,Area):
    # Sample a random point from a 2D shape defined with a mesh: (nodes, elements)
    el_sample = np.random.choice(np.arange(elements.shape[0]),p=Area.flatten()/Area.sum())
    point_sample = RandomTriangSample(nodes[elements[el_sample]])
    return point_sample

# def GetRandomTPmesh(nodes,elements,areaElements,isInDomainF,interpD2B,elSizeMin=0.04,elSizeMax=0.07,sRand=0.1):
#     # Get a random interior training patch mesh from a mesh defined by (nodes, elements, boundaryNodes)
#     # The mesh will be scaled so that the longest edge is approximately between elSizeMin and elSizeMax
#     # In case the randomly sampled point is so close to the boundary that the Training Patch mesh does not fit fully inside the domain the point is resampled
#     iterations = 0
#     while (iterations < 1_000):
#         samplePoint = RandomMeshSample(nodes,elements,areaElements)
#         notesTP, elementsTP = GenerateTriangTrainMeshInter(sRand)
#         elSize = np.random.uniform(elSizeMin, elSizeMax)
#         if interpD2B(samplePoint.reshape(-1,2)) < elSize*1.5:
#             iterations += 1
#             continue
#         nodesTP = notesTP * elSize + samplePoint.reshape(1,2)
#         isInDomain = isInDomainF(nodesTP)
#         if np.all(isInDomain):
#             break
#         iterations += 1
#     return nodesTP, elementsTP

def GetRandomTPmesh(nodes,elements,areaElements,isInDomainF,interpD2B,elSizeMin=0.04,elSizeMax=0.07,elSizeF=None,sRand=0.1):
    # Get a random interior training patch mesh from a mesh defined by (nodes, elements, boundaryNodes)
    # The mesh will be scaled so that the longest edge is approximately between elSizeMin and elSizeMax
    # In case the randomly sampled point is so close to the boundary that the Training Patch mesh does not fit fully inside the domain the point is resampled
    iterations = 0
    while (iterations < 1_000):
        samplePoint = RandomMeshSample(nodes,elements,areaElements).reshape(-1,2)
        notesTP, elementsTP = GenerateTriangTrainMeshInter(sRand)
        if elSizeF is None:
            elSize = np.random.uniform(elSizeMin, elSizeMax)
        else:
            elSize = elSizeF(samplePoint) * np.random.uniform(elSizeMin, elSizeMax)
        if interpD2B(samplePoint) < elSize*1.6:
            iterations += 1
            continue
        nodesTP = notesTP * elSize + samplePoint
        isInDomain = isInDomainF(nodesTP)
        if np.all(isInDomain):
            break
        iterations += 1
    return nodesTP, elementsTP

# Defined the points of the encoder input mask
def GetRadialEncoderInputMask(n_r,n_theta,k=0.5,Esize=0.05):
    # Get Radial interior encoder input mask
    #   n_theta - number of radial directions along which points are defined
    #   n_r - number of points along each radial direction
    #   k - coefficient between 0 and 1 which defines how much should the points be denser towards the centre compared to the edge
    #   Esize - size of the interour encoder mask, radius of the circle
    alphas = np.linspace(0,2*np.pi,n_theta+1)[:-1].reshape(1,-1)
    r = np.linspace(0,1,n_r+1)[1:].reshape(-1,1)
    r = k*r+(1-k)*r**3
    nodesIE = np.stack([r*np.cos(alphas),r*np.sin(alphas)],axis=-1)
    nodesIE = np.concatenate([[[0,0]],nodesIE.reshape(-1,2)],axis=0)
    return nodesIE*Esize

def GetSquareEncoderInputMask(n,k=0.5,Esize=0.05):
    # Get Square interior encoder input mask
    #   The mask is a 2n x 2n grid points of size 2*Esize x 2*Esize (in x,y coordinates)
    #   k - coefficient between 0 and 1 which defines how much should the points be denser towards the centre compared to the edge
    x = np.linspace(-1,1,2*n+1)
    x = k*x+(1-k)*x**3
    X,Y = np.meshgrid(x,x)
    nodesIE = np.stack([X,Y],axis=-1).reshape(-1,2)
    return nodesIE*Esize

def GetBoundaryEncoderInputMask(n,k,Esize=0.05):
    # Get boundary encoder input mask whcih is defined along the boundary
    #   total number of points in teh boundary encoder input mask is 2*n-1
    #   k - coefficient between 0 and 1 which defines how much should the points be denser towards the centre compared to the edge
    #   Esize - the size of the  boundary encoder input mask, the length of the mask is 2*Esize
    x = np.linspace(-1,1,2*n-1)
    x = (k*np.abs(x)+(1-k)*np.abs(x)**2) * np.sign(x)
    return x*Esize

def GetEncoderInputTP(nodesTP,nodesIE):
    # Based on training patch mesh nodes (nodesTP) and nodes of the interior encoder input mask (nodesIE) generates all interior encoder input mask points within the training patch
    nodesTP = nodesTP.reshape(-1,1,2)
    nodesIE = nodesIE.reshape(1,-1,2)
    nodesTP_IE = nodesTP + nodesIE
    return nodesTP_IE

def GetRandomTrainingPatchInternal(data, nodesIE, nodesD, elSizeMin=0.04, elSizeMax=0.07, debugging=False,sRand=0.1,variableElSize=False):
    # Samples a random training patch from the dataset of solutions and generates all the relevent outputs
    #   data - list of dictionaries which defines dataset of solutions
    #   nodesIE - interior encoder input mask nodes
    #   nodesD - decoder output mask nodes
    #   elSizeMin, elSizeMax - approximate min, max size of the logest edge in the training patch mesh
    idx = np.random.randint(len(data))
    nodes = data[idx]['nodes']
    # if variableElSize:
    #     elements = data[idx]['elements']
    #     areaElements = np.ones(elements.shape[0])
    # else:
    elements = data[idx]['elementsBoundaryNodesOnly']
    areaElements = data[idx]['areaElementsBoundaryNodesOnly']
    interpSE = data[idx]['interpSE']
    interpSD = data[idx]['interpSD']
    isInDomainF = data[idx]['isInDomainF']
    interpD2B = data[idx]['interpD2B']
    if variableElSize:
        interpELS = data[idx]['interpELS']
    else:
        interpELS = None

    # nodesTP, elementsTP = GetRandomTPmesh(nodes,elements,areaElements,isInDomainF,elSizeMin,elSizeMax,sRand=sRand) # get training patch mesh
    nodesTP, elementsTP = GetRandomTPmesh(nodes,elements,areaElements,isInDomainF,interpD2B,elSizeMin=elSizeMin,elSizeMax=elSizeMax,elSizeF=interpELS,sRand=sRand)
    nodesTP_IE = GetEncoderInputTP(nodesTP,nodesIE) # get all nodes of the training patch
    uTP_IE = interpSE(nodesTP_IE.reshape(-1,2)).reshape(7,nodesIE.shape[0],-1) # sample solution at training patch nodes
    bTP_IE = isInDomainF(nodesTP_IE.reshape(-1,2)).reshape(7,nodesIE.shape[0]) # sample boundary mask for interior econder input (1 if inside the domain, 0 if outside)
    uTP_IE[~bTP_IE] = 0 # set solution values outside of the domain to 0
    bTP_IE = np.expand_dims(bTP_IE,-1)

    nodesTP_D = nodesD+nodesTP[:1,:]
    u_D = interpSD(nodesTP_D) # sample solution at decoder output mask nodes
    b_D = isInDomainF(nodesTP_D) # sample boundary mask for decoder output (1 if inside the domain, 0 if outside)
    u_D[~b_D] = 0 # set solution values outside of the domain to 0

    if debugging:
        return nodesTP, elementsTP, uTP_IE, bTP_IE, u_D, b_D, nodesTP_IE
    else:
        return nodesTP, elementsTP, uTP_IE, bTP_IE, u_D, b_D

def GenerateTriangTrainMeshBoundary(v_rand = 0.1):
    # Generates a mesh of 6 triangular elements in a hexagon configuration where the two bottom points have x,y coordinates (0,0) and (1,0)
    alphas = np.linspace(0,2*np.pi,7)[:-1]-2*np.pi/3
    bNodes = np.stack([np.cos(alphas),np.sin(alphas)],axis=1)
    nodes = np.concatenate([[[0,0]],bNodes],axis=0)
    # randomly perturb the nodes except the two bottom ones
    v_rand_nodes = np.random.uniform(-v_rand,v_rand,(7,2))
    v_rand_nodes[[1,2],:] = 0
    nodes += v_rand_nodes
    nodes = nodes-nodes[1]
    elements = np.arange(3).reshape(1,-1)+np.arange(6).reshape(-1,1)
    elements[:,0] = 0
    elements[elements>6]=1
    return nodes,elements

# def GetRandomTPmeshBoundary(nodesB,length,interpNodesB,IsInDomainF,elSizeMin=0.04,elSizeMax=0.07,idx_curve_0=None,distance_0=None):
#     # Get a random boundary training patch mesh (nodesTPB, elementsTPB) from a mesh defined by (nodesB, alpha)
#     #   nodesB - [N,2] tensor which contains the x,y coordinates of the boundary nodes

#     length = np.array(length)
#     if idx_curve_0 is None:
#         idx_curve_0 = np.random.choice(np.arange(len(nodesB)),p=length/length.sum())
#         # idx_curve_0 = np.random.randint(0,len(nodesB))
#     if distance_0 is None:
#         distance_0 = np.random.uniform(0,length[idx_curve_0])# randomly sample a point on the boundary

#     iterations = 0

#     while iterations < 1_000:

#         elSize = np.random.uniform(elSizeMin,elSizeMax) # define the element size of the training patch mesh

#         scale = 1 # scaling factor elSize which is used to fite the training patch mesh in areas of high boundary curvature were distance is not a good measure of the element size
#         for i in range(100):
#             # iterate to find the second point of the training patch mesh on the boundary
#             distance_1 = (distance_0 + elSize*scale)#%length[idx_curve_0] # define the second point of the training patch mesh on the boundary

#             # interpolate the boundary nodes between the two sampled points
#             nodes_0 = interpNodesB[idx_curve_0](distance_0)
#             nodes_1 = interpNodesB[idx_curve_0](distance_1)

#             # Computes the appropriate scaling and rotation for the normalized training patch mesh
#             rotTPB = -np.arctan2(nodes_1[1]-nodes_0[1],nodes_1[0]-nodes_0[0])
#             elSizeTPB = np.sqrt(((nodes_1 - nodes_0)**2).sum())
#             if abs(elSizeTPB-elSize)/elSize < 0.1:
#                 break
            
#             scale = scale*np.sqrt(elSize/elSizeTPB)

#         rot_mat = np.array([[np.cos(rotTPB),np.sin(rotTPB)],[-np.sin(rotTPB),np.cos(rotTPB)]])
#         nodesTPB, elementsTPB = GenerateTriangTrainMeshBoundary()
#         nodesTPB = (rot_mat@nodesTPB.T).T * elSizeTPB + nodes_0
#         isInDomain = IsInDomainF(nodesTPB[[0,3,4,5,6]])
        
#         # Check if the training patch mesh is within the domain and the element size is within the bounds
#         if np.all(isInDomain) and (elSizeTPB < elSizeMax) and (elSizeTPB > elSizeMin):
#             break
#         iterations += 1
#         distance_0 = np.random.uniform(0,length[idx_curve_0])# randomly sample a point on the boundary

#     # print(iterations)
#     return nodesTPB, elementsTPB, distance_0, distance_1, idx_curve_0

def GetRandomTPmeshBoundary(nodesB,length,interpNodesB,IsInDomainF,elSizeMin=0.04,elSizeMax=0.07,elSizeBF=None,idx_curve_0=None,distance_0=None):
    # Get a random boundary training patch mesh (nodesTPB, elementsTPB) from a mesh defined by (nodesB, alpha)
    #   nodesB - [N,2] tensor which contains the x,y coordinates of the boundary nodes

    length = np.array(length)
    if idx_curve_0 is None:
        # idx_curve_0 = np.random.choice(np.arange(len(nodesB)),p=length/length.sum())
        idx_curve_0 = np.random.randint(0,len(nodesB))
    if distance_0 is None:
        distance_0 = np.random.uniform(0,length[idx_curve_0]) # randomly sample a point on the boundary

    iterations = 0

    while iterations < 1_000:

        if elSizeBF is None:
            elSize = np.random.uniform(elSizeMin,elSizeMax) # define the element size of the training patch mesh
        else:
            elSize = elSizeBF[idx_curve_0](distance_0) #* np.random.uniform(elSizeMin,elSizeMax)

        scale = 1 # scaling factor elSize which is used to fite the training patch mesh in areas of high boundary curvature were distance is not a good measure of the element size
        for i in range(100):
            # iterate to find the second point of the training patch mesh on the boundary
            distance_1 = (distance_0 + elSize*scale) # define the second point of the training patch mesh on the boundary

            # interpolate the boundary nodes between the two sampled points
            nodes_0 = interpNodesB[idx_curve_0](distance_0)
            nodes_1 = interpNodesB[idx_curve_0](distance_1)

            # Computes the appropriate scaling and rotation for the normalized training patch mesh
            rotTPB = -np.arctan2(nodes_1[1]-nodes_0[1],nodes_1[0]-nodes_0[0])
            elSizeTPB = np.sqrt(((nodes_1 - nodes_0)**2).sum())
            if abs(elSizeTPB-elSize)/elSize < 0.1:
                break
            
            scale = scale*np.sqrt(elSize/elSizeTPB)

        rot_mat = np.array([[np.cos(rotTPB),np.sin(rotTPB)],[-np.sin(rotTPB),np.cos(rotTPB)]])
        nodesTPB, elementsTPB = GenerateTriangTrainMeshBoundary()
        nodesTPB = (rot_mat@nodesTPB.T).T * elSizeTPB + nodes_0
        isInDomain = IsInDomainF(nodesTPB[[0,3,4,5,6]])
        
        # Check if the training patch mesh is within the domain and the element size is within the bounds
        if np.all(isInDomain):
            break
        iterations += 1
        distance_0 = np.random.uniform(0,length[idx_curve_0])# randomly sample a point on the boundary

    return nodesTPB, elementsTPB, distance_0, distance_1, idx_curve_0


def GetRandomTrainingPatchBoundary(data, nodesIE, distanceBE, nodesD, elSizeMin=0.04, elSizeMax=0.07, elSizeBF=None, debugging=False):
    # Samples a random boundary training patch (2 nodes lie on the boundary) from the dataset of solutions and generates all the relevent outputs
    #   data - list of dictionaries which defines dataset of solutions
    #   nodesIE - interior encoder input mask nodes
    #   s_alphaBE - boundary encoder input mask nodes (defined as distance along the boundary)
    #   elSizeMin, elSizeMax - approximate min, max size of the logest edge in the training patch mesh
    idx = np.random.randint(len(data))
    nodesB = data[idx]['nodesCurves']
    lengthCurves = data[idx]['lengthCurves']
    interpNodesB = data[idx]['interpBC']
    isInDomainF = data[idx]['isInDomainF']
    if elSizeBF is not None:
        elSizeBF = [lambda x: 0.3, lambda x: 0.01]

    # Get random training patch mesh
    # nodesTPB, elementsTPB, distance_0, distance_1, idx_curve_0 = GetRandomTPmeshBoundary(nodesB,lengthCurves,interpNodesB,isInDomainF,elSizeMin,elSizeMax)
    nodesTPB, elementsTPB, distance_0, distance_1, idx_curve_0 = GetRandomTPmeshBoundary(nodesB,lengthCurves,interpNodesB,isInDomainF,elSizeMin=elSizeMin,elSizeMax=elSizeMax,elSizeBF=elSizeBF)

    # Get interior encoder and boundary encoder intpus
    #  interiour encoder inputs: uTP_BEi (physical variabel u values), bTP_BEi (is inside the domain)
    #  boundary encoder inputs: uTP_BEb (physical variabel u values), normalTP_BEb (normal vector of the boundary)
    interpSE = data[idx]['interpSE']
    interpSD = data[idx]['interpSD']
    interpBS = data[idx]['interpBS'][idx_curve_0]
    interpBN = data[idx]['interpBN'][idx_curve_0]

    nodesTP_BEi = GetEncoderInputTP(nodesTPB[[0,3,4,5,6]],nodesIE) # get nodes for the interour encoder input
    uTP_BEi = interpSE(nodesTP_BEi.reshape(-1,2)).reshape(nodesTP_BEi.shape[0],nodesTP_BEi.shape[1],-1)
    bTP_BEi = isInDomainF(nodesTP_BEi.reshape(-1,2)).reshape(nodesTP_BEi.shape[0],nodesTP_BEi.shape[1])
    uTP_BEi[~bTP_BEi] = 0
    bTP_BEi = np.expand_dims(bTP_BEi,-1)
    distanceTP_BEb = np.array([[distance_0],[distance_1]])+distanceBE # get alpha for the boundary encoder input
    uTP_BEb = interpBS(distanceTP_BEb)
    normalTP_BEb = interpBN(distanceTP_BEb)

    nodesTP_D = nodesD+nodesTPB[:1,:]
    u_D = interpSD(nodesTP_D) # sample solution at decoder output mask nodes
    b_D = isInDomainF(nodesTP_D) # sample boundary mask for decoder output (1 if inside the domain, 0 if outside)
    u_D[~b_D] = 0 # set solution values outside of the domain to 0

    if debugging:
        interpBC = data[idx]['interpBC'][idx_curve_0]
        nodesTP_BEb = interpBC(distanceTP_BEb)
        return nodesTPB, elementsTPB, uTP_BEi, bTP_BEi, uTP_BEb, normalTP_BEb, u_D, b_D, nodesTP_BEi, nodesTP_BEb
    else:
        return nodesTPB, elementsTPB, uTP_BEi, bTP_BEi, uTP_BEb, normalTP_BEb, u_D, b_D

def InteriorDatasetGenerator(data, nodesIE, nodesD, batch_size, n_batches, elSizeMin=0.04, elSizeMax=0.07,sRand=0.1):
    for i in range(batch_size*n_batches):
        nodesTP, elementsTP, uTP_IE, bTP_IE, u_D, b_D = GetRandomTrainingPatchInternal(data, nodesIE, nodesD, elSizeMin=elSizeMin, elSizeMax=elSizeMax,sRand=sRand)
        nodesTP = tf.convert_to_tensor(nodesTP, dtype=tf.float32)
        elementsTP = tf.convert_to_tensor(elementsTP, dtype=tf.int32)
        uTP_IE = tf.convert_to_tensor(uTP_IE, dtype=tf.float32)
        bTP_IE = tf.convert_to_tensor(bTP_IE, dtype=tf.float32)
        u_D = tf.convert_to_tensor(u_D, dtype=tf.float32)
        b_D = tf.convert_to_tensor(b_D, dtype=tf.float32)
        yield nodesTP, elementsTP, uTP_IE, bTP_IE, u_D, b_D

def GetInteriorDataset(data, nodesIE, nodesD, batch_size, n_batches, elSizeMin=0.04, elSizeMax=0.07):
    d_IE = data[0]['interpSE'].nDims
    d_D = data[0]['interpSD'].nDims
    dataset = tf.data.Dataset.from_generator(
        lambda: InteriorDatasetGenerator(data, nodesIE, nodesD, batch_size, n_batches, elSizeMin=elSizeMin, elSizeMax=elSizeMax),
        output_types=(tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([7,2]), tf.TensorShape([6,3]), 
                       tf.TensorShape([7,nodesIE.shape[0],d_IE]), tf.TensorShape([7,nodesIE.shape[0],1]), 
                       tf.TensorShape([nodesD.shape[0],d_D]), tf.TensorShape([nodesD.shape[0],1])))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def GetInteriorDatasetFixed(data, nodesIE, nodesD, batch_size, n_batches, elSizeMin=0.04, elSizeMax=0.07,sRand=0.1,variableElSize=False):
    d_IE = data[0]['interpSE'].nDims
    d_D = data[0]['interpSD'].nDims
    dataset_tesor_nodesTP = np.zeros((batch_size*n_batches, 7, 2),dtype=np.float32)
    dataset_tesor_elementsTP = np.zeros((batch_size*n_batches, 6, 3),dtype=np.int32)
    dataset_tesor_uTP_IE = np.zeros((batch_size*n_batches, 7, nodesIE.shape[0], d_IE),dtype=np.float32)
    dataset_tesor_bTP_IE = np.zeros((batch_size*n_batches, 7, nodesIE.shape[0], 1),dtype=np.float32)
    dataset_tesor_u_D = np.zeros((batch_size*n_batches, nodesD.shape[0], d_D),dtype=np.float32)
    dataset_tesor_b_D = np.zeros((batch_size*n_batches, nodesD.shape[0], 1),dtype=np.float32)
    for i in tqdm(range(batch_size*n_batches)):
        nodesTP, elementsTP, uTP_IE, bTP_IE, u_D, b_D = GetRandomTrainingPatchInternal(data, nodesIE, nodesD, elSizeMin=elSizeMin, elSizeMax=elSizeMax,sRand=sRand,variableElSize=variableElSize)
        dataset_tesor_nodesTP[i] = nodesTP
        dataset_tesor_elementsTP[i] = elementsTP
        dataset_tesor_uTP_IE[i] = uTP_IE
        dataset_tesor_bTP_IE[i] = bTP_IE
        dataset_tesor_u_D[i] = u_D
        dataset_tesor_b_D[i] = b_D
    dataset = tf.data.Dataset.from_tensor_slices((dataset_tesor_nodesTP, dataset_tesor_elementsTP, 
                                                  dataset_tesor_uTP_IE, dataset_tesor_bTP_IE, 
                                                  dataset_tesor_u_D, dataset_tesor_b_D))

    dataset = dataset.shuffle(batch_size*n_batches).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def BoundaryDatasetGenerator(data, nodesIE, distanceBE, nodesD, batch_size, n_batches, elSizeMin=0.04, elSizeMax=0.07, elSizeBF=None):
    for i in range(batch_size*n_batches):
        nodesTPB, elementsTPB, uTP_BEi, bTP_BEi, uTP_BEb, normalTP_BEb, u_D, b_D = GetRandomTrainingPatchBoundary(data, nodesIE, distanceBE, nodesD, elSizeMin=elSizeMin, elSizeMax=elSizeMax, elSizeBF=elSizeBF)
        nodesTPB = tf.convert_to_tensor(nodesTPB, dtype=tf.float32)
        elementsTPB = tf.convert_to_tensor(elementsTPB, dtype=tf.int32)
        uTP_BEi = tf.convert_to_tensor(uTP_BEi, dtype=tf.float32)
        bTP_BEi = tf.convert_to_tensor(bTP_BEi, dtype=tf.float32)
        uTP_BEb = tf.convert_to_tensor(uTP_BEb, dtype=tf.float32)
        normalTP_BEb = tf.convert_to_tensor(normalTP_BEb, dtype=tf.float32)
        u_D = tf.convert_to_tensor(u_D, dtype=tf.float32)
        b_D = tf.convert_to_tensor(b_D, dtype=tf.float32)
        yield nodesTPB, elementsTPB, uTP_BEi, bTP_BEi, uTP_BEb, normalTP_BEb, u_D, b_D

def GetBoundaryDataset(data, nodesIE, distanceBE, nodesD, batch_size, n_batches, elSizeMin=0.04, elSizeMax=0.07):
    d_IE = data[0]['interpSE'].nDims
    d_BE = data[0]['interpBS'][0].spline.c.shape[-1]
    d_D = data[0]['interpSD'].nDims

    dataset = tf.data.Dataset.from_generator(
        lambda: BoundaryDatasetGenerator(data, nodesIE, distanceBE, nodesD, batch_size, n_batches, elSizeMin=elSizeMin, elSizeMax=elSizeMax),
        output_types=(tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([7,2]), tf.TensorShape([6,3]), 
                       tf.TensorShape([5,nodesIE.shape[0],d_IE]), tf.TensorShape([5,nodesIE.shape[0],1]), 
                       tf.TensorShape([2,distanceBE.shape[0],d_BE]), tf.TensorShape([2,distanceBE.shape[0],2]), 
                       tf.TensorShape([nodesD.shape[0],d_D]), tf.TensorShape([nodesD.shape[0],1])))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def GetBoundaryDatasetFixed(data, nodesIE, distanceBE, nodesD, batch_size, n_batches, elSizeMin=0.04, elSizeMax=0.07, elSizeBF=None):
    d_IE = data[0]['interpSE'].nDims
    d_BE = data[0]['interpBS'][0].spline.c.shape[-1]
    d_D = data[0]['interpSD'].nDims

    dataset_tesor_nodesTPB = np.zeros((batch_size*n_batches, 7, 2),dtype=np.float32)
    dataset_tesor_elementsTPB = np.zeros((batch_size*n_batches, 6, 3),dtype=np.int32)
    dataset_tesor_uTP_BEi = np.zeros((batch_size*n_batches, 5, nodesIE.shape[0], d_IE),dtype=np.float32)
    dataset_tesor_bTP_BEi = np.zeros((batch_size*n_batches, 5, nodesIE.shape[0], 1),dtype=np.float32)
    dataset_tesor_uTP_BEb = np.zeros((batch_size*n_batches, 2, distanceBE.shape[0], d_BE),dtype=np.float32)
    dataset_tesor_normalTP_BEb = np.zeros((batch_size*n_batches, 2, distanceBE.shape[0], 2),dtype=np.float32)
    dataset_tesor_u_D = np.zeros((batch_size*n_batches, nodesD.shape[0], d_D),dtype=np.float32)
    dataset_tesor_b_D = np.zeros((batch_size*n_batches, nodesD.shape[0], 1),dtype=np.float32)
    for i in tqdm(range(batch_size*n_batches)):
        nodesTPB, elementsTPB, uTP_BEi, bTP_BEi, uTP_BEb, normalTP_BEb, u_D, b_D = GetRandomTrainingPatchBoundary(data, nodesIE, distanceBE, nodesD, elSizeMin=elSizeMin, elSizeMax=elSizeMax, elSizeBF=elSizeBF)
        dataset_tesor_nodesTPB[i] = nodesTPB
        dataset_tesor_elementsTPB[i] = elementsTPB
        dataset_tesor_uTP_BEi[i] = uTP_BEi
        dataset_tesor_bTP_BEi[i] = bTP_BEi
        dataset_tesor_uTP_BEb[i] = uTP_BEb
        dataset_tesor_normalTP_BEb[i] = normalTP_BEb
        dataset_tesor_u_D[i] = u_D
        dataset_tesor_b_D[i] = b_D

    dataset = tf.data.Dataset.from_tensor_slices((dataset_tesor_nodesTPB, dataset_tesor_elementsTPB,
                                                    dataset_tesor_uTP_BEi, dataset_tesor_bTP_BEi,
                                                    dataset_tesor_uTP_BEb, dataset_tesor_normalTP_BEb,
                                                    dataset_tesor_u_D, dataset_tesor_b_D))
    
    dataset = dataset.shuffle(batch_size*n_batches).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def GenerateTriangTrainMeshCorner(v_rand = 0.1):
    # Generates a mesh of 6 triangular elements in a hexagon configuration where the 3 nodes in the left bottom corner have coordinates (0,1), (0,0) and (1,0)
    sqrt_2 = np.sqrt(2)
    nodes = np.array([[(sqrt_2+1)/(2*sqrt_2), (sqrt_2+1)/(2*sqrt_2)],
                      [0,0],
                      [1,0],
                      [1+1/sqrt_2,1/sqrt_2],
                      [1+1/sqrt_2,1+1/sqrt_2],
                      [1/sqrt_2,1+1/sqrt_2],
                      [0,1]])
    
    nodes[[0,3,4,5]] += np.random.uniform(-v_rand,v_rand,(4,2))

    elements = np.arange(3).reshape(1,-1)+np.arange(6).reshape(-1,1)
    elements[:,0] = 0
    elements[elements>6]=1
    return nodes,elements

def GetRandomTPmeshCorner(distaceCornerCurve,idxCurveCorner,interpNodesB,IsInDomainF,elSizeMin=0.04,elSizeMax=0.07,sRand=0.1,idx_corner=None):
    # Get a random boundary training patch mesh (nodesTPB, elementsTPB) from a mesh defined by (nodesB, alpha)
    #   nodesB - [N,2] tensor which contains the x,y coordinates of the boundary nodes

    if idx_corner is None:
        idx_corner = np.random.randint(0,len(distaceCornerCurve))
    
    idx_curve = idxCurveCorner[idx_corner]
    distance_1 = distaceCornerCurve[idx_corner] # distance along the boundary of the node[1]

    iterations = 0

    while iterations < 1_000:

        elSize1 = np.random.uniform(elSizeMin,elSizeMax) # define the element size of the training patch mesh
        elSize2 = np.random.uniform(elSize1*0.8,elSize1*1.2) # define the element size of the training patch mesh
        distance_2 = (distance_1 + elSize1) # distance along the boundary of the node[2]
        distance_6 = (distance_1 - elSize2) # distance along the boundary of the node[6]


        # interpolate the boundary nodes
        nodes_1 = interpNodesB[idx_curve](distance_1)
        nodes_2 = interpNodesB[idx_curve](distance_2)
        nodes_6 = interpNodesB[idx_curve](distance_6)

        # compute the transformation matrix
        R = np.stack([nodes_2-nodes_1,nodes_6-nodes_1],1)

        # generate the training patch mesh and transform it so that fits the corner
        nodesTPC, elementsTPC = GenerateTriangTrainMeshCorner(sRand)
        nodesTPC = (R@nodesTPC.T).T + nodes_1
        isInDomain = IsInDomainF(nodesTPC[[0,3,4,5]])
        # break
        if np.all(isInDomain):
            break
        iterations += 1
        print(iterations)

    # print(iterations)
    return nodesTPC, elementsTPC, distance_1, distance_2, distance_6, idx_curve

def GetRandomTrainingPatchCorner(data, nodesIE, distanceBE, nodesD, elSizeMin=0.04, elSizeMax=0.07,sRand=0.1, debugging=False):
    # Samples a random boundary training patch (2 nodes lie on the boundary) from the dataset of solutions and generates all the relevent outputs
    #   data - list of dictionaries which defines dataset of solutions
    #   nodesIE - interior encoder input mask nodes
    #   s_alphaBE - boundary encoder input mask nodes (defined as distance along the boundary)
    #   elSizeMin, elSizeMax - approximate min, max size of the logest edge in the training patch mesh
    idx = np.random.randint(len(data))

    distaceCornerCurve = data[idx]['distaceCornerCurve']
    idxCurveCorner = data[idx]['idxCurveCorner']
    interpNodesB = data[idx]['interpBC']
    isInDomainF = data[idx]['isInDomainF']
    nodesTPC, elementsTPC, distance_1, distance_2, distance_6, idx_curve = GetRandomTPmeshCorner(distaceCornerCurve,idxCurveCorner,interpNodesB,isInDomainF,elSizeMin,elSizeMax,sRand=sRand)

    # Get interior encoder and boundary encoder intpus
    #  interiour encoder inputs: uTP_BEi (physical variabel u values), bTP_BEi (is inside the domain)
    #  boundary encoder inputs: uTP_BEb (physical variabel u values), normalTP_BEb (normal vector of the boundary)
    interpSE = data[idx]['interpSE']
    interpSD = data[idx]['interpSD']
    interpBS = data[idx]['interpBS'][idx_curve]
    interpBN = data[idx]['interpBN'][idx_curve]

    nodesTP_CEi = GetEncoderInputTP(nodesTPC[[0,3,4,5]],nodesIE) # get nodes for the interour encoder input
    uTP_CEi = interpSE(nodesTP_CEi.reshape(-1,2)).reshape(nodesTP_CEi.shape[0],nodesTP_CEi.shape[1],-1)
    bTP_CEi = isInDomainF(nodesTP_CEi.reshape(-1,2)).reshape(nodesTP_CEi.shape[0],nodesTP_CEi.shape[1])
    uTP_CEi[~bTP_CEi] = 0
    bTP_CEi = np.expand_dims(bTP_CEi,-1)
    distanceTP_CEb = np.array([[distance_1],[distance_2],[distance_6]])+distanceBE # get alpha for the boundary encoder input
    uTP_CEb = interpBS(distanceTP_CEb)
    normalTP_CEb = interpBN(distanceTP_CEb)

    nodesTP_D = nodesD+nodesTPC[:1,:]
    u_D = interpSD(nodesTP_D) # sample solution at decoder output mask nodes
    b_D = isInDomainF(nodesTP_D) # sample boundary mask for decoder output (1 if inside the domain, 0 if outside)
    u_D[~b_D] = 0 # set solution values outside of the domain to 0

    if debugging:
        interpBC = data[idx]['interpBC'][idx_curve]
        nodesTP_CEb = interpBC(distanceTP_CEb)
        return nodesTPC, elementsTPC, uTP_CEi, bTP_CEi, uTP_CEb, normalTP_CEb, u_D, b_D, nodesTP_CEi, nodesTP_CEb
    else:
        return nodesTPC, elementsTPC, uTP_CEi, bTP_CEi, uTP_CEb, normalTP_CEb, u_D, b_D
    
def CornerDatasetGenerator(data, nodesIE, distanceBE, nodesD, batch_size, n_batches, elSizeMin=0.04, elSizeMax=0.07):
    for i in range(batch_size*n_batches):
        nodesTPC, elementsTPC, uTP_CEi, bTP_CEi, uTP_CEb, normalTP_CEb, u_D, b_D =  GetRandomTrainingPatchCorner(data, nodesIE, distanceBE, nodesD, elSizeMin, elSizeMax)
        nodesTPC = tf.convert_to_tensor(nodesTPC, dtype=tf.float32)
        elementsTPC = tf.convert_to_tensor(elementsTPC, dtype=tf.int32)
        uTP_CEi = tf.convert_to_tensor(uTP_CEi, dtype=tf.float32)
        bTP_CEi = tf.convert_to_tensor(bTP_CEi, dtype=tf.float32)
        uTP_CEb = tf.convert_to_tensor(uTP_CEb, dtype=tf.float32)
        normalTP_CEb = tf.convert_to_tensor(normalTP_CEb, dtype=tf.float32)
        u_D = tf.convert_to_tensor(u_D, dtype=tf.float32)
        b_D = tf.convert_to_tensor(b_D, dtype=tf.float32)
        yield nodesTPC, elementsTPC, uTP_CEi, bTP_CEi, uTP_CEb, normalTP_CEb, u_D, b_D

def GetCornerDataset(data, nodesIE, distanceBE, nodesD, batch_size, n_batches, elSizeMin=0.04, elSizeMax=0.07):
    d_IE = data[0]['interpSE'].nDims
    d_BE = data[0]['interpBS'][0].spline.c.shape[-1]
    d_D = data[0]['interpSD'].nDims

    dataset = tf.data.Dataset.from_generator(
        lambda: CornerDatasetGenerator(data, nodesIE, distanceBE, nodesD, batch_size, n_batches, elSizeMin=elSizeMin, elSizeMax=elSizeMax),
        output_types=(tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([7,2]), tf.TensorShape([6,3]), 
                       tf.TensorShape([4,nodesIE.shape[0],d_IE]), tf.TensorShape([4,nodesIE.shape[0],1]), 
                       tf.TensorShape([3,distanceBE.shape[0],d_BE]), tf.TensorShape([3,distanceBE.shape[0],2]), 
                       tf.TensorShape([nodesD.shape[0],d_D]), tf.TensorShape([nodesD.shape[0],1])))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def GetCornerDatasetFixed(data, nodesIE, distanceBE, nodesD, batch_size, n_batches, elSizeMin=0.04, elSizeMax=0.07,sRand=0.1):
    d_IE = data[0]['interpSE'].nDims
    d_BE = data[0]['interpBS'][0].spline.c.shape[-1]
    d_D = data[0]['interpSD'].nDims

    dataset_tesor_nodesTPC = np.zeros((batch_size*n_batches, 7, 2),dtype=np.float32)
    dataset_tesor_elementsTPC = np.zeros((batch_size*n_batches, 6, 3),dtype=np.int32)
    dataset_tesor_uTP_CEi = np.zeros((batch_size*n_batches, 4, nodesIE.shape[0], d_IE),dtype=np.float32)
    dataset_tesor_bTP_CEi = np.zeros((batch_size*n_batches, 4, nodesIE.shape[0], 1),dtype=np.float32)
    dataset_tesor_uTP_CEb = np.zeros((batch_size*n_batches, 3, distanceBE.shape[0], d_BE),dtype=np.float32)
    dataset_tesor_normalTP_CEb = np.zeros((batch_size*n_batches, 3, distanceBE.shape[0], 2),dtype=np.float32)
    dataset_tesor_u_D = np.zeros((batch_size*n_batches, nodesD.shape[0], d_D),dtype=np.float32)
    dataset_tesor_b_D = np.zeros((batch_size*n_batches, nodesD.shape[0], 1),dtype=np.float32)
    for i in tqdm(range(batch_size*n_batches)):
        nodesTPC, elementsTPC, uTP_CEi, bTP_CEi, uTP_CEb, normalTP_CEb, u_D, b_D =  GetRandomTrainingPatchCorner(data, nodesIE, distanceBE, nodesD, elSizeMin, elSizeMax,sRand=sRand)
        dataset_tesor_nodesTPC[i] = nodesTPC
        dataset_tesor_elementsTPC[i] = elementsTPC
        dataset_tesor_uTP_CEi[i] = uTP_CEi
        dataset_tesor_bTP_CEi[i] = bTP_CEi
        dataset_tesor_uTP_CEb[i] = uTP_CEb
        dataset_tesor_normalTP_CEb[i] = normalTP_CEb
        dataset_tesor_u_D[i] = u_D
        dataset_tesor_b_D[i] = b_D

    dataset = tf.data.Dataset.from_tensor_slices((dataset_tesor_nodesTPC, dataset_tesor_elementsTPC,
                                                    dataset_tesor_uTP_CEi, dataset_tesor_bTP_CEi,
                                                    dataset_tesor_uTP_CEb, dataset_tesor_normalTP_CEb,
                                                    dataset_tesor_u_D, dataset_tesor_b_D))
    
    dataset = dataset.shuffle(batch_size*n_batches).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def GetEncoder(N_interiorEncoder,n_latent,layers,d_input=1,activation='tanh'):
    inputU = Input((N_interiorEncoder,d_input))
    inputB = Input((N_interiorEncoder,1))
    x = Concatenate(axis = 1)([Flatten()(inputU),Flatten()(inputB)])
    for i in range(len(layers)):
        x = Dense(layers[i],activation=activation)(x)
    latent = Dense(n_latent)(x)
    encoder = Model([inputU,inputB],latent,name='encoder')
    return encoder

def GetEncoderBoundary(N_boundarEncoder,n_latent,layers,d_input=1,activation='tanh'):
    inputU = Input((N_boundarEncoder,d_input))
    inputB = Input((N_boundarEncoder,2))
    x = Concatenate(axis = 1)([Flatten()(inputU),Flatten()(inputB)])
    for i in range(len(layers)):
        x = Dense(layers[i],activation=activation)(x)
    latent = Dense(n_latent)(x)
    encoderB = Model([inputU,inputB],latent,name='encoderB')
    return encoderB

def GetDecoder(N_decoderOut,n_latent,layers,d_out=1,activation='tanh'):
    input = Input((n_latent))
    x = input
    for i in range(len(layers)-1,-1,-1):
        x = Dense(layers[i],activation=activation)(x)
    output = Dense(N_decoderOut*d_out)(x)
    output = Reshape((N_decoderOut,d_out))(output)
    decoder = Model(input,output,name='decoder')
    return decoder

def GetModels(hyperParams):
    # Get the encoder, boundary encoder and decoder models based on the hyperparameters
    N_interiorEncoder = hyperParams['N_interiorEncoder']
    d_interiorEncoder = hyperParams['d_interiorEncoder']
    layers_interiorEncoder = hyperParams['layers_interiorEncoder']
    N_boundarEncoder = hyperParams['N_boundarEncoder']
    d_boundarEncoder = hyperParams['d_boundarEncoder']
    layers_boundarEncoder = hyperParams['layers_boundarEncoder']
    N_decoderOut = hyperParams['N_decoderOut']
    d_decoderOut = hyperParams['d_decoderOut']
    layers_decoder = hyperParams['layers_decoder']
    n_latent = hyperParams['n_latent']
    activation = hyperParams['activation']
    encoder = GetEncoder(N_interiorEncoder,n_latent,layers_interiorEncoder,d_interiorEncoder,activation)
    encoderB = GetEncoderBoundary(N_boundarEncoder,n_latent,layers_boundarEncoder,d_boundarEncoder,activation)
    decoder = GetDecoder(N_decoderOut,n_latent,layers_decoder,d_decoderOut,activation)
    return encoder,encoderB,decoder

def GetK_el_triang(A,nodes):
    r = int(A.shape[0]/2)
    b = np.roll(nodes[:,1],1) - np.roll(nodes[:,1],2)
    c = np.roll(nodes[:,0],2) - np.roll(nodes[:,0],1)
    Area = np.abs(np.dot(nodes[:,0],b))/2
    B = np.concatenate([
        np.concatenate([b[i]*np.eye(r) for i in range(3)],1),
        np.concatenate([c[i]*np.eye(r) for i in range(3)],1)
    ],0)/(2*Area)
    return np.dot(np.dot(B.T,A),B)*Area


def SolveFEM(nodes, elements, boundaryNodes, l_BC, internalNodes, r, A, A_nl=False, l=None):
    if l is None:
        l = np.zeros((nodes.shape[0], r))
    if not A_nl:
        A_l = A

    # Assemble the global stiffness matrix
    K = np.zeros((nodes.shape[0]*r, nodes.shape[0]*r))
    for el in elements:
        el_idx = [[r*k+j for j in range(r)] for k in el]
        el_idx = np.concatenate(el_idx)
        nodes_el = tf.gather(nodes, indices=el)
        X_idx,Y_idx = np.meshgrid(el_idx,el_idx)
        if A_nl:
            A_l = A(l[el_idx])
        # print(A_l)
        K_el = GetK_el_triang(A_l,nodes_el)
        K[Y_idx,X_idx] += K_el


    bc_idx = [[r*i+j for j in range(r)] for i in boundaryNodes]
    bc_idx = np.concatenate(bc_idx)
    internal_idx = [[r*i+j for j in range(r)] for i in internalNodes]
    internal_idx = np.concatenate(internal_idx)

    f = - (K[:,bc_idx] @ l_BC.flatten().reshape(-1,1))

    K_BC = K[internal_idx,:][:,internal_idx]
    f = f[internal_idx]

    # Solve the system
    l_internal = np.linalg.solve(K_BC, f)
    n_CDOF = int(l_internal.shape[0]/r)
    l_internal = l_internal.reshape(n_CDOF, r)

    l[internalNodes,:] = l_internal
    l[boundaryNodes,:] = l_BC.reshape(-1,r)
    return l

def SINNsPredict(data,encoderB,decoder,B,r,distanceBE):
    nodes = data['nodes']
    elements = data['elements']
    boundaryNodes = np.concatenate([idxCurve[:-1] for idxCurve in data['idxCurves']])
    internalNodes = np.setdiff1d(np.unique(elements.flatten()),boundaryNodes)
    u_D = data['solution'][...,:2]
    # distance= data['distanceCurves']
    distance = [d[:-1] for d in data['distanceCurves']]
    interpBS = data['interpBS']
    interpBN = data['interpBN']
    nd_BE = data['interpBS'][0].spline.c.shape[-1]

    dBE_S = [(d.reshape(-1,1) + distanceBE.reshape(1,-1)) for d in distance]
    u_BE = [interpBS[i](d.flatten()).reshape(d.shape[0],d.shape[1],nd_BE) for i,d in enumerate(dBE_S)]
    u_BE = np.concatenate(u_BE,0)
    normal_BE = [interpBN[i](d.flatten()).reshape(d.shape[0],d.shape[1],2) for i,d in enumerate(dBE_S)]
    normal_BE = np.concatenate(normal_BE,0)

    lB = encoderB([u_BE,normal_BE])


    l_model = SolveFEM(nodes, elements, boundaryNodes, lB.numpy(), internalNodes, r, GetA(B).numpy())
    l_modelI = l_model[internalNodes]
    u_modelI = decoder(l_modelI)
    u_model = u_D.copy()
    u_model[internalNodes] = u_modelI[:,0,:].numpy()
    return u_model, l_model

def EncodeData(data,encoder,encoderB, nodesIE, distanceBE):
    nodes = data['nodes']
    boundaryNodes = np.concatenate([idxCurve[:-1] for idxCurve in data['idxCurves']])
    internalNodes = np.setdiff1d(np.arange(nodes.shape[0]),boundaryNodes)
    nodesI = nodes[internalNodes]
    distance = [d[:-1] for d in data['distanceCurves']]
    interpSE = data['interpSE']
    interpBS = data['interpBS']
    interpBN = data['interpBN']
    isInDomainF = data['isInDomainF']
    # nd_IE = data['interpSE'].nDims
    nd_BE = data['interpBS'][0].spline.c.shape[-1]
    # nd_D = data['interpSD'].nDims

    dBE_S = [(d.reshape(-1,1) + distanceBE.reshape(1,-1)) for d in distance]
    u_BE = [interpBS[i](d.flatten()).reshape(d.shape[0],d.shape[1],nd_BE) for i,d in enumerate(dBE_S)]
    u_BE = np.concatenate(u_BE,0)
    normal_BE = [interpBN[i](d.flatten()).reshape(d.shape[0],d.shape[1],2) for i,d in enumerate(dBE_S)]
    normal_BE = np.concatenate(normal_BE,0)

    lB = encoderB([u_BE,normal_BE])

    nodesI_IE = GetEncoderInputTP(nodesI,nodesIE)
    uI_IE = interpSE(nodesI_IE.reshape(-1,2)).reshape(nodesI_IE.shape[0],nodesI_IE.shape[1],-1)
    bI_IE = isInDomainF(nodesI_IE.reshape(-1,2)).reshape(nodesI_IE.shape[0],nodesI_IE.shape[1],-1)
    uI_IE[~bI_IE[:,:,0]] = 0.0

    lI = encoder([uI_IE,bI_IE])

    l = np.zeros((nodes.shape[0],lI.shape[-1]))
    l[internalNodes] = lI
    l[boundaryNodes] = lB
    return l

def PlotFEMsolution(nodes, elements,l, axisLabels = True, useColorBar = True, labels = False, subplotsize=(6,5),dpi = 100):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    if elements.shape[1] == 4:
        # Convert quadrlateral mesh to triangular mesh
        elements = np.concatenate([elements[:,:3],elements[:,1:]],0)

    # Create a Triangulation object
    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    # Plotting
    r = l.shape[1]
    n_rows = np.ceil(r/3).astype(int)
    n_cols = min(r,3)
    plt.figure(figsize=(subplotsize[0]*n_cols,subplotsize[1]*n_rows),dpi=dpi)
    for i in range(r):
        plt.subplot(n_rows,n_cols,i+1)
        plt.tricontourf(triangulation, l[:,i],10)
        # plt.scatter(nodes[:,0],nodes[:,1],s=100,c=l[:,i])
        if useColorBar:
            plt.colorbar()
        if axisLabels:
            plt.xlabel('$x$')
            plt.ylabel('$y$')
        else:
            plt.axis('off')
        if labels:
            plt.title(labels[i])
        plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    return

def SINNsEvalErr(data,encoderB,decoder,B,r,distanceBE):
    u_model,_  = SINNsPredict(data,encoderB,decoder,B,r,distanceBE)
    u = data['solution'][...,:2]
    return np.mean(np.square(u_model-u))

def SINNsEvalErr2(data,encoderB,decoder,B,r,s_alphaBE):
    u_model,_ = SINNsPredict(data,encoderB,decoder,B,r,s_alphaBE)
    u = data['solution'][...,:2]
    mse = np.mean(np.square(u_model-u))
    eta = (np.max(u_model-u,0)/(np.max(u,0)-np.min(u,0))).mean()
    return mse,eta

def GetK_tf(nodes_el, A, r):
    # Get the stiffness matrix for a triangular elements, function is vectorized so that a batch of elements can be processed at once
    #   nodes_el - [N,3,2] tensor which containes the x,y positions of the 3 nodes of N triangles
    #   A - [2*r,2*r] tensor which defines the linear PDE coefficients
    #   r - number of latent variables

    # calculate the triangle area and the B matrix which is a derivative of the shape functions (B = div(N))
    b = (tf.roll(nodes_el[:,:,1],1,axis=1) - tf.roll(nodes_el[:,:,1],2,axis=1))
    b = tf.reshape(b,(-1,3,1))
    c = (tf.roll(nodes_el[:,:,0],2,axis=1) - tf.roll(nodes_el[:,:,0],1,axis=1))
    c = tf.reshape(c,(-1,3,1))
    Area = tf.abs(tf.matmul(tf.reshape(nodes_el[:,:,0],(-1,1,3)),b))/2
    I = tf.reshape(tf.eye(r),(1,r,r))
    B = tf.concat([
        tf.concat([b[:,0:1]*I, b[:,1:2]*I, b[:,2:3]*I],-1),
        tf.concat([c[:,0:1]*I, c[:,1:2]*I, c[:,2:3]*I],-1)
    ],-2)/(2*Area)
    B_T = tf.transpose(B,(0,2,1))
    return tf.matmul(tf.matmul(B_T,A),B)*Area # calculates the stiffness matrix

def PredictLatentC(nodes_el, l_el_b, A, r):
    # Predicts the latent value of a cetral node of a training patch mesh made up of 6 triangular elements in the shape of hexagon
    # Theoretically could be extended to more general training patch mesh but has not been done yet
    #   nodes_el - [N,6,3,2] tensor which containes the x,y positions of the 3 nodes of 6 triangle elements of N training patches
    #   l_el_b - [N,6,2,r] tensor which containes the latent values of the boundary nodes of the 6 triangle elements of N training patches
    #   A - [2*r,2*r] tensor which defines the linear PDE coefficients
    #   r - number of latent variables
    l_el_b = tf.concat([l_el_b[...,0,:],l_el_b[...,1,:]],-1)[...,tf.newaxis]
    nodes_el_flat = tf.reshape(nodes_el,(-1,nodes_el.shape[-2],2))
    K_el_flat = GetK_tf(nodes_el_flat, A, r)
    K_el = tf.reshape(K_el_flat,(nodes_el.shape[0],nodes_el.shape[1],K_el_flat.shape[-2],K_el_flat.shape[-1]))
    K = tf.reduce_sum(K_el[:,:,:r,:r],1)
    b = -K_el[:,:,:r,r:]@l_el_b
    b = tf.reduce_sum(b,1)
    l_c = tf.linalg.solve(K,b)
    return l_c[...,0]

def GetA(B):
    A = tf.matmul(B,B,transpose_a=True)
    return A

# def LossFunc(u_D,u_D_pred,b_D,l_c,l_c_pred,A,fd_l_weight,ellicpit_weight):
#     r_loss = tf.reduce_mean(tf.square((u_D-u_D_pred)*b_D),axis=(1,2))
#     fd_l_loss = tf.reduce_mean(tf.square(l_c-l_c_pred),axis=(1))
#     elliptic_loss = -tf.math.log(tf.linalg.det(A))
#     loss = r_loss + fd_l_weight*fd_l_loss + ellicpit_weight*elliptic_loss
#     return loss, (tf.reduce_mean(loss),tf.reduce_mean(r_loss),tf.reduce_mean(fd_l_loss),tf.reduce_mean(elliptic_loss))

def LossFunc2(u_D,u_D_pred,b_D,l_enc,l_c_pred,A,fd_l_weight,ellicpit_weight):
    r_loss = tf.reduce_mean(tf.square((u_D-u_D_pred)*b_D),axis=(1,2))
    # fd_l_loss = tf.reduce_mean(tf.square(l_enc[:,0,:]-l_c_pred)/tf.math.reduce_variance(l_enc,axis=1),axis=(1))
    fd_l_loss = tf.reduce_mean(tf.square(l_enc[:,0,:]-l_c_pred)/tf.math.pow(tf.math.reduce_variance(l_enc,axis=1),0.8),axis=(1))
    # fd_l_loss = tf.reduce_mean(tf.square(l_enc[:,0,:]-l_c_pred)/tf.math.reduce_std(l_enc,axis=1),axis=(1))
    elliptic_loss = -tf.math.log(tf.linalg.det(A))
    loss = r_loss + fd_l_weight*fd_l_loss + ellicpit_weight*elliptic_loss
    return loss, (tf.reduce_mean(loss),tf.reduce_mean(r_loss),tf.reduce_mean(fd_l_loss),tf.reduce_mean(elliptic_loss))

def InteriorForwardPass(encoder, decoder, A, r, nodesTP, elementsTP, uTP_IE, bTP_IE, u_D, b_D, fd_l_weight, elliptc_weight):
    uTP_IE_reshaped = tf.reshape(uTP_IE,(uTP_IE.shape[0]*uTP_IE.shape[1],uTP_IE.shape[2],uTP_IE.shape[3]))
    bTP_IE_reshaped = tf.reshape(bTP_IE,(bTP_IE.shape[0]*bTP_IE.shape[1],bTP_IE.shape[2]))
    l_reshaped = encoder([uTP_IE_reshaped,bTP_IE_reshaped])
    l = tf.reshape(l_reshaped,(uTP_IE.shape[0],uTP_IE.shape[1],l_reshaped.shape[-1]))
    l_el_b = tf.gather(l,elementsTP[0,:,1:], axis=1)
    nodes_el = tf.gather(nodesTP,elementsTP[0], axis=1)
    l_c_pred = PredictLatentC(nodes_el,l_el_b,A,r)
    u_D_pred = decoder(l_c_pred)
    # loss, logloss = LossFunc(u_D,u_D_pred,b_D,l[:,0],l_c_pred,A,fd_l_weight,elliptc_weight)
    loss, logloss = LossFunc2(u_D,u_D_pred,b_D,l,l_c_pred,A,fd_l_weight,elliptc_weight)
    return loss, logloss


def BoundaryForwardPass(encoder, encoderB, decoder, A, r, nodesTPB, elementsTPB, uTP_BEi, bTP_BEi, uTP_BEb, normalTP_BEb, u_D, b_D, fd_l_weight, elliptc_weight):
    uTP_BEi_reshaped = tf.reshape(uTP_BEi,(uTP_BEi.shape[0]*uTP_BEi.shape[1],*uTP_BEi.shape[2:]))
    bTP_BEi_reshaped = tf.reshape(bTP_BEi,(bTP_BEi.shape[0]*bTP_BEi.shape[1],*bTP_BEi.shape[2:]))
    l_i_reshaped = encoder([uTP_BEi_reshaped,bTP_BEi_reshaped])
    l_i = tf.reshape(l_i_reshaped,(uTP_BEi.shape[0],uTP_BEi.shape[1],l_i_reshaped.shape[-1]))
    uTP_BEb_reshaped = tf.reshape(uTP_BEb,(uTP_BEb.shape[0]*uTP_BEb.shape[1],*uTP_BEb.shape[2:]))
    normalTP_BEb_reshaped = tf.reshape(normalTP_BEb,(normalTP_BEb.shape[0]*normalTP_BEb.shape[1],*normalTP_BEb.shape[2:]))
    l_b_reshaped = encoderB([uTP_BEb_reshaped,normalTP_BEb_reshaped])
    l_b = tf.reshape(l_b_reshaped,(uTP_BEb.shape[0],uTP_BEb.shape[1],l_b_reshaped.shape[-1]))
    l = tf.concat([l_i[:,0:1],l_b,l_i[:,1:]],-2)
    l_el_b = tf.gather(l,elementsTPB[0,:,1:], axis=1)
    nodes_el = tf.gather(nodesTPB,elementsTPB[0], axis=1)
    l_c_pred = PredictLatentC(nodes_el,l_el_b,A,r)
    u_D_pred = decoder(l_c_pred)
    # loss, logloss = LossFunc(u_D,u_D_pred,b_D,l[:,0],l_c_pred,A,fd_l_weight,elliptc_weight)
    loss, logloss = LossFunc2(u_D,u_D_pred,b_D,l,l_c_pred,A,fd_l_weight,elliptc_weight)
    return loss, logloss

def CornerForwardPass(encoder, encoderB, decoder, A, r, nodesTPC, elementsTPC, uTP_CEi, bTP_CEi, uTP_CEb, normalTP_CEb, u_D, b_D, fd_l_weight, elliptc_weight):
    uTP_CEi_reshaped = tf.reshape(uTP_CEi,(uTP_CEi.shape[0]*uTP_CEi.shape[1],*uTP_CEi.shape[2:]))
    bTP_CEi_reshaped = tf.reshape(bTP_CEi,(bTP_CEi.shape[0]*bTP_CEi.shape[1],*bTP_CEi.shape[2:]))
    l_i_reshaped = encoder([uTP_CEi_reshaped,bTP_CEi_reshaped])
    l_i = tf.reshape(l_i_reshaped,(uTP_CEi.shape[0],uTP_CEi.shape[1],l_i_reshaped.shape[-1]))
    uTP_CEb_reshaped = tf.reshape(uTP_CEb,(uTP_CEb.shape[0]*uTP_CEb.shape[1],*uTP_CEb.shape[2:]))
    normalTP_CEb_reshaped = tf.reshape(normalTP_CEb,(normalTP_CEb.shape[0]*normalTP_CEb.shape[1],*normalTP_CEb.shape[2:]))
    l_b_reshaped = encoderB([uTP_CEb_reshaped,normalTP_CEb_reshaped])
    l_b = tf.reshape(l_b_reshaped,(uTP_CEb.shape[0],uTP_CEb.shape[1],l_b_reshaped.shape[-1]))
    l = tf.concat([l_i[:,0:1],l_b[:,:2],l_i[:,1:],l_b[:,2:3]],-2)
    l_el_b = tf.gather(l,elementsTPC[0,:,1:], axis=1)
    nodes_el = tf.gather(nodesTPC,elementsTPC[0], axis=1)
    l_c_pred = PredictLatentC(nodes_el,l_el_b,A,r)
    u_D_pred = decoder(l_c_pred)
    # loss, logloss = LossFunc(u_D,u_D_pred,b_D,l[:,0],l_c_pred,A,fd_l_weight,elliptc_weight)
    loss, logloss = LossFunc2(u_D,u_D_pred,b_D,l,l_c_pred,A,fd_l_weight,elliptc_weight)
    return loss, logloss


@tf.function
def TrainStep(encoder, encoderB, decoder, B, optimizer, batchI, batchB, batchC, r, fd_l_weight, elliptc_weight):

    with tf.GradientTape() as enc_tape, tf.GradientTape() as encB_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as b_tape:
        A = GetA(B)
        lossI, loglossI = InteriorForwardPass(encoder, decoder, A, r, *batchI, fd_l_weight, elliptc_weight)
        lossB, loglossB = BoundaryForwardPass(encoder, encoderB, decoder, A, r, *batchB, fd_l_weight, elliptc_weight)
        lossC, loglossC = CornerForwardPass(encoder, encoderB, decoder, A, r, *batchC, fd_l_weight, elliptc_weight)
        loss = tf.concat([lossI,lossB,lossC],0)

    grads_enc = enc_tape.gradient(loss,encoder.trainable_variables)
    grads_encB = encB_tape.gradient(loss,encoderB.trainable_variables)
    grads_dec = dec_tape.gradient(loss,decoder.trainable_variables)
    grads_b = b_tape.gradient(loss,B)
    optimizer.apply_gradients(zip(grads_enc,encoder.trainable_variables))
    optimizer.apply_gradients(zip(grads_dec,decoder.trainable_variables))
    optimizer.apply_gradients(zip(grads_encB,encoderB.trainable_variables))
    optimizer.apply_gradients(zip([grads_b],[B]))

    return loglossI+loglossB+loglossC

def TrainModel(encoder, encoderB, decoder, B, optimizer, dataset, dataset_b, dataset_c, r, epochs, n_batches, hyperParams, history=None, data_test=None, data_train=None):
    fd_l_weight = hyperParams['fd_l_weight']
    elliptc_weight = hyperParams['elliptc_weight']
    distanceBE = np.array(hyperParams['distanceBE'])
    if history is None:
        history = {'loss':[], 'loss_i':[], 'loss_b':[], 'loss_c': [],'r_loss_i':[],'r_loss_b':[],'r_loss_c':[],'fd_l_loss_i':[],'fd_l_loss_b':[],'fd_l_loss_c':[],'elliptic_loss':[],'err_train':[],'err_test':[],'err_epoch':[],'eta_test':[],'eta_train':[]}
    history_epoch = np.zeros((n_batches,12))
    for epoch in range(epochs):
        for i, (batchI, batchB, batchC) in enumerate(zip(dataset,dataset_b,dataset_c)):
            logloss = TrainStep(encoder, encoderB, decoder, B, optimizer, batchI, batchB, batchC, r, fd_l_weight, elliptc_weight)
            history_epoch[i,:] = logloss
            print(f'epoch: {epoch+1}/{epochs}; batch: {i+1}/{n_batches}; loss: {history_epoch[:,0].mean():.4g}', end='\r')
        
        logloss = history_epoch.mean(0)
        history['loss'].append((logloss[0]+logloss[4]+logloss[8])/3)
        history['loss_i'].append(logloss[0])
        history['loss_b'].append(logloss[4])
        history['loss_c'].append(logloss[8])
        history['r_loss_i'].append(logloss[1])
        history['r_loss_b'].append(logloss[5])
        history['r_loss_c'].append(logloss[9])
        history['fd_l_loss_i'].append(logloss[2])
        history['fd_l_loss_b'].append(logloss[6])
        history['fd_l_loss_c'].append(logloss[10])
        history['elliptic_loss'].append(logloss[3])
        print(f'epoch: {epoch+1}/{epochs}; batch: {i+1}/{n_batches}; loss: {history_epoch[:,0].mean():.4g}', end='')

        if (data_test is not None) and ((epoch+1)%5 == 0):
            mse_arr = np.zeros((len(data_test)))
            eta_arr = np.zeros((len(data_test)))
            for i in range(len(data_test)):
                mse_arr[i], eta_arr[i] = SINNsEvalErr2(data_test[i],encoderB,decoder,B,r,distanceBE)
            history['err_test'].append(mse_arr.mean())
            history['eta_test'].append(eta_arr.mean())
            history['err_epoch'].append(len(history['loss'])-1)
            print(f'; err_test: {mse_arr.mean():.4g}', end='')
            if (data_train is not None):
                mse_arr_train = np.zeros((len(data_train)))
                eta_arr_train = np.zeros((len(data_train)))
                for i in range(len(data_train)):
                    # err_arr_train[i] = SINNsEvalErr(data_train[i],encoderB,decoder,B,r,distanceBE)
                    mse_arr_train[i], eta_arr_train[i] = SINNsEvalErr2(data_train[i],encoderB,decoder,B,r,distanceBE)
                history['err_train'].append(mse_arr_train.mean())
                history['eta_train'].append(eta_arr_train.mean())
                print(f'; err_train: {mse_arr_train.mean():.4g}', end='')
            print()
        else:
            print()

    return history

def Save(modelFolder,encoder,encoderB,decoder,P,history,hyperParams):
    # Saves the all SINNs models, the training history and the hyper parameters
    if not os.path.exists(modelFolder):
        # Create folder for the model if it does not exist
        os.makedirs(modelFolder)      
    
    # Save models within the folder
    encoder.save(modelFolder+"/encoder.keras")
    encoderB.save(modelFolder+"/encoderB.keras")
    decoder.save(modelFolder+"/decoder.keras")
    np.save(modelFolder+"/P.npy",P.numpy())
    with open( modelFolder+'/history.json', 'w') as fp:
        json.dump(history, fp)
    with open( modelFolder+'/hyperParams.json', 'w') as fp:
        json.dump(hyperParams, fp)

def LoadModels(modelFolder):
    # Load SINNs models, the training history and the hyper parameters
    encoder = tf.keras.models.load_model(modelFolder+"/encoder.keras")
    encoderB = tf.keras.models.load_model(modelFolder+"/encoderB.keras")
    decoder = tf.keras.models.load_model(modelFolder+"/decoder.keras")
    P = np.load(modelFolder+"/P.npy")
    P = tf.Variable(tf.constant(P,tf.float32))
    with open(modelFolder+'/history.json', 'r') as fp:
        history = json.load(fp)
    with open(modelFolder+'/hyperParams.json', 'r') as fp:
        hyperParams = json.load(fp)
    return encoder,encoderB,decoder,P,history,hyperParams

def polygon_area(points):
    n = len(points)
    if n < 3:
        return 0  # Not a polygon
    
    area = 0
    for i in range(n-1):
        j = (i + 1)
        x_i, y_i = points[i]
        x_j, y_j = points[j]
        area += x_i * y_j - x_j * y_i

    return abs(area) / 2