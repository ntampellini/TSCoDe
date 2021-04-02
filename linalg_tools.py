import numpy as np
from scipy.spatial.transform import Rotation as R

def norm(vec):
    return vec / np.linalg.norm(vec)

def polygonize(lengths):
    '''
    Returns coordinates for the polygon vertexes used in cyclical TS construction,
    as a list of vector couples specifying starting and ending point of each pivot 
    vector. For bimolecular TSs, returns vertexes for the centered superposition of
    two segments. For trimolecular TSs, returns triangle vertexes.

    :params vertexes: list of floats, used as polygon side lenghts.
    :return vertexes_out: list of vectors couples (start, end)
    '''
    assert len(lengths) in (2,3)
    lengths = sorted(lengths)
    arr = np.zeros((len(lengths),2,3))

    if len(lengths) == 2:
        arr[0,0] = np.array([-lengths[0]/2,0,0])
        arr[0,1] = np.array([+lengths[0]/2,0,0])
        arr[1,0] = np.array([-lengths[1]/2,0,0])
        arr[1,1] = np.array([+lengths[1]/2,0,0])

        vertexes_out = np.vstack(([arr],[arr]))
        vertexes_out[1,1] *= -1

    else:
        arr[0,1] = np.array([lengths[0],0,0])
        arr[1,0] = np.array([lengths[0],0,0])

        a = np.power(lengths[0], 2)
        b = np.power(lengths[1], 2)
        c = np.power(lengths[2], 2)
        x = (a-b+c)/(2*a**0.5)
        y = (c-x**2)**0.5

        arr[1,1] = np.array([x,y,0])
        arr[2,0] = np.array([x,y,0])

        vertexes_out = np.vstack(([arr],[arr],[arr],[arr],
                                  [arr],[arr],[arr],[arr]))

        swaps = [(1,2),(2,1),(3,1),(3,2),(4,0),(5,0),(5,1),(6,0),(6,2),(7,0),(7,1),(7,2)]

        for t,v in swaps:
            # triangle, vector couples to be swapped
            vertexes_out[t,v][[0,1]] = vertexes_out[t,v][[1,0]]

    return vertexes_out

def cartesian_product(*arrays):
    return np.stack(np.meshgrid(*arrays), -1).reshape(-1, len(arrays))

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.

    """
    assert vec1.shape == (3,)
    assert vec2.shape == (3,)

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if np.linalg.norm(v) != 0:
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix
    
    else:
    # if the cross product is zero, then vecs must be parallel or perpendicular
        if np.linalg.norm(a + b) == 0:
            pointer = np.array([0,0,1])
            return rot_mat_from_pointer(pointer, 180)
            
        else:
            return np.eye(3)

def rot_mat_from_pointer(pointer, angle):
    '''
    Returns the rotation matrix that rotates a system around the given pointer
    of angle degrees. The algorithm is based on scipy quaternions.
    :params pointer: a 3D vector
    :params angle: a int/float, in degrees
    :return rotation_matrix: matrix that applied to a point, rotates it along the pointer
    '''
    assert pointer.shape[0] == 3

    pointer = norm(pointer)
    angle *= np.pi/180
    quat = np.array([np.sin(angle/2)*pointer[0],
                    np.sin(angle/2)*pointer[1],
                    np.sin(angle/2)*pointer[2],
                    np.cos(angle/2)])            # normalized quaternion, scalar last (i j k w)
    
    return R.from_quat(quat).as_matrix()
