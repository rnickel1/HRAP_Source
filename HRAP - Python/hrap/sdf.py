# Author: Thomas A. Scott, https://www.scott-aero.com/

import numpy as np

def clean_points(p: np.ndarray):
    """
    Takes in a disordered array of points and orders them counterclockwise, removing duplicates.

    Args:
        p (np.ndarray): An array of shape (N, 2) representing the points.
    Returns:
        np.ndarray: The cleaned array of shape (M, 2).
    """
    
    cx, cy = np.mean(p[:,0]), np.mean(p[:,1])
    px, py = p[:,0] - cx, p[:,1] - cy
    # Sort ascending from angle then by distance from centroid
    I = np.lexsort(np.stack([px**2 + py**2, np.arctan2(py, px)], axis=0))
    sp = p[I,:]
    # Edge case: only 1 point
    if np.array_equal(sp[0,:], sp[-1,:]):
        return sp[0,:]
    # Return unique and sorted (unique if not equal to its right element, last always included)
    sp1 = np.roll(sp,-1,axis=0)
    usp = sp[~((sp[:,0] == sp1[:,0]) & (sp[:,1] == sp1[:,1])),:]

    return usp

def sd_poly(v: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Calculates the signed distance to a 2D polygon. The polygon may be self-intersecting.
    Ported from Inigo Quilez's GLSL https://iquilezles.org/articles/distfunctions2d/.

    Args:
        v (np.ndarray): An array of shape (N, 2) representing the counterclockwise polygon vertices, cannot have consecutive duplicates.
        p (np.ndarray): An array of shape (..., 2) representing the test points.
    Returns:
        np.ndarray: The signed distance of shape (...) from the test points to the polygon.
    """
    # Get distance from each point to each line segment via closest point along the segment
    v1 = np.roll(v,1,axis=0) # previous vertices
    e = v1 - v
    w = p[...,np.newaxis,:] - v
    b = w - e * np.clip((w[...,0]*e[...,0] + w[...,1]*e[...,1]) / (e[...,0]**2 + e[...,1]**2), 0.0, 1.0)[...,np.newaxis]
    # Get shortest distance for each point from every line segment
    d = np.sqrt(np.min((b[...,0]**2 + b[...,1]**2), axis=-1))

    # Use winding number trick for sign
    c = np.stack([p[...,np.newaxis,1] >= v[:,1], p[...,np.newaxis,1] < v1[:,1], e[:,0] * w[...,1] > e[:,1] * w[...,0]], axis=-1)
    s = np.ones(list(p.shape)[:-1] + [v.shape[0]])
    s[np.all(c, axis=-1) | ~np.any(c, axis=-1)] = -1.0
    s = np.prod(s, axis=-1)
    
    return s * d

def area_poly(v: np.ndarray) -> float:
    """
    Calculates the area of a 2D polygon using shoelace formula. The polygon may be self-intersecting.

    Args:
        v (np.ndarray): An array of shape (N, 2) representing the counterclockwise polygon vertices.
    Returns:
        float: The polygon.
    """
    v1 = np.roll(v,-1,axis=0) # next vertices
    A = np.sum((v[:,1] + v1[:,1]) * (v[:,0] - v1[:,0])) / 2 # 1/2 sum (yi + yi+1) (xi - xi+1)

    return A

if __name__ == "__main__":
    print('BEGIN UNIT SQUARE TEST')
    v = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    # Inside, on boundary, outside
    p = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 2.0]])
    sd = sd_poly(v, p)
    area1 = area_poly(v)
    print('Test points:\n', p)
    print('Signed distances:\n', sd)
    print('Is correct:', np.isclose(sd, [-1.0, 0.0, np.sqrt(2)]))
    print('Poly area', area1, 'is correct:', np.isclose(area1, 4.0))

    print()

    print('BEGIN DIRTY UNIT SQUARE TEST')
    # Initially wrong order (hourglass shape) and includes a duplicate
    v = clean_points(np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]))
    # Inside, on boundary, outside
    p = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 2.0]])
    sd = sd_poly(v, p)
    area1 = area_poly(v)
    print('Clean shape:\n', v)
    print('Test points:\n', p)
    print('Signed distances:\n', sd)
    print('Is correct:', np.isclose(sd, [-1.0, 0.0, np.sqrt(2)]))
    print('Poly area', area1, 'is correct:', np.isclose(area1, 4.0))

    print()

    print('BEGIN UNIT CIRCLE TEST')
    t = np.linspace(0.0, 2*np.pi, 1000, endpoint = False)
    v = np.stack([np.cos(t), np.sin(t)], axis=-1)
    # Inside, on boundary, outside
    p = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 2.0]])
    sd = sd_poly(v, p)
    area1 = area_poly(v)
    print('Test points:\n', p)
    print('Signed distances:\n', sd)
    print('Is correct:', np.isclose(sd, [-1.0, 0.0, np.sqrt(8)-1]))
    print('Poly area', area1, 'is correct:', np.isclose(area1, np.pi))
