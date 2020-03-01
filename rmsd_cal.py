import numpy as np
import copy
import gzip
import re
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import argparse
import sys
"""
The root-mean-square deviation (RMSD) is calculated, using straight-forward RMSD and Kabsch algorithm (1976)
"""
def get_coordinates_pdb(filename):
    x_column = None
    V = list()

    # Same with atoms and atom naming.
    # The most robust way to do this is probably
    # to assume that the atomtype is given in column 3.
    atoms = list()
    if filename[-2:] =="gz":
        openfunc = gzip.open
        openarg = 'rt'
    else:
        openfunc = open
        openarg = 'r'
    with openfunc(filename, openarg) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("TER") or line.startswith("END"):continue
            #if line.startswith("ATOM"):
            if line.startswith("HETATM"):
                tokens = line.split()
                if x_column is None:
                    try:
                        # look for x column
                        for i, x in enumerate(tokens):
                            if "." in x and "." in tokens[i + 1] and "." in tokens[i + 2]:
                                x_column = i
                                #print(f'x_column = {i}')
                                break
                    except IndexError:
                        exit("error: Parsing coordinates for the following line: \n{0:s}".format(line))
                # Try to get the atomtype
                try:
                    atom_type = tokens[2][0]
                    #if atom in ("H", "C", "N", "O", "S", "P"):
                    if atom_type !="H":#as only consider heavy atoms                        
                        atoms.append(tokens[2])
                        try:                # Try to read the coordinates
                            V.append(np.asarray(tokens[x_column:x_column + 3], dtype=float))
                        except:
                            # If that doesn't work, use hardcoded indices
                            try:
                                x = line[30:38]
                                y = line[38:46]
                                z = line[46:54]
                                V.append(np.asarray([x, y ,z], dtype=float))
                            except:
                                exit("error: Parsing input for the following line: \n{0:s}".format(line))
                    #else:
                    #    # e.g. 1HD1
                    #    atom = tokens[2][1]
                    #    if atom == "H":
                    #        atoms.append(atom)
                    #    else:
                    #        raise Exception
                except:
                    exit("error: Parsing atomtype for the following line: \n{0:s}".format(line))
    V = np.asarray(V)
    atoms = np.asarray(atoms)
    #print(V.shape[0],atoms)
    assert V.shape[0] == atoms.size
    return atoms, V
def rmsd(V, W):
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.
    Parameters
    ----------
    V : array
        (N,D) matrix, where N is points and D is dimension.
    W : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    rmsd : float
        Root-mean-square deviation between the two vectors
    """
    diff = np.array(V) - np.array(W)
    N = len(V)
    return np.sqrt((diff * diff).sum() / N)

def kabsch_rmsd(P, Q, W=None, translate=False):
    """
    Rotate matrix P unto Q using Kabsch algorithm and calculate the RMSD.
    An optional vector of weights W may be provided.
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    W : array or None
        (N) vector, where N is points.
    translate : bool
        Use centroids to translate vector P and Q unto each other.
    Returns
    -------
    rmsd : float
        root-mean squared deviation
    """

    if translate:
        Q = Q - centroid(Q)
        P = P - centroid(P)

    if W is not None:
        return kabsch_weighted_rmsd(P, Q, W)

    P = kabsch_rotate(P, Q)
    return rmsd(P, Q)

def centroid(X):
    """
    Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.
    https://en.wikipedia.org/wiki/Centroid
    C = sum(X)/len(X)
    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    C : float
        centroid
    """
    C = X.mean(axis=0)
    return C

def kabsch_rotate(P, Q):
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm.
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    P : array
        (N,D) matrix, where N is points and D is dimension,
        rotated
    """
    U = kabsch(P, Q)

    # Rotate P
    P = np.dot(P, U)
    return P

def kabsch(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U



def kabsch_fit(P, Q, W=None):
    """
    Rotate and translate matrix P unto matrix Q using Kabsch algorithm.
    An optional vector of weights W may be provided.
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    W : array or None
        (N) vector, where N is points.
    Returns
    -------
    P : array
        (N,D) matrix, where N is points and D is dimension,
        rotated and translated.
    """
    if W is not None:
        P = kabsch_weighted_fit(P, Q, W, rmsd=False)
    else:
        QC = centroid(Q)
        Q = Q - QC
        P = P - centroid(P)
        P = kabsch_rotate(P, Q) + QC
    return P


def kabsch_weighted_fit(P, Q, W=None, rmsd=False):
    """
    Fit P to Q with optional weights W.
    Also returns the RMSD of the fit if rmsd=True.
    Parameters
    ----------
    P    : array
           (N,D) matrix, where N is points and D is dimension.
    Q    : array
           (N,D) matrix, where N is points and D is dimension.
    W    : vector
           (N) vector, where N is points
    rmsd : Bool
           If True, rmsd is returned as well as the fitted coordinates.
    Returns
    -------
    P'   : array
           (N,D) matrix, where N is points and D is dimension.
    RMSD : float
           if the function is called with rmsd=True
    """
    R, T, RMSD = kabsch_weighted(Q, P, W)
    PNEW = np.dot(P, R.T) + T
    if rmsd:
        return PNEW, RMSD
    else:
        return PNEW


def rmsd_cal(li1="TTTTTT.pdb",li2="TTTT.pdb"):
    #p_all_atoms, p_all = get_coordinates(args.structure_a, args.format)
    p_all_atoms, p_all = get_coordinates_pdb(li1)
    q_all_atoms, q_all = get_coordinates_pdb(li2)
    p_coord = copy.deepcopy(p_all)
    q_coord = copy.deepcopy(q_all)
    p_atoms = copy.deepcopy(p_all_atoms)
    q_atoms = copy.deepcopy(q_all_atoms)
    p_size = p_all.shape[0]
    q_size = q_all.shape[0]
    p_cent = centroid(p_coord)
    q_cent = centroid(q_coord)
    p_coord -= p_cent
    q_coord -= q_cent
    # Get rotation matrix
    #U = kabsch(q_coord, p_coord)
    #print(p_all,q_all,p_size,q_size)
    krmsd=kabsch_rmsd(p_coord, q_coord)#kabsch_rmsd also contain the ksbasch 将P 旋转到Q上
    straight_rmsd=rmsd(p_all,q_all)
    #use kabsch_fit to replace the kabsch_rotate and check diff
    #kfit_p=kabsch_fit(p_coord, q_coord)
    #kfit=rmsd(kfit_p, q_coord)#as get same to krmsd
    print(f'{li1}---{li2}')
    print(f'kabsch_rmsd={krmsd}\n straight_rmsd={straight_rmsd}')
    id=li1.split('.')[-2]
    with open("rmsd.log.csv","a+")as wf:
        wf.write(f'{id},{krmsd},{straight_rmsd},{li1}---{li2}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pos1',  type=str,  default='T2.pdb',help='structures in pdb format')
    parser.add_argument('--pos2', type=str,  default='TTTT.pdb',help='structures in.pdb format')
    args = parser.parse_args()
    print(args)
    rmsd_cal(args.pos1,args.pos2)

