#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math


# In[2]:


# from numpy import genfromtxt
coords = np.genfromtxt('data/coords.csv', delimiter=',',skip_header=1)
data_2d = coords[:6,:2]
data_3d = coords[6:,:]
print(data_2d)
print(data_3d)


# In[3]:


def get_avg_dist(points):
    d_avg = 0
    centroid = points.mean(0)
    for p in points:
        d_avg += np.linalg.norm(p - centroid)
    d_avg /= points.shape[0]
    return d_avg

def normalize_2d(points_2d):
    d_avg_2d = get_avg_dist(points_2d)
    centroid_2d = points_2d.mean(0)
    
    # Compute T matrix for normalizing 2D points
    T = np.array([[1,0,(-1)*centroid_2d[0]], [0,1,(-1)*centroid_2d[1]], [0,0,1]])
    T *= math.sqrt(2)/d_avg_2d
    T[2,2] = 1    
    print("T matrix for 2D normalization:\n" + str(T))
    
    # Find normalized 2D coordinates
    x = np.append(points_2d.T, np.array([[1,1,1,1,1,1]]), axis=0)
    x_norm = np.dot(T, x)
    points_2d_norm = x_norm[0:2].T
    print("Normalized 2D coordinates:\n" + str(points_2d_norm))
    return points_2d_norm, T

def normalize_3d(points_3d):
    d_avg_3d = get_avg_dist(points_3d)
    centroid_3d = points_3d.mean(0)
    
    # Compute U matrix for normalizing 3D points
    U = np.array([[1,0,0,(-1)*centroid_3d[0]], [0,1,0,(-1)*centroid_3d[1]], [0,0,1,(-1)*centroid_3d[2]], [0,0,0,1]])
    U *= math.sqrt(3)/d_avg_3d
    U[3,3] = 1  
    print("U matrix for 3D normalization:\n" + str(U))
    
    # Find normalized 3D coordinates
    X = np.append(points_3d.T, np.array([[1,1,1,1,1,1]]), axis=0)
    X_norm = np.dot(U, X)
    points_3d_norm = X_norm[0:3].T
    print("Normalized 3D coordinates:\n" + str(points_3d_norm))
    return points_3d_norm, U

def compute_P(points_3d, points_2d):
    # Form the P matrix
    points_3d_hom = np.append(points_3d, np.ones((6,1)), axis=1)
    P=np.empty((0,12))

    for i in range(points_3d.shape[0]):
        first = np.append(points_3d_hom[i], np.zeros(4))
        first = np.append(first, (-1)*points_2d[i][0]*points_3d_hom[i]) 
        first = first.reshape((1,12))
        P = np.append(P, first, axis=0)
        second = np.append(np.zeros(4), points_3d_hom[i])
        second = np.append(second, (-1)*points_2d[i][1]*points_3d_hom[i]) 
        second = second.reshape((1,12))
        P = np.append(P, second, axis=0)

    print("P matrix:\n" + str(P))  
    return P

def compute_m(P, T, U):
    # Find solution to Pm = 0 
    w, v = np.linalg.eig(np.dot(P.T,P))
    m = v[:,np.argmin(w)]
    M = m.real.reshape((3,4))    
    
    # Denormalize Projection Matrix m
    M = np.dot(M, U)
    M = np.dot(np.linalg.inv(T), M)
    print("Denormalized projection matrix m:\n" + str(M)) 
    return M

def recover_2d_points(points_2d,points_3d, M):
    # Recover 2D points using projection matrix found above
    points_3d_test = np.append(points_3d, np.ones((6,1)), axis=1)
    points_3d_test = points_3d_test.T
    points_2d_test = np.empty((0,2))

    for i in range(points_2d.shape[0]):
        x_i = np.dot(M[0], points_3d_test[:,i])/np.dot(M[2], points_3d_test[:,i])
        y_i = np.dot(M[1], points_3d_test[:,i])/np.dot(M[2], points_3d_test[:,i])
        points_2d_test = np.append(points_2d_test, np.array([[x_i,y_i]]), axis=0)

    print("Recovered 2D points:\n" + str(points_2d_test))
    return points_2d_test

def get_rmse(points_2d_test, points_2d):
    # Compute RMSE
    rmse = (points_2d_test - points_2d)**2
    rmse = rmse.mean(0)
    rmse = math.sqrt((rmse[0]+rmse[1])/2)
    print("Root mean squared Error: ", rmse)  
    return rmse

def get_intrinsic_parameters(M):
    # Compute Intrinsic Parameters
    A = M[:,0:3]
    b = M[:,3:4]
    epsilon = 1
    rho = epsilon/np.linalg.norm(A[2])
    x0 = rho*rho*np.dot(A[0],A[2])
    y0 = rho*rho*np.dot(A[1],A[2])
    print("Scaling Factor (rho): ", rho)
    print("x0: ", x0)
    print("y0: ", y0)

    a1_a3 = np.cross(A[0],A[2])
    a2_a3 = np.cross(A[1],A[2])
    cos_theta = (-1)*np.dot(a1_a3, a2_a3)/(np.linalg.norm(a1_a3)*np.linalg.norm(a2_a3))
    theta = math.acos(cos_theta)*180/math.pi
    sin_theta = math.sin(theta*math.pi/180)
    print("Theta: ", theta)
    print("Cos(theta): ", cos_theta)

    alpha = rho*rho*np.linalg.norm(a1_a3)*sin_theta
    beta = rho*rho*np.linalg.norm(a2_a3)*sin_theta
    print("Alpha: ", alpha)
    print("Beta: ", beta)

    K = np.array([[alpha, (-1)*alpha*cos_theta/sin_theta, x0], [0, beta/sin_theta, y0], [0, 0, 1]])
    print("K matrix:\n" + str(K))    
    return alpha, beta, x0, y0, theta, K

def get_extrinsic_parameters(M, K):
    # Compute Extrinsic parameters
    A = M[:,0:3]
    b = M[:,3:4]
    epsilon = 1
    rho = epsilon/np.linalg.norm(A[2])  
    a1_a3 = np.cross(A[0],A[2])
    a2_a3 = np.cross(A[1],A[2])    
    
    r3 = rho*A[2]
    r1 = a2_a3/np.linalg.norm(a2_a3)
    r2 = np.cross(r3, r1)
    R = np.empty((0,3))
    R = np.append(R, r1.reshape((1,3)), axis=0)
    R = np.append(R, r2.reshape((1,3)), axis=0)
    R = np.append(R, r3.reshape((1,3)), axis=0)
    print("R matrix:\n" + str(R))

    t = rho*np.dot(np.linalg.inv(K), b)
    print("t matrix:\n" + str(t))    
    
    return R, t


# In[4]:


points_2d_norm, T = normalize_2d(data_2d)
points_3d_norm, U = normalize_3d(data_3d)


# In[5]:


P = compute_P(points_3d_norm, points_2d_norm)


# In[6]:


M = compute_m(P, T, U)


# In[7]:


points_2d_test = recover_2d_points(data_2d,data_3d, M)
rmse = get_rmse(points_2d_test, data_2d)


# In[8]:


np.savetxt("data/recovered_coords.csv", points_2d_test, delimiter=",")


# In[9]:


alpha, beta, x0, y0, theta, K = get_intrinsic_parameters(M)


# In[10]:


R, t = get_extrinsic_parameters(M, K)

