import torch
import numpy as np
import scipy.sparse as sp

class ClassicIBM:
    def __init__(self,h,delta_s,p_e,p_l):
        self.h = h
        self.delta_s = delta_s
        self.p_e = p_e
        self.p_l = p_l

        xMax_l = np.max(p_e[:,0])
        xMin_l = np.min(p_e[:,0])
        yMax_l = np.max(p_e[:, 1])
        yMin_l = np.min(p_e[:, 1])
        self.IsInPoly = np.ones(self.p_e.shape[0], dtype='float32')
        for i in range(self.p_e.shape[0]):
            if self.p_e[i,0]>=xMin_l and self.p_e[i,0]<=xMax_l and self.p_e[i,1]>=yMin_l and self.p_e[i,1]<=yMax_l:
                if self.isInPoly(self.p_e[i,:]):
                    self.IsInPoly[i]=0
        self.p_l = self.p_l[:-1,:]
        self.GetMatrix()
        print('IBM initialized!')

    def isRayIntersectsSegment(self, poi, s_point, e_point):
        if s_point[1] == e_point[1]:
            return False
        if s_point[1] > poi[1] and e_point[1] > poi[1]:
            return False
        if s_point[0] < poi[0] and e_point[1] < poi[1]:
            return False
        if s_point[1] < poi[1] and e_point[1] < poi[1]:
            return False
        if s_point[1] == poi[1] and e_point[1] > poi[1]:
            return False
        if e_point[1] == poi[1] and s_point[1] > poi[1]:
            return False

        xseg = e_point[0] - (e_point[0] - s_point[0]) * (e_point[1] - poi[1]) / (e_point[1] - s_point[1])
        if xseg < poi[0]:
            return False
        else:
            return True

    def isInPoly(self, poi):
        intersec = 0
        for i in range(self.p_l.shape[0]-1):
            s_point = self.p_l[i,:]
            e_point = self.p_l[i+1,:]
            if self.isRayIntersectsSegment(poi, s_point, e_point):
                intersec += 1
        if intersec % 2 == 1:
            return True
        else:
            return False

    def GetMatrix(self):
        E2L = sp.coo_matrix((self.p_l.shape[0], self.p_e.shape[0]), ['float32']).tolil()
        L2E = sp.coo_matrix((self.p_e.shape[0], self.p_l.shape[0]), ['float32']).tolil()
        for i in range(self.p_l.shape[0]):
            for j in range(self.p_e.shape[0]):
                if self.IsInPoly[j] == 0:
                    continue
                rx=abs(self.p_l[i,0]-self.p_e[j,0])/self.h
                if rx>=2:
                    continue
                ry=abs(self.p_l[i,1]-self.p_e[j,1])/self.h
                if rx<2 and ry<2:
                    if rx<1:
                        dx = (3-2*rx+np.sqrt(1+4*rx-4*rx*rx))/8
                    elif rx<2:
                        dx = (5-2*rx-np.sqrt(-7+12*rx-4*rx*rx))/8
                    if ry < 1:
                        dy = (3 - 2 * ry + np.sqrt(1 + 4 * ry - 4 * ry * ry)) / 8
                    elif ry < 2:
                        dy = (5 - 2 * ry - np.sqrt(-7 + 12 * ry - 4 * ry * ry)) / 8
                    E2L[i,j] = dx*dy
                    L2E[j, i] = dx*dy/self.h*self.delta_s
        self.E2L = E2L.tocsc().astype('float32')
        self.L2E = L2E.tocsc().astype('float32')

    def step(self,u,v,NGx,NGy):
        u1 = u.cpu().reshape(-1).numpy()
        v1 = v.cpu().reshape(-1).numpy()
        U = self.E2L.dot(u1)
        V = self.E2L.dot(v1)
        fx = self.L2E.dot(-U)
        fy = self.L2E.dot(-V)
        delta_u = fx
        delta_v = fy
        u1 = ((u1+delta_u)*self.IsInPoly).reshape([NGy+1,NGx+1])
        v1 = ((v1+delta_v)*self.IsInPoly).reshape([NGy+1,NGx+1])
        u1 = torch.from_numpy(u1).cuda()
        v1 = torch.from_numpy(v1).cuda()
        return u1, v1

