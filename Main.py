# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:15:29 2018
Bounce~Bounce~
@author: lijiang
"""

import cv2
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class Track:
    k,b=0,0
    x,y=0,0
    A,B,C=0,0,0
    def __init__(self,x,y):
        self.x,self.y=x,y
        theta=random.random()*180-90
        self.k=math.tan(theta/360*2*math.pi)
#        print("theta:",theta)
#        print("k:",self.k)
        self.b=y-self.k*x
        self.A=self.k
        self.B=-1
        self.C=self.b
    def dist(self,x1,y1):
        return abs((self.A*x1+self.B*y1+self.C)/(self.A**2+self.B**2)**0.5)
    def get_y(self,x):
        return self.k*x+self.b
    def draw_line(self,img,color,thick):
        for i in range(len(img)):
            for j in range(len(img)):
                if track.dist(j,len(img)-i)<thick:
                    img[i,j]=color
class Line:
    A,B,C=0,0,0
    x1,x2,y1,y2=0,0,0,0
    def __init__(self,point1,point2):
        (self.x1,self.y1)=point1
        (self.x2,self.y2)=point2
    
    def get_y(self,x):
        return (self.y1-self.y2)*(x-self.x2)/(self.x1-self.x2)+self.y2

def point_dist(point1,point2):
#    (x1,y1,z1)=point1
#    (x2,y2,z2)=point2
#    return ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5
    return math.sqrt(np.power(point1 - point2, 2).sum())

def init_point(img):
#    for i in range(len(img)):
#        for j in range(len(img[0])):
#            if img[i,j].all()>0:
#                return j,i
    return 150,180


##统计时间窗中心点

def window_center(scatters,window_length=8,window_freqency=4):#8
    centers=[]
#    for i in range(0,len(scatters),window_length):##每隔一定时间才产生时间窗
#        window_scatters=scatters[i:i+window_length]
#        center=[0,0,(i+(i+window_length))/2]
#        for j in window_scatters:
#            center[0]+=j[0]
#            center[1]+=j[1]
#        center[0]/=window_length
#        center[1]/=window_length
#        centers.append(center)
    for i in range(0,len(scatters)-window_length,window_freqency):##每个时间都产生时间窗
        window_scatters=scatters[i:i+window_length]
        center=[0,0,(i+(i+window_length))/2]
        for j in window_scatters:
            center[0]+=j[0]
            center[1]+=j[1]
        center[0]/=window_length
        center[1]/=window_length
#        print("第%s个窗的中心点为%s"%(len(centers),center))
        centers.append(center)
        
#    n_centers=centers.copy()
#    del_count=0
#    for i in range(1,len(centers)):
#        isolated=1
#        for j in range(i):
#            if point_dist(np.array(centers[i]),np.array(centers[j]))<30:
#                isolated=0
#                break
#        if isolated==1:
#            print("del",n_centers[i-del_count])
#            del(n_centers[i-del_count])
#            del_count+=1
#    centers=n_centers
        
    return centers
def cluster_scatters(centroids):
    cluster=scatters.copy()
    color=['r', 'y', 'g', 'b', 'c', 'k', 'm']
    
    for i in range(len(scatters)):
        min_dist=float("inf")
        for j in range(len(centroids)):
            if point_dist([scatters[i][0],scatters[i][1]],centroids[j])<min_dist:
                cluster[i]=j
                min_dist=point_dist([scatters[i][0],scatters[i][1]],centroids[j])
#    for i in range(len(centroids)):
#        plt.scatter(centroids[i][0],centroids[i][1],color="b",marker='*')
    plt.figure()
    
    for i in range(len(cluster)):
        plt.scatter(scatters[i][0],len(img)-scatters[i][1]-1,color=color[cluster[i]%len(color)],label=cluster[i])
    plt.scatter([centroids[i][0] for i in range(len(centroids))],[len(img)-1-centroids[i][1] for i in range(len(centroids))],color="b",marker='*')
    plt.show()
    plt.ylabel("y")
    plt.xlabel("x")
#        plt.axis([-1,11,0,7])
    return cluster 
    

def shrink_centers(centers,method="meanshift",clusters=2):
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    if method=="kmeans":
        from sklearn.cluster import KMeans
        estimator=KMeans(n_clusters=clusters)
        res=estimator.fit_predict([(centers[i][0],centers[i][1]) for i in range(len(centers))])
        # 预测类别标签结果
        lable_pred=estimator.labels_
        # 各个类别的聚类中心值
        centroids=estimator.cluster_centers_
        # 聚类中心均值向量的总和
        inertia=estimator.inertia_
        ax.clear()
        ax.scatter([centroids[i][0] for i in range(len(centroids))],
                    [centroids[i][1] for i in range(len(centroids))],
                    [i for i in range(len(centroids))],
                    c='b',marker='*',s=100)
    elif method=="meanshift":
        from sklearn.cluster import MeanShift, estimate_bandwidth
        from sklearn.datasets.samples_generator import make_blobs
#        centers = [[1, 1,1], [-1, -1,2], [1, -1,0]]
        X, _ = make_blobs(n_samples=1000, centers=[(centers[i][0],centers[i][1]) for i in range(len(centers))],
                                                   cluster_std=0.5)##调整簇内距离，越大簇越少#0.8
        
        # #############################################################################
        # Compute clustering with MeanShift
        
        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)#0.2
        
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
        centroids = ms.cluster_centers_
        
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        
        print("number of estimated clusters : %d" % n_clusters_)
        ax.clear()
        ax.scatter([centroids[i][0] for i in range(len(centroids))],
                    [centroids[i][1] for i in range(len(centroids))],
                     [0]*len(centroids), c='b',marker='*',s=100)
    
    ax.scatter([scatters[i][0] for i in range(len(scatters))],
                [scatters[i][1] for i in range(len(scatters))],
                [scatters[i][2] for i in range(len(scatters))],
                c='r')
    ax.scatter([centers[i][0] for i in range(len(centers))],
                [centers[i][1] for i in range(len(centers))],
                [centers[i][2] for i in range(len(centers))],
                c='g')
    
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()
    return centroids

def next_move(x,y,track,direct):
    nx=x+direct
    ny=int(track.get_y(nx))
    
    if nx<0 or nx>=len(img[0]) or ny<0 or ny>=len(img):##如果超出界限
        return cx,cy,-1
    return nx,ny,0

def bounce(i,j,times=500):##迭代次数
    def rotate(track,th):
        kernel=np.mat([[math.cos(th),math.sin(th),0],
                        [-math.sin(th),math.cos(th),0],
                        [0,0,1]])
        for i in range(len(track)):
            result=kernel*np.mat([track[i][0],track[i][1],1]).T
#            print([int(result[i]) for i in range(len(result)-1)])
            track[i]=[int(result[i]) for i in range(len(result)-1)]
        return track
    def generate_track(length,long,fat):
        a,b=0,0
        track=[]
        for i in range(length):
            a=(int((i+1)/2)*2)*(-1)**(i+1)*fat
            b=(i*2-2)*long
            track.append((a,b))
        return track
    global l
    upbar=[(0,j) for j in range(len(img[0]))]
    leftbar=[(i,0) for i in range(len(img))]
    downbar=[(len(img)-1,j) for j in range(len(img[0]))]
    rightbar=[(i,len(img[0])) for i in range(len(img))]
    barrier=upbar+leftbar+downbar+rightbar
    
    scatters=[]
    t=0
    track=generate_track(50,long=8,fat=1)#4,4
    while(len(scatters)<times):
        if int(t%(times/10))==0:
            print(t/times*100,"%")##进度显示
        angle=int(random.random()*2*math.pi)
        random_track=rotate(track,angle)
        donestep=[(i,j)]
        retrack=False
        for ii in range(1,len(random_track)):
            if retrack==True:
                break
            nextstep=[int(random_track[ii][0]+i),int(random_track[ii][1]+j)]
#            print("now",(i,j))
#            print("nextstep",nextstep)
            
            l=Line((donestep[-1][0],donestep[-1][1]),(nextstep[0],nextstep[1]))
            
            try:
                
                step=-int((donestep[-1][0]-nextstep[0])/abs((donestep[-1][0]-nextstep[0])))
                
                for jj in range(donestep[-1][0],int(nextstep[0]),step):
                    
            #        print((jj,int(l.get_y(jj))),state[int(l.get_y(jj)),jj])
                    if state[int(l.get_y(jj)),jj]>0 and state[int(l.get_y(jj+step)),jj+step]==0:
                        t+=1
#                        print("t+1=",t)
                        cv2.circle(img,(jj,int(l.get_y(jj))),3,[0,0,255])
                        scatters.append((jj,int(l.get_y(jj)),t))
                        (i,j)=(jj,int(l.get_y(jj)))
                        retrack=True
            except: 
                continue
            cv2.line(img,(donestep[-1][0],donestep[-1][1]),(nextstep[0],nextstep[1]),[0,255,255])
            donestep.append((nextstep[0],nextstep[1]))
            cv2.imshow("",img)
            cv2.waitKey(1)
            
    return scatters

if __name__=="__main__":
    img=cv2.imread(".\img7.png")
#    img=cv2.imread("./dataset/20051020_55701_0100_PP.jpg")
    
    kernel=np.uint8(np.zeros((5,5)))
    for x in range(5):
    	kernel[x,2]=1;
    	kernel[2,x]=1;
    img=cv2.dilate(img,kernel)
    img=cv2.dilate(img,kernel)
    img=cv2.erode(img,kernel)
    img=cv2.erode(img,kernel)

#    img=cv2.imread("./img3.png")
    state=img.copy()[:,:,0]
    
    '''
    开始弹跳
    '''
    ci,cj=init_point(img)
    print((ci,cj))
    cv2.circle(img,(ci,cj),3,[0,255,0])
    
    
    l=object
    scatters=bounce(ci,cj)##调整弹珠方向与跨度
    
#    plt.show()
    
    '''
    统计时间窗中心点
    '''
    ##window_length越低越适合接触多的图形
    centers=window_center(scatters,window_length=8,window_freqency=4)##调整时间窗长度和出现频率##6,4
#    centroids=shrink_centers(centers,method="kmeans",clusters=2)##调整聚类方式和簇数
    centroids=shrink_centers(centers,method="meanshift")##调整聚类方式和簇数
    cluster_scatters(centroids)
    
#    centroids=shrink_centers(scatters,method="kmeans",clusters=2)##调整聚类方式和簇数
#    cluster_scatters(centroids)