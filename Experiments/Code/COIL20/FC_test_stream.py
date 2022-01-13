# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 22:11:01 2019

@author: xyan
"""

import numpy as np
import math
import pandas as pd
from math import exp, ceil
import numpy.matlib as b
from sklearn.preprocessing import normalize
import time
# from entropy_estimators import *
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score, precision_score, recall_score

# Change the filename here:
global filename, Batchsize
filename = 'COIL20.csv'
# Change the chunk size here:
Batchsize = 256

def Input():
    # Read the data from the txt file
    sample = pd.read_csv(filename,header=None)
    (N, L) = np.shape(sample)
    dim = L - 1
    label1 = sample.iloc[:,L-1]
    label = label1.values
    data = sample.iloc[:,0:dim] 
    NewData = Pre_Data(data)
    return NewData,label
def Pre_Data(data):
    [N,L] = np.shape(data)
    NewData = np.zeros((N,L))
    for i in range(L):
        Temp = data.iloc[:,i]
        if max(Temp)==0:
            NewData[:,i] = 0
        else:
            NewData[:,i] = Temp
    return NewData
                
def Distribution_Est(histfeatures, data, dim):
    L =  dim
    DC_mean = np.zeros(L)
    DC_std = np.zeros(L)
    for i in range(dim):
        TempClass = data[:, i]
        DC_mean[i] = np.mean(TempClass)
        DC_std[i] = np.std(TempClass)
    return DC_mean, DC_std

def Feature_Dist(histfeatures, data, dim):
    Dist = []
    L = dim
    Var = np.var(data, axis=0)
    Corr = np.corrcoef(data.T)
    DisC = np.zeros((L,L))
    # histf = histfeatures[:,0]
    for i in range(L):
        for j in range(i, L):
            DisC[i,j] = KLD_Cal(data,i,j,Var,Corr)
            # DisC[i, j] = Sym_Cal(data, i, j)
            DisC[j, i] = DisC[i, j]
            Dist.append(DisC[i, j])
    return DisC,Dist

def KLD_Cal(data,i,j,Var,Corr):
    Var1 = Var[i]
    Var2 = Var[j]
    
    P = Corr[i,j]
    Sim = Var1 + Var2 - ((Var1 + Var2)**2 - 4 * Var1 * Var2 * (1 - P**2))**0.5
    D_KL = Sim / (Var1 + Var2)
    return D_KL 

def Sym_Cal(data,i,j):
    if len(np.unique(data[:,i]))==0 or len(np.unique(data[:,i]))==0:
        D_KL = 1
    else:
        I_ij = midd(data[:,i],data[:,j])
        H_I = entropyd(data[:,i])
        H_J = entropyd(data[:,j])
        if H_I == 0 or H_J == 0:
            D_KL = 0
        else:
            D_KL = 1 - 2*(I_ij)/(H_I + H_J)
    return D_KL

def fitness_cal(histsummary, DisC, DC_means, DC_std, data, StdF, gamma, histStdF):
    fitness = np.zeros(len(DC_means))
    for i in range(len(DC_means)):
        TempSum = 0
        for j in range(len(DC_means)):
            if j != i:
                D = DisC[i,j]
                TempSum = TempSum + (exp(- (D**2) / StdF))**gamma
        fitness[i] = TempSum
    if len(histStdF) == 0:
        updatedfit = fitness
    else:
        updatedfit = fitness_update(DisC, histsummary, histStdF, fitness, StdF, gamma)
    return updatedfit


def fitness_update(DisC, histsummary, histStdF, rawfitness, StdF, gamma):
    num_hist = np.shape(histsummary)[0]
    num_fit = len(rawfitness)
    histdist = DisC[:, :num_hist]
    histfitness = histsummary[:,1]
    fitness_new = rawfitness
    for i in range(num_fit):
        curr_dist = histdist[i, :]
        n_idx = np.argmin(curr_dist)
        n_dist = np.min(curr_dist)
        if i < num_hist:
            fitness_new[i] = histsummary[n_idx,1]
        else:
            fitness_new[i] += (exp(- (n_dist**2) / StdF))**gamma * (histfitness[n_idx] ** (histStdF[-1]/StdF))
    return fitness_new

def Pseduo_Peaks(DisC, Dist, DC_Mean, DC_Std, data, fitness, StdF, gamma):
    
    # The temporal sample space in terms of mean and standard deviation
    sample = np.vstack((DC_Mean,DC_Std)).T
    # Spread= np.max(Dist)
    # Search Stage of Pseduo Clusters at the temporal sample space
#    NeiRad = 0.25 * StdF
    NeiRad = 0.01*np.mean(Dist)
    # NeiRad = (StdF/gamma)
    i = 0
    marked = []
    C_Indices = np.arange(1, len(DC_Mean)+1) # The pseduo Cluster label of features
    PeakIndices = []
    Pfitness = []
    co = []
    F = fitness
    while True:
        PeakIndices.append(np.argmax(F))
        Pfitness.append(np.max(F))
    
        indices = NeighborSearch(DisC, data, sample, PeakIndices[i], marked, NeiRad)
        
        C_Indices[indices] = PeakIndices[i]
        if len(indices) == 0:
            indices=[PeakIndices[i]]
        
        co.append(len(indices)) # Number of samples belong to the current 
    # identified pseduo cluster
        marked = np.concatenate(([marked,indices]))
  
        # Fitness Proportionate Sharing
        F = Sharing(F, indices) 
        
        # Check whether all of samples has been assigned a pseduo cluster label
        if np.sum(co) >= (len(F)):
            PeakIndices = np.unique(PeakIndices)
            C_Indices = Close_FCluster(PeakIndices, DisC, np.shape(DisC)[0])
            break
        
        i=i+1 # Expand the size of the pseduo cluster set by 1
    return PeakIndices,Pfitness,C_Indices

def NeighborSearch(DisC, data, sample, P_indice, marked, radius):
    Cluster = []
    for i in range(np.shape(sample)[0]):
        if i not in marked:
            Dist = DisC[i, P_indice]
            if Dist <= radius:
                Cluster.append(i)
    Indices = Cluster
    return Indices
#---------------------------------------------------------------------------------------------------
def Sharing(fitness, indices):
    newfitness = fitness
    sum1 = 0
    for j in range(len(indices)):
        sum1 = sum1 + fitness[indices[j]]
    for th in range(len(indices)):
            newfitness[indices[th]] = fitness[indices[th]] / (1+sum1)
            
    return newfitness
#-----------------------------------------------------------------------------------------------------    
def Pseduo_Evolve(DisC, PeakIndices, PseDuoF, C_Indices, DC_Mean, DC_Std, data, fitness, StdF, gamma):
    
    # Initialize the indices of Historical Pseduo Clusters and their fitness values
    HistCluster = PeakIndices
    HistClusterF = PseDuoF
    while True:
        # Call the merge function in each iteration
        [Cluster,Cfitness,F_Indices] = Pseduo_Merge(DisC, HistCluster, HistClusterF, C_Indices, DC_Mean, DC_Std, data, fitness, StdF, gamma)
        # Check for the stablization of clutser evolution and exit the loop
        if len(np.unique(Cluster)) == len(np.unique(HistCluster)) or len(Cluster) == 1:
            break
    
        # Update the feature indices of historical pseduo feature clusters and
        # their corresponding fitness values
    
        HistCluster=Cluster
        HistClusterF=Cfitness
        C_Indices = F_Indices
    # Compute final evolved feature cluster information 
    FCluster = np.unique(Cluster)
    Ffitness = Cfitness
    C_Indices = Close_FCluster(FCluster, DisC, np.shape(DisC)[0])
    
    return FCluster, Ffitness, C_Indices
#----------------------------------------------------------------------------------------------------------
def Pseduo_Merge(DisC, PeakIndices, PseDuoF, C_Indices, DC_Mean, DC_Std, data, fitness, StdF, gamma):
    # Initialize the pseduo feature clusters lables for all features 
    F_Indices = C_Indices
    # Initialize the temporal sample space for feature means and stds
    sample = np.vstack((DC_Mean,DC_Std)).T
    ML = [] # Initialize the merge list as empty
    marked = [] #List of checked Pseduo Clusters Indices 
    Unmarked = [] # List of unmerged Pseduo Clusters Indices 
    for i in range(len(PeakIndices)):
            M = 1 # Set the merge flag as default zero
            MinDist = math.inf # Set the default Minimum distance between two feature clusters as infinite
            MinIndice = 0 # Set the default Neighboring feature cluster indices as zero
            # Check the current Pseduo Feature Cluster has been evaluated or not
            if PeakIndices[i] not in marked:
                for j in range(len(PeakIndices)):
                        if j != i:
                            # Divergence Calculation between two pseduo feature clusters
                            D = DisC[PeakIndices[i], PeakIndices[j]]
                            if MinDist > D:
                                MinDist = D
                                MinIndice = j
                if MinIndice != 0:
                    # Current feature pseduo cluster under check
                    Current = sample[PeakIndices[i],:]
                    CurrentFit = PseDuoF[i]
                    # Neighboring feature pseduo cluster of the current checked cluster
                    Neighbor = sample[PeakIndices[MinIndice],:]
                    NeighborFit = PseDuoF[MinIndice]
                    
                    # A function to identify the bounady feature instance between two 
                    # neighboring pseduo feature clusters
                    BP=Boundary_Points(DisC, F_Indices,data, PeakIndices[i], PeakIndices[MinIndice])
                    BPF=fitness[BP]
                    if BPF < 1*min(CurrentFit,NeighborFit): # 0.95
                        M = 0 # Change the Merge flag
                    else:
                        M = 1
                    
                    if M == 1:
                        ML.append([PeakIndices[i],PeakIndices[MinIndice]])
                        marked.append(PeakIndices[i])
                        marked.append(PeakIndices[MinIndice])
                    else:
                        Unmarked.append(PeakIndices[i])
    NewPI = []
    # Update the pseduo feature clusters list with the obtained mergelist 
    for m in range(np.shape(ML)[0]):
        # print(ML[m][0],ML[m][1])
        if fitness[ML[m][0]] > fitness[ML[m][1]]:
            NewPI.append(ML[m][0])
            F_Indices[C_Indices==ML[m][1]] = ML[m][0]
        else:
            NewPI.append(ML[m][1])
            F_Indices[C_Indices==ML[m][0]] = ML[m][1]
    # Update the pseduo feature clusters list with pseduo clusters that have not appeared in the merge list 
    for n in range(len(PeakIndices)):
        if PeakIndices[n] in Unmarked:
            NewPI.append(PeakIndices[n])

    # Updated pseduo feature clusters information after merging
    FCluster = np.unique(NewPI)
    if len(FCluster) == 0:
        return [], [], []
    Ffitness = fitness[FCluster]
    F_Indices = Close_FCluster(FCluster, DisC, np.shape(DisC)[0])
    return FCluster, Ffitness, F_Indices
#----------------------Boundary Feature Identification------------------------------------#
def Boundary_Points(DisC, F_Indices, data, Current, Neighbor):
    
    [N, dim] = np.shape(data)
    TempCluster1 = np.where(F_Indices == Current)
    TempCluster2 = np.where(F_Indices == Neighbor)
    
    TempCluster = np.append(TempCluster1,TempCluster2)
    
    D = []
    
    for i in range(len(TempCluster)):
        D1 = DisC[TempCluster[i], Current]
        D2 = DisC[TempCluster[i], Neighbor]
        
        D.append(abs(D1 - D2))
    
    if not D:
        BD = Current
    else:
        FI = np.argmin(D)
        BD = TempCluster[FI]
#    BD = TempCluster[FI]
    
    return BD

def FC_evolve(DisC, histsummary, FCluster, Ffitness, C_Indices, Features, fitness, t, Batchsize):
    histFclusters = histsummary[:, 0]
    histFfitness = histsummary[:, 1]
    currentFcluster = FCluster
    currentFfitness = Ffitness
    newFcluster = hist_merge(DisC, currentFcluster, currentFfitness, C_Indices,
                                                           histFclusters, histFfitness, t, Batchsize)
    return newFcluster

def hist_merge(DisC, fitness, currentFcluster, currentFfitness, C_Indices, histFcluster, histFfitness, t, Batchsize):
    num_cur = np.shape(currentFcluster)[0]
    num_hist = np.shape(histFcluster)[0]
    merge_list = []
    ml1 = []
    ml2 = []
    unique_c = np.unique(C_Indices)
    for i in range(num_cur):
        M = False
        tempdist = DisC[i+num_hist, : num_hist]
        nearhist = np.argmin(tempdist)
#        neighborhist = histFcluster[nearhist]
        neighborhistf = histFfitness[nearhist]
        current = int(currentFcluster[i])
        currentf = currentFfitness[i]
        # Identify the boundary features from the current feature clusters
        temp_clusteridx = np.where(C_Indices == current)[0]
        if len(temp_clusteridx) == 0:
            temp_clusteridx = current - t * Batchsize
            BP = temp_clusteridx
        else:
            temp_clusterd1 = DisC[temp_clusteridx + num_hist, current-t*Batchsize]
            temp_clusterd2 = DisC[temp_clusteridx + num_hist, nearhist]
            bet_clusters = temp_clusterd1 + temp_clusterd2
            sort_idx = np.argsort(bet_clusters)
            if len(sort_idx) == 1:
                BP = temp_clusteridx[sort_idx[0]]
            else:
                if sort_idx[0] != current:
                    BP = temp_clusteridx[sort_idx[0]]
                else:
                    BP = temp_clusteridx[sort_idx[1]]
        BPF = fitness[BP]
        if BPF < 1 * min(currentf, neighborhistf):
            M = False
        else:
            M = True
        if M:
            if len(merge_list) == 0:
                merge_list = [[i, nearhist]]
                ml1 = [i]
                ml2 = [nearhist]
            else:
                merge_list.append([i, nearhist])
                ml1.append(i)
                ml2.append(nearhist)
    evolved = []
    evolvedf = []

    for k in range(np.shape(merge_list)[0]):
        idx1 = merge_list[k][0]
        # print(idx1)
        idx2 = merge_list[k][1]
        # print(idx2)
        if currentFfitness[idx1] > histFfitness[idx2]:
            evolved.append(currentFcluster[idx1])
            evolvedf.append(currentFfitness[idx1])
        else:
            evolved.append(histFcluster[idx2])
            evolvedf.append(histFfitness[idx2])
    for l1 in range(len(histFcluster)):
        if l1 not in ml2:
            evolved.append(int(histFcluster[l1]))
            evolvedf.append(histFfitness[l1])
    for l2 in range(len(currentFcluster)):
        if l2 not in ml1:
            evolved.append(currentFcluster[l2])
            evolvedf.append(currentFfitness[l2])
    evolved = np.unique(evolved)
    evolved = evolved.astype(int)
    # evolvedf = evolvedf[evolved - t * Batchsize]
    # evolvedI = Close_FCluster(evolved, DisC, dim)
    # return evolved, evolvedf, evolvedI
    return evolved

def Close_FCluster(FCluster, DisC, dim):
    F_Indices = np.arange(dim)
    for i in range(dim):
        dist_fcluster = DisC[i, FCluster]
        F_Indices[i] = FCluster[np.argmin(dist_fcluster)]
    return F_Indices

#------------------Pseudo Feature Generation----------------#
    """Generate the Pseudo Features using the Gaussian Distribution
    """
def PseduoGeneration(PseP,N):
    # Extract the Pseudo Feature Means and Standard deviations
    Pse_Mean = PseP[:,0]
    Pse_Std = PseP[:,1]
    # Initialize the Pseudo Data as empty array
    Data = np.zeros((N,len(Pse_Mean)))
    # Generate the Pseudo features using Gaussian Distribution
    for i in range(len(Pse_Mean)):
        Data[:, i] = (np.repeat(Pse_Mean[i],N) + Pse_Std[i] * np.random.randn(N)).T
    return Data

def Psefitness_cal( PseP, sample, data, PseduoData, StdF, gamma):
    
    OriFN = np.shape(sample)[0]
    PN = np.shape(PseP)[0]
    PsePF = np.zeros(PN)
    for i in range(PN):
        TempSum = 0
        for j in range(OriFN):
            Var1 = np.var(data[:,j])
            Var2 = np.var(PseduoData[:,i])
            
            P = np.corrcoef(data[:,j],PseduoData[:,i])[0,1]
            
            Sim = Var1 + Var2 - ((Var1 + Var2)**2 - 4 * Var1 * Var2 * (1 - P**2))**0.5
            
            D_KL = Sim / (Var1 + Var2)
            
            TempSum = TempSum + (math.exp(-(D_KL**2)/StdF))**gamma
        PsePF[i] = TempSum
    return PsePF
#--------------Define Functions for Performance Evaluation---------------#
    """
    Three Different Classifiers are defined here: Decision Tree, K-Nearest Neighborhood
    and Naive Bayes Classifier
    """
def Performance_Eval_DT(X1,X2,Y):
    
    clf1 = tree.DecisionTreeClassifier()
    clf2 = tree.DecisionTreeClassifier()
    
    scores1 = cross_val_score(clf1, X1, Y, cv=10)
    scores2 = cross_val_score(clf2, X2, Y, cv=10)
    
    return scores1,scores2

def Performance_Eval_NB(X1,X2,Y):
    
    clf1 = GaussianNB()
    clf2 = GaussianNB()
    
    scores1 = cross_val_score(clf1, X1, Y, cv=10)
    scores2 = cross_val_score(clf2, X2, Y, cv=10)
    
    return scores1,scores2

def Performance_Eval_KNN(X1,X2,Y):
    
    clf1 = KNeighborsClassifier(n_neighbors=5)
    clf2 = KNeighborsClassifier(n_neighbors=5)
    
    scores1 = cross_val_score(clf1, X1, Y, cv=10)
    scores2 = cross_val_score(clf2, X2, Y, cv=10)
    
    return scores1,scores2
    
#--------------------------------------------------------------------------------------------------------------  
if __name__ == '__main__':
    [data, label] = Input()
    [N, dim] = np.shape(data)
    label = label.reshape(N,)
    group_kfold = StratifiedKFold(n_splits=10)
    iterations = 10
    acck_hist, f1k_hist, sfk_hist = [], [], []
    accd_hist, f1d_hist, sfd_hist = [], [], []
    for it in range(iterations):
        acck_cross, f1k_cross, sfk_cross = [], [], []
        accd_cross, f1d_cross, sfd_cross = [], [], []
        for train_idx, test_idx in group_kfold.split(data,label):
            train_data = data[train_idx,:]
            train_label = label[train_idx]
            test_data = data[test_idx,:]
            test_label = label[test_idx]
            NumofBatches = round(dim/Batchsize)
            Extract_FIndices = []
            FCluster = []
            histsummary = []
            histStdF = []
            
            for i in range(NumofBatches):
                if i < NumofBatches-1:
                    process_fidx = np.arange(i * Batchsize, (i + 1) * Batchsize)
                    Features = train_data[:, process_fidx]
                else:
                    process_fidx = np.arange(i*Batchsize, dim)
                    Features = train_data[:, process_fidx]
                num_hist = np.shape(histsummary)[0]
                if num_hist == 0:
                    concate_features = Features
                else:
                    histfidx  = np.asarray(histsummary[:, 0])
                    histfidx = histfidx.astype(int)
                    concate_features = np.concatenate([train_data[:, histfidx], Features], axis = 1)
                N_F = np.shape(concate_features)[1]
                [DC_means, DC_std] = Distribution_Est(histsummary, concate_features, N_F)
                [DisC, Dist] = Feature_Dist(histsummary, concate_features, N_F)            
                StdF = max(Dist)
                gamma = 5
            
                fitness = fitness_cal(histsummary, DisC, DC_means, DC_std, Features, StdF, gamma, histStdF)
                oldfitness = np.copy(fitness)
        
                # Extract the information needed for processing the current feature chunk
                if i == 0:
                    disc_curr = DisC
                    dist_curr = Dist
                    fitness_curr = fitness
                    oldfit = np.copy(fitness)
                    curr_mean = DC_means
                    curr_std = DC_std
                else:
                    disc_curr = DisC[num_hist:, num_hist:]
                    dist_curr = Dist[num_hist:]
                    fitness_curr = fitness[num_hist:]
                    oldfit = np.copy(fitness[num_hist:])
                    curr_mean = DC_means[num_hist:]
                    curr_std = DC_std[num_hist:]
                [PeakIndices, Pfitness, C_Indices] = Pseduo_Peaks(disc_curr, dist_curr, curr_mean, curr_std, Features, fitness_curr,
                                                                  StdF, gamma)
                fitness = oldfitness
                # Pseduo Clusters Infomormation Extraction
                PseDuo = PeakIndices  # Pseduo Feature Cluster centers
                PseDuoF = Pfitness  # Pseduo Feature Clusters fitness values
                if len(PseDuo) > 1:
                    [FCluster, Ffitness, C_Indices] = Pseduo_Evolve(disc_curr, PeakIndices, PseDuoF, C_Indices, curr_mean,
                                                                    curr_std, Features, oldfit, StdF, gamma)
                else:
                    FCluster = PeakIndices
                    Ffitness = fitness[PeakIndices]
                if i > 0:
                    histFcluster = np.asarray(histsummary[:, 0])
                    histFcluster = histFcluster.astype(int)
                    histFfitness = histsummary[:, 1]
                    FCluster2 = FCluster + i * Batchsize
                    C_Indices2 = C_Indices + (i * Batchsize)
                    EvolvedFC = hist_merge(DisC, fitness_curr, FCluster2, Ffitness, C_Indices2, histFcluster, histFfitness, i, Batchsize)
                    EvolvedF = []
                    for k in range(len(EvolvedFC)):
                        if EvolvedFC[k] in histFcluster:
                            histidx = np.where(histFcluster == EvolvedFC[k])[0]
                            if len(EvolvedF) == 0:
                                EvolvedF = histFfitness[histidx]
                            else:
                                EvolvedF = np.append(EvolvedF, histFfitness[histidx])
                        else:
                            curridx = np.where(FCluster == (EvolvedFC[k] - i*Batchsize))[0]
                            if len(EvolvedF) == 0:
                                EvolvedF = fitness[curridx]
                            else:
                                EvolvedF = np.append(EvolvedF, fitness[curridx])
                else:
                    EvolvedFC = FCluster
                    EvolvedF = Ffitness
                SF = EvolvedFC
                SFfit = EvolvedF
                current_summary = np.asarray([SF, SFfit])
                current_summary = current_summary.T
#                current_summary = np.reshape(current_summary, (len(SF),2))
                histsummary = current_summary
                histStdF.append(StdF)
            Extract_FIndices = SF
            Extract_FIndices = Extract_FIndices.astype(int)
            
            Extract_FIndices = SF
            Extract_FIndices = Extract_FIndices.astype(int)
            
            clf1 = KNeighborsClassifier(n_neighbors=5)
            clf2 = tree.DecisionTreeClassifier()
            
            clf1 = clf1.fit(train_data[:, Extract_FIndices], train_label)
            clf2 = clf2.fit(train_data[:, Extract_FIndices], train_label)
            
            predict_label1 = clf1.predict(test_data[:, Extract_FIndices])
            predict_label2 = clf2.predict(test_data[:, Extract_FIndices])
            
            accuracy1 = accuracy_score(test_label, predict_label1)
            f11 = f1_score(test_label, predict_label1,average='macro')
            
            accuracy2 = accuracy_score(test_label, predict_label2)
            f12 = f1_score(test_label, predict_label2,average='macro')
            
            acck_cross.append(accuracy1)
            f1k_cross.append(f11)
            sfk_cross.append(len(SF))
            accd_cross.append(accuracy2)
            f1d_cross.append(f12)
            sfd_cross.append(len(SF))  
        
        acck_hist.append(np.mean(acck_cross))
        f1k_hist.append(np.mean(f1k_cross)) 
        sfk_hist.append(np.mean(sfk_cross))
        
        accd_hist.append(np.mean(accd_cross))
        f1d_hist.append(np.mean(f1d_cross)) 
        sfd_hist.append(np.mean(sfd_cross))
        print("Iteration "+str(it)+" Finished!")
    print("Results on KNN classifiers:")    
    print('10 runs average of 10-fold cross validated accuracy: ', np.mean(acck_hist))
    print('10 runs std of 10-fold cross validated accuracy: ', np.std(acck_hist))
    print('10 runs average of 10-fold cross validated f1: ', np.mean(f1k_hist))
    print('10 runs std of 10-fold cross validated f1: ', np.std(f1k_hist))
    print('10 runs average of 10-fold cross validated time: ', np.mean(sfk_hist))
    
    print("Results on DT classifiers:")    
    print('10 runs average of 10-fold cross validated accuracy: ', np.mean(accd_hist))
    print('10 runs std of 10-fold cross validated accuracy: ', np.std(accd_hist))
    print('10 runs average of 10-fold cross validated f1: ', np.mean(f1d_hist))
    print('10 runs std of 10-fold cross validated f1: ', np.std(f1d_hist))
    print('10 runs average of 10-fold cross validated time: ', np.mean(sfd_hist))
