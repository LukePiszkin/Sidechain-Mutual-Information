### mutual_info_sidechain.py ###
### script for predicting the ring current shift effects based on MD simulation. ###

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np
import pytraj as pt
import time
import multiprocessing
from multiprocessing import Process
import json
import scipy
import math
import sys
from adaptive_hist import *

## style stuff for matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]
mpl.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

### ----------------------------------------------------------------------------------------- ###
## Define functions ##
def D_vec(atoms,frame,pair_iterate):
    global num_slice
    global pbar
    global num_trials
    
    pairs = []
    for a1 in atoms:
        for a2 in atoms:
            if a1 != a2:
                if [a2,a1] not in pairs:
                    pairs.append([a1,a2])
    
    if pair_iterate == 0: # only update pbar once the last process starts
        pbar.update(100*(1/(4*29999/num_slice)))
        
    return pt.distance(traj[[frame]],pairs)
                
def dist_mat(atoms,start_frame,end_frame,pair_iterate):
    dist_mat = []
    dist_avgs = [0 for ele in range(len(D_vec(atoms,0,1)))]
    dist_list = [] 

    for f in range(start_frame,end_frame):
        dist_list.append(D_vec(atoms,f,pair_iterate))
    
    dist_mat = [[0 for ele in dist_avgs] for ele in dist_avgs]
    
    for i in range(len(dist_avgs)):
        for j in range(len(dist_avgs)):
            disti_avg = np.mean([dist_list[f][i] for f in range(start_frame,end_frame)])
            distj_avg = np.mean([dist_list[f][j] for f in range(start_frame,end_frame)])
            
            dist_mat[i][j] = np.sum([(dist_list[f][i] - disti_avg)*(dist_list[f][j] - distj_avg) for f in range(start_frame,end_frame)])/(end_frame-start_frame-1)
            
    return dist_mat

def proj(residue,start_frame,end_frame,pair_iterate):
    atoms = pt.select(':'+str(residue)+'@N,C,CA,CB,CG,CG1,CG2,CD,CD1,CD2,CE,CE1,CE2,CZ,CZ1,CZ2',traj.top)
    eigvals, eigvecs = np.linalg.eig(dist_mat(atoms,start_frame,end_frame,pair_iterate))

    normalized_eigvals = [ele/np.sum(eigvals) for ele in eigvals]

    cum_sum = []
    contr = 0
    for i in range(len(eigvals)):
        contr = contr + normalized_eigvals[i]
        if contr < 0.8:
            cum_sum.append(contr)
        else:
            normalized_eigvals[i] = contr-0.8
            cum_sum.append(contr)
            break
    
    eig_sum = []
    for j in range(len(eigvecs[0])):
        eig_sum_j = 0
        for i in range(len(cum_sum)):
            eig_sum_j = eig_sum_j + normalized_eigvals[i]*eigvecs[i][j]
        eig_sum.append(eig_sum_j)
    
    D_vecs = [D_vec(atoms,f,pair_iterate) for f in range(start_frame,end_frame)]
    
    return_vec = [np.dot([ele for inner in D_vecs[f] for ele in inner],eig_sum) for f in range(start_frame,end_frame)]
    return return_vec

def joint_dist(res1,res2,num_bins,start_frame,end_frame,pair_iterate):
    proj1_list = proj(res1,start_frame,end_frame,pair_iterate)
    proj2_list = proj(res2,start_frame,end_frame,pair_iterate)

    pdf, xedges, yedges = np.histogram2d(proj1_list,proj2_list,bins=num_bins)
    norm = np.sum(pdf)
    pdf = [[pdf[i][j]/norm for j in range(len(pdf[i]))] for i in range(len(pdf))]
    
    return pdf

def mutual_info(res1,res2,num_bins,start_frame,end_frame,pair_iterate): # manually selection of the number of bins
    global num_pairs
    mi = 0
    joint_dist_xy = joint_dist(res1,res2,num_bins,start_frame,end_frame,pair_iterate)
    x_dist = [np.sum([joint_dist_xy[j][i] for i in range(len(joint_dist_xy[j]))]) for j in range(len(joint_dist_xy))]
    y_dist = [np.sum([joint_dist_xy[i][j] for i in range(len(joint_dist_xy[j]))]) for j in range(len(joint_dist_xy))]
    
    ## from probability
    for i in range(num_bins): 
        for j in range(num_bins):
            if joint_dist_xy[i][j] != 0 and x_dist[i] != 0 and y_dist[j] != 0:
                mi = mi + joint_dist_xy[i][j]*np.log(joint_dist_xy[i][j]/(x_dist[i]*y_dist[j]))

    ## calculate MI from entropies, it should give same answer.
    # ent_x = -np.sum([x*np.log(x) for x in x_dist if x != 0])
    # ent_y = -np.sum([y*np.log(y) for y in y_dist if y != 0])
    # ent_xy = 0 
    # for x in range(len(joint_dist_xy)):
    #     for y in range(len(joint_dist_xy)):
    #         if joint_dist_xy[x][y] != 0:
    #             ent_xy = ent_xy - joint_dist_xy[x][y]*np.log(joint_dist_xy[x][y])
    
    # mi = ent_x + ent_y - ent_xy

    ## rescale MI
    # mi = 2*mi/(ent_x + ent_y)
    # mi = np.sqrt(1-np.exp(-2*mi))
    
    return mi

def mutual_info_adaptive(res1,res2,start_frame,end_frame,pair_iterate): # create joint pdf using adaptive_hist routine
    x = proj(res1,start_frame,end_frame,pair_iterate)
    y = proj(res2,start_frame,end_frame,pair_iterate)
    
    pdf,x_marg,y_marg = adaptive_hist(x,y,False)
    mi = mutual_info_adapt(pdf,x_marg,y_marg)
    
    return mi

def run_mi(pair,num_bins,start_frame,end_frame,pair_iterate):
    global pairs
    global residues

    mi_matrix[tuple([residues[pair[0]],residues[pair[1]]])] = mutual_info(residues[pair[0]],residues[pair[1]],num_bins,start_frame,end_frame,pair_iterate)

def run_mi_adapt(pair,start_frame,end_frame,pair_iterate):
    global pairs
    global residues

    mi =  mutual_info_adaptive(residues[pair[0]],residues[pair[1]],start_frame,end_frame,pair_iterate)
    
    ## correct or normalize mi
    mi = np.sqrt(1-np.exp(-2*mi))
    
    mi_matrix[tuple([residues[pair[0]],residues[pair[1]]])] = mi
    
### ----------------------------------------------------------------------------------------- ###
if __name__ == "__main__":
    
    ## define globals, call functions, and plot ##
    num_slice = 1 # load every Xth frame of the data
    num_cores = 64 # how many cores 
    total_frames = 30000  # how many frames are in your trajectory
    
    ## load data
    exp_traj = 'example.nc'
    exp_parm = 'example.parm7'

    traj = pt.iterload(exp_traj,exp_parm, frame_slice=(0,total_frames,num_slice))

    residues = [40,41,42,43]  # which residues to analysis
    
    frames = [0,int(total_frames/num_slice)]

    ## make list shareable between multiprocesses
    manager = multiprocessing.Manager()
    mi_matrix = manager.dict()
    
    pairs = []
    for res1 in range(len(residues)):
        for res2 in range(len(residues)):
            if res1 != res2:
                if [res1,res2] not in pairs:
                    if [res2,res1] not in pairs:
                        pairs.append([res1,res2])
                    
    num_pairs = len(pairs)

    num_trials = math.ceil(num_pairs/num_cores)
    job_bins = []
    for i in range(num_trials+1):
        job_bins.append(num_cores*i)
    job_bins[-1] = num_pairs

    start_time = time.time()
    procs = [] 
    with tqdm(total=100,disable=False) as pbar: # progress bar
        for n in range(num_trials):
            for i, pair in enumerate(pairs[job_bins[n]:job_bins[n+1]]):
                proc = Process(target=run_mi_adapt,args=(pair,frames[0],frames[1],i))
                procs.append(proc)
                proc.start()
            
            for proc in procs:
                proc.join()
            print('Pairs ' + str(n+1) + '/' + str(num_trials) + ' completed')
            
    end_time = time.time()
    
    plot_matrix = [[0 for ele in residues] for ele in residues]
    res_pairs = mi_matrix.keys()

    p = 0
    for p in res_pairs:
        plot_matrix[residues.index(p[0])][residues.index(p[1])] = mi_matrix[p]
        plot_matrix[residues.index(p[1])][residues.index(p[0])] = mi_matrix[p]
    
    with open('mutual_info_example.json','w') as f:
        json.dump(plot_matrix.copy(), f)
        
    print(plot_matrix)
    print('Time per pair: ' + str((end_time-start_time)/num_pairs) + ' seconds')
    plt.imshow(plot_matrix, interpolation='nearest',origin='lower',cmap='Reds',vmin=0,vmax=1)
    plt.xlabel('residues')
    plt.ylabel('residues')
    plt.xticks([ele for ele in range(len(residues))],[str(residues[i]+31) for i in range(len(residues))])
    plt.yticks([ele for ele in range(len(residues))],[str(residues[i]+31) for i in range(len(residues))])
    plt.colorbar()
    plt.tight_layout()
    plt.show()