from all_func import VelocityGrid, vor_volumes, smooth_matrix, is_addnode
from copy import copy
from scipy.sparse.linalg import lsqr
import concurrent.futures
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
import logging

logger=logging.getLogger(__name__)
logFileFormatter = logging.Formatter(fmt=f"%(levelname)s %(asctime)s : - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",)
fileHandler = logging.FileHandler(filename='app.log',mode='w')
fileHandler.setFormatter(logFileFormatter)
fileHandler.setLevel(logging.INFO)
logger.addHandler(fileHandler)
logging.getLogger().setLevel(logging.DEBUG)


if __name__ == '__main__':
    #load model parameter
    modvel = pd.read_csv("vel_result17.csv")
    modvel_outer = pd.read_csv("modawal_outer_agu20.csv")
    node = np.vstack((modvel.X, modvel.Y, modvel.Z)).T
    node_outer = np.vstack((modvel_outer.X, modvel_outer.Y, modvel_outer.Z)).T
    node_all = np.vstack((node, node_outer))
    vel_allP = np.hstack((modvel.Vp, modvel_outer.Vp))
    vel_allS = np.hstack((modvel.Vs, modvel_outer.Vs))
    delt = 0.25
    deltn = 0.5
    xfac = 1.3
    iter1 = 50
    iter2 = 50
    tmin = 0.0001
    iteration_number = 20
    up_threshold = 500
    low_threshold = 75
    d_rms=0.002
    r_time_P = 0.8
    r_time_S = 0.8
    damping_1 = 10
    damping_2 = 0.1
    update_grid = False
    logger.info("\n delt: "+str(delt)+
                "\n deltn: "+str(deltn)+
                "\n xfac: "+str(xfac)+
                "\n iter1: "+str(iter1)+
                "\n iter2: "+str(iter2)+
                "\n tmin: "+str(tmin)+
                "\n iteration_number: "+str(iteration_number)+
                "\n up_threshold: "+str(up_threshold)+
                "\n low_threshold: "+str(low_threshold)+
                "\n d_rms: "+str(d_rms)+
                "\n r_time_P: "+str(r_time_P)+
                "\n r_time_S: "+str(r_time_S)+
                "\n damping_1: "+str(damping_1)+
                "\n damping_2: "+str(damping_2)+
                "\n update_grid: "+str(update_grid))

    xmin = 310.5
    xmax = 360.5
    ymin = 9059.5
    ymax = 9095.5
    zmin = -52
    zmax = 8

    #interpolate velocity model into regular grid
    xnode = np.arange(xmin, xmax+deltn, deltn)
    ynode = np.arange(ymin, ymax+deltn, deltn)
    znode = np.arange(zmin, zmax+deltn, deltn)
    Xinter, Yinter, Zinter = np.meshgrid(xnode, ynode, znode, indexing='ij')
    interpP = LinearNDInterpolator(node_all,vel_allP)
    interpS = LinearNDInterpolator(node_all,vel_allS)
    gridVp = np.round_(interpP(Xinter, Yinter, Zinter),decimals=4)
    gridVs = np.round_(interpS(Xinter, Yinter, Zinter), decimals=4)
    #determine velgrid object
    velgridP = VelocityGrid(node, xnode, ynode, znode, gridVp, deltn, delt, xfac, iter1, iter2, tmin)
    velgridS = VelocityGrid(node, xnode, ynode, znode, gridVs, deltn, delt, xfac, iter1, iter2, tmin)
    vel_node_P=np.array(modvel.Vp)
    vel_node_S=np.array(modvel.Vs)

    #load data
    source_list=pd.read_csv('event.dat', sep= '\s+', header=None, names=['id','easting','northing','depth'])
    source_list['elevation'] = source_list['depth'] * (-1)
    receiver_list=pd.read_csv('stasiun2.dat', sep= '\s+', header=None, names=['id','easting','northing','elevation'])
    phase_list=pd.read_csv('phase_same', header=None, names=['id_event','id_sta','t_time','phase'])
    phase_listP = phase_list[phase_list['phase']=='P']
    phase_listS = phase_list[phase_list['phase']=='S']
    paths_P=[]
    ttobs_P=[]
    paths_S=[]
    ttobs_S=[]
    source_list_invers=source_list.copy()
    source_list_invers['to_update']=0
    source_list_invers['to_update_i']=0


    #filter phase data (menghilangkan data yang diluar treshold)
    phase_listP_use=[]
    for i in range (len(phase_listP)):
        source=np.array(source_list[source_list['id']==phase_listP.iloc[i,0]][['easting','northing','elevation']])
        receiver=np.array(receiver_list[receiver_list['id']==phase_listP.iloc[i,1]][['easting','northing','elevation']])
        if (source.size == 0) or (receiver.size == 0):
            continue
        if (float(phase_listP.iloc[i,2]))<=0:
            continue

        phase_listP_use.append(phase_listP.iloc[i,:])
        path = np.vstack((source, receiver))
        paths_P.append(path)
        ttobs_P.append(float(phase_listP.iloc[i][2]))

    phase_listP_use=pd.DataFrame(phase_listP_use)


    phase_listS_use=[]
    for i in range (len(phase_listS)):
        source=np.array(source_list[source_list['id']==phase_listS.iloc[i,0]][['easting','northing','elevation']])
        receiver=np.array(receiver_list[receiver_list['id']==phase_listS.iloc[i,1]][['easting','northing','elevation']])
        if (source.size == 0) or (receiver.size == 0):
            continue
        if (float(phase_listS.iloc[i,2]))<=0:
            continue
        phase_listS_use.append(phase_listS.iloc[i,:])
        path = np.vstack((source, receiver))
        paths_S.append(path)
        ttobs_S.append(float(phase_listS.iloc[i][2]))

    phase_listS_use = pd.DataFrame(phase_listS_use)

    #start initial forward modeling
    print('start forward')
    logger.info('start forward')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f1 = [executor.submit(velgridP.forwardtwopoints, pat) for pat in paths_P]

    ttcal_P = []
    path_list_P =[]
    vp_kernel = np.zeros((len(paths_P), node.shape[0]))
    hypo_kernel_P = np.zeros((len(paths_P),len(source_list)*4))
    phase_listP_use_dum=[]
    ttobsP_dum=[]
    for i in range(0, len(paths_P)):
        try:
            grpli = f1[i]._result
            krn_i = grpli[2]
            ttcal_i = grpli[1]
            hypoder_i = grpli[3]
            path_list_P.append(grpli[0])
            vp_kernel[i, :] = (np.array(krn_i))
            event_index_i=int((source_list.index[source_list.iloc[:,0]==int(phase_listP_use.iloc[i,0])]).tolist()[0])
            hypo_kernel_P[i,(event_index_i*4):(event_index_i*4)+4]=hypoder_i
            ttcal_P.append(float(ttcal_i))
            phase_listP_use_dum.append(phase_listP_use.iloc[i,:])
            ttobsP_dum.append(ttobs_P[i])
        except:
            continue

    ttobs_P = copy(ttobsP_dum)
    vp_kernel = vp_kernel[~np.all(vp_kernel == 0, axis=1)]
    hypo_kernel_P = hypo_kernel_P[~np.all(hypo_kernel_P == 0, axis=1)]
    phase_listP_use = pd.DataFrame(phase_listP_use_dum)

    #simpan ray-tracing file
    ray1=path_list_P[0]
    ray2=np.ones((len(ray1),1))*0
    ray_out=np.hstack((ray1,ray2))
    for i in range(1,len(path_list_P)):
        ray1=path_list_P[i]
        ray2=np.ones((len(ray1),1))*i
        ray3=np.hstack((ray1,ray2))
        ray_out = np.vstack((ray_out, ray3))
    np.savetxt('ray_awal',ray_out,delimiter=',',fmt='%.4f')

    with concurrent.futures.ProcessPoolExecutor() as executor:
        f1 = [executor.submit(velgridS.forwardtwopoints, pat) for pat in paths_S]

    ttcal_S = []
    path_list_S =[]
    vs_kernel = np.zeros((len(paths_S), node.shape[0]))
    hypo_kernel_S = np.zeros((len(paths_S),len(source_list)*4))
    phase_listS_use_dum = []
    ttobsS_dum = []
    for i in range(0, len(paths_S)):
        try:
            grpli = f1[i]._result
            krn_i = grpli[2]
            ttcal_i = grpli[1]
            hypoder_i = grpli[3]
            path_list_S.append(grpli[0])
            vs_kernel[i, :] = (np.array(krn_i))
            event_index_i=int((source_list.index[source_list.iloc[:,0]==float(phase_listS_use.iloc[i,0])]).tolist()[0])
            hypo_kernel_S[i,(event_index_i*4):(event_index_i*4)+4]=hypoder_i
            ttcal_S.append(float(ttcal_i))
            phase_listS_use_dum.append(phase_listS_use.iloc[i, :])
            ttobsS_dum.append(ttobs_S[i])
        except:
            continue

    ttobs_S = copy(ttobsS_dum)
    vs_kernel = vs_kernel[~np.all(vs_kernel == 0, axis=1)]
    hypo_kernel_S = hypo_kernel_S[~np.all(hypo_kernel_S == 0, axis=1)]
    phase_listS_use = pd.DataFrame(phase_listS_use_dum)

    print('add syntetik data to file')
    phase_listP_use['t_sin']=np.array(ttcal_P)
    phase_listS_use['t_sin']=np.array(ttcal_S)
    pd_out=pd.concat([phase_listP_use,phase_listS_use])
    pd_out.to_csv('synthetic_time.csv', index=False)
    print('process is done')
