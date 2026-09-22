from .all_func import VelocityGrid, vor_volumes, smooth_matrix, is_addnode
import concurrent.futures
from scipy.interpolate import RBFInterpolator
import pandas as pd
import numpy as np
from zipfile import ZipFile
import logging
import os
import random
import scipy.sparse as scsp

def run_forward(modvel, modvel_outer, source_list, receiver_list, phase_list, delt, deltn, xfac, iter1, iter2,
                tmin, folder_name, rnd_koef=1,nu_cpu=os.cpu_count()-1):
    f_path='./'+folder_name
    if not os.path.exists(f_path):
        os.makedirs(f_path)
    #load model parameter
    logger = logging.getLogger()
    logFileFormatter = logging.Formatter(
        fmt=f"%(levelname)s %(asctime)s : - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fileHandler = logging.FileHandler(filename=folder_name+'/app.log',mode='w')
    fileHandler.setFormatter(logFileFormatter)
    fileHandler.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logging.getLogger().setLevel(logging.DEBUG)
    logger.info("\n delt: "+str(delt)+
                "\n deltn: "+str(deltn)+
                "\n xfac: "+str(xfac)+
                "\n iter1: "+str(iter1)+
                "\n iter2: "+str(iter2)+
                "\n tmin: "+str(tmin))

    print('start forward modeling only')
    logger.info('start forward modeling only')
    #modvel.columns = modvel.iloc[0]
    #modvel=modvel[1:]
    modvel = modvel.apply(pd.to_numeric)
    #modvel_outer.columns = modvel_outer.iloc[0]
    #modvel_outer=modvel_outer[1:]
    modvel_outer = modvel_outer.apply(pd.to_numeric)
    node = np.vstack((modvel.X, modvel.Y, modvel.Z)).T
    node_outer = np.vstack((modvel_outer.X, modvel_outer.Y, modvel_outer.Z)).T
    node_all = np.vstack((node, node_outer))
    vel_allP = np.hstack((modvel.Vp, modvel_outer.Vp))
    vel_allS = np.hstack((modvel.Vs, modvel_outer.Vs))
    delt = float(delt)
    deltn = float(deltn)
    xfac = float(xfac)
    iter1 = int(iter1)
    iter2 = int(iter2)
    tmin = float(tmin)

    xmin = np.amin(modvel.X)
    xmax = np.amax(modvel.X)
    ymin = np.amin(modvel.Y)
    ymax = np.amax(modvel.Y)
    zmin = np.amin(modvel.Z)
    zmax = np.amax(modvel.Z)

    #interpolate velocity model into regular grid
    xnode = np.arange(xmin, xmax+deltn, deltn)
    ynode = np.arange(ymin, ymax+deltn, deltn)
    znode = np.arange(zmin, zmax+deltn, deltn)
    Xinter, Yinter, Zinter = np.meshgrid(xnode, ynode, znode, indexing='ij')
    interpP = RBFInterpolator(node_all,vel_allP,kernel='linear',neighbors=8)
    interpS = RBFInterpolator(node_all,vel_allS,kernel='linear',neighbors=8)
    gridVp = np.round(interpP(np.column_stack((Xinter.flatten(), Yinter.flatten(), Zinter.flatten()))),
                       decimals=4).reshape(len(xnode),len(ynode),len(znode))
    gridVs = np.round(interpS(np.column_stack((Xinter.flatten(), Yinter.flatten(), Zinter.flatten()))),
                       decimals=4).reshape(len(xnode),len(ynode),len(znode))
    #determine velgrid object
    velgridP = VelocityGrid(node, xnode, ynode, znode, gridVp, deltn, delt, xfac, iter1, iter2, tmin)
    velgridS = VelocityGrid(node, xnode, ynode, znode, gridVs, deltn, delt, xfac, iter1, iter2, tmin)
    vel_node_P=np.array(modvel.Vp)
    vel_node_S=np.array(modvel.Vs)

    #load data
    source_list.columns=['id','easting','northing','depth','type']
    cols_to_convert = ['easting', 'northing', 'depth']
    source_list[cols_to_convert] = source_list[cols_to_convert].apply(pd.to_numeric)
    source_list['elevation'] = source_list['depth'] * (-1)
    receiver_list.columns=['id','easting','northing','elevation']
    cols_to_convert = ['easting', 'northing', 'elevation']
    receiver_list[cols_to_convert] = receiver_list[cols_to_convert].apply(pd.to_numeric)
    phase_list.columns=['id_event','id_sta','t_time','phase']
    phase_list['t_time']=phase_list['t_time'].apply(pd.to_numeric)
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
        ttobs_P.append(float(phase_listP.iloc[i,2]))

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
        ttobs_S.append(float(phase_listS.iloc[i,2]))

    phase_listS_use = pd.DataFrame(phase_listS_use)

    #start initial forward modeling
    print('start forward')
    logger.info('start forward')
    ttcal_P = []
    path_list_P =[]
    phase_listP_use_dum=[]

    with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
        results = executor.map(velgridP.safe_ttime_only,paths_P, chunksize=1)
        for i, result in enumerate(results):
            if isinstance(result, dict) and "error" in result:
                logger.info('not convergen in path P: ' +' '.join(phase_listP_use.iloc[i, :].astype(str)))
                logger.info(result['type'])
                logger.info(result['error'])
                logger.info(result['traceback'])
                continue
            ttcal_i = result[1]
            rnd_noise = random.uniform(-rnd_koef, rnd_koef)
            ttcal_i = ttcal_i * (1 + (rnd_noise / 100))
            ttcal_P.append(float(ttcal_i))
            path_list_P.append(result[0])
            ttcal_P.append(float(ttcal_i))
            phase_listP_use_dum.append(phase_listP_use.iloc[i,:])



    phase_listP_use = pd.DataFrame(phase_listP_use_dum)

    #save initial ray-tracing file
    ray1=path_list_P[0]
    ray2=np.ones((len(ray1),1))*0
    ray_out=np.hstack((ray1,ray2))
    for i in range(1,len(path_list_P)):
        ray1=path_list_P[i]
        ray2=np.ones((len(ray1),1))*i
        ray3=np.hstack((ray1,ray2))
        ray_out = np.vstack((ray_out, ray3))
    np.savetxt(folder_name+'/_raypath',ray_out,delimiter=',',fmt='%.4f')

    ttcal_S = []
    path_list_S =[]
    phase_listS_use_dum = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
        results = executor.map(velgridS.safe_ttime_only,paths_S, chunksize=1)
        for i, result in enumerate(results):
            if isinstance(result, dict) and "error" in result:
                logger.info('not convergen in path S: ' +' '.join(phase_listS_use.iloc[i, :].astype(str)))
                logger.info(result['type'])
                logger.info(result['error'])
                logger.info(result['traceback'])
                continue
            ttcal_i = result[1]
            rnd_noise = random.uniform(-rnd_koef, rnd_koef)
            ttcal_i = ttcal_i * (1 + (rnd_noise / 100))
            ttcal_S.append(float(ttcal_i))
            path_list_S.append(result[0])
            ttcal_S.append(float(ttcal_i))
            phase_listS_use_dum.append(phase_listP_use.iloc[i,:])


    phase_listS_use = pd.DataFrame(phase_listS_use_dum)

    print('add syntetik data to file')
    logger.info('add syntetik data to file')
    phase_listP_use['t_sin']=np.array(ttcal_P)
    phase_listS_use['t_sin']=np.array(ttcal_S)
    pd_out=pd.concat([phase_listP_use,phase_listS_use])
    pd_out.to_csv(folder_name + '/synthetic.csv', index=False)
    with ZipFile(folder_name+'/compress.zip', 'w') as zip:
        zip.write(folder_name+'/synthetic.csv')


    print('process is done')
    logger.info('process is done')