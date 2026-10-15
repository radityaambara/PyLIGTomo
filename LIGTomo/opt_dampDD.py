from .all_func import VelocityGrid, vor_volumes, smooth_matrix, is_addnode, count_tensor_dens, update_ttobs_DD
from copy import copy
from scipy.sparse.linalg import lsmr
import scipy.sparse as scsp
import concurrent.futures
from scipy.spatial import Delaunay
from scipy.interpolate import RBFInterpolator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from zipfile import ZipFile
import logging
import os


def run_opt_paramDD(modvel, modvel_outer, source_list, receiver_list, phase_list, dt_list, damping_list, delt, deltn, xfac, iter1, iter2, tmin,
                 up_threshold, low_threshold, dens_thres, d_rms, r_time_P, r_time_S, max_sep, weightDD,
                 update_grid,folder_name,if_art=True,update_grid_after=False,nu_cpu=os.cpu_count()-1):
    #load model parameter
    f_path='./'+folder_name
    if not os.path.exists(f_path):
        os.makedirs(f_path)
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
                "\n tmin: "+str(tmin)+
                "\n up_threshold: "+str(up_threshold)+
                "\n low_threshold: "+str(low_threshold)+
                "\n dens_thres: "+str(dens_thres)+
                "\n d_rms: "+str(d_rms)+
                "\n r_time_P: "+str(r_time_P)+
                "\n r_time_S: "+str(r_time_S)+
                "\n max_sep: "+str(max_sep)+
                "\n weightDD: "+str(weightDD)+
                "\n damping_hypo: "+str(damping_hypo)+
                "\n damping_p: "+str(damping_p)+
                "\n damping_s: "+str(damping_s)+
                "\n damping_smooth: "+str(damping_smooth)+
                "\n update_grid: "+str(update_grid))

    print('running in progress')
    logger.info('running in progress')

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
    up_threshold = float(up_threshold)
    low_threshold = float(low_threshold)
    d_rms=float(d_rms)
    r_time_P = float(r_time_P)
    r_time_S = float(r_time_S)
    damping_hypo = float(damping_hypo)
    damping_p = float(damping_p)
    damping_s = float(damping_s)
    damping_smooth = float(damping_smooth)
    update_grid = bool(update_grid)
    update_grid_after = bool(update_grid_after)
    folder_name=str(folder_name)


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
    gridVp = (np.round(interpP(np.column_stack((Xinter.flatten(), Yinter.flatten(), Zinter.flatten()))),
                       decimals=4).reshape(len(xnode),len(ynode),len(znode))).astype(np.float32)
    gridVs = (np.round(interpS(np.column_stack((Xinter.flatten(), Yinter.flatten(), Zinter.flatten()))),
                       decimals=4).reshape(len(xnode),len(ynode),len(znode))).astype(np.float32)

    #determine velgrid object
    velgridP = VelocityGrid(node, xnode, ynode, znode, gridVp, deltn, delt, xfac, iter1, iter2, tmin, if_art)
    velgridS = VelocityGrid(node, xnode, ynode, znode, gridVs, deltn, delt, xfac, iter1, iter2, tmin, if_art)
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
    phase_listP = phase_list[phase_list['phase']=='P'].reset_index(drop=True)
    phase_listS = phase_list[phase_list['phase']=='S'].reset_index(drop=True)
    DDphase_P = dt_list[dt_list["phase"] == "P"].reset_index(drop=True)
    DDphase_S = dt_list[dt_list["phase"] == "S"].reset_index(drop=True)
    paths_P=[]
    ttobs_P=[]
    types_P=[]
    paths_S=[]
    ttobs_S=[]
    types_S=[]

    #source_list_invers=source_list.copy()
    #source_list_invers['to_update']=0
    #source_list_invers['to_update_i']=0
    hypo_list = (source_list[source_list['type'] == 'e'].copy().reset_index(drop=True))
    hypo_list['to_update']=0.0
    hypo_list['to_update_i']=0.0
    blast_list = (source_list[source_list['type'] == 'b'].copy().reset_index(drop=True))
    blast_list['to_update']=0.0
    blast_list['to_update_i']=0.0

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
        types_P.append(str((source_list[source_list['id']==phase_listP.iloc[i,0]]['type']).item()))

    phase_listP_use=pd.DataFrame(phase_listP_use).reset_index(drop=True)


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
        types_S.append(str((source_list[source_list['id'] == phase_listS.iloc[i, 0]]['type']).item()))

    phase_listS_use = pd.DataFrame(phase_listS_use).reset_index(drop=True)

    DDphase_P_use=[]
    for i in range(len(DDphase_P)):
        row = DDphase_P.iloc[i, :]
        source1 = row['ev1']
        source2 = row['ev2']
        receiver = row['id_sta']
        if (source1 not in source_list['id'].values) or (source2 not in source_list['id'].values) or \
                (receiver not in receiver_list['id'].values):
            continue
        DDphase_P_use.append(row)
    DDphase_P_use=pd.DataFrame(DDphase_P_use).reset_index(drop=True)

    DDphase_S_use=[]
    for i in range(len(DDphase_S)):
        row = DDphase_S.iloc[i, :]
        source1 = row['ev1']
        source2 = row['ev2']
        receiver = row['id_sta']
        if (source1 not in source_list['id'].values) or (source2 not in source_list['id'].values) or \
                (receiver not in receiver_list['id'].values):
            continue
        DDphase_S_use.append(row)
    DDphase_S_use=pd.DataFrame(DDphase_S_use).reset_index(drop=True)

    #start initial forward modeling
    print('start forward')
    logger.info('start forward')
    ttcal_P = []
    path_list_P =[]
    phase_listP_use_dum=[]
    ttobsP_dum=[]

    if len(hypo_list) != 0:
        hypo_id_to_idx = {int(val): idx for idx, val in enumerate(hypo_list.iloc[:, 0])}
    #v_rows, v_cols, v_data = [], [], []
    vp_kernel=[]
    hypo_rows, hypo_cols, hypo_data = [], [], []
    successful_row_idx = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
        results = executor.map(velgridP.safe_forwardtwopoints,paths_P, chunksize=1)
        for i, result in enumerate(results):
            if isinstance(result, dict) and "error" in result:
                logger.info('not convergen in path P: ' +' '.join(phase_listP_use.iloc[i, :].astype(str)))
                logger.info(result['type'])
                logger.info(result['error'])
                logger.info(result['traceback'])
                continue

            ttcal_i = result[1]
            hypoder_i = result[3]
            vp_kernel.append(result[2])
            path_list_P.append(result[0])
            ttcal_P.append(float(ttcal_i))
            phase_listP_use_dum.append(phase_listP_use.iloc[i,:])
            ttobsP_dum.append(ttobs_P[i])
            #v_nz = np.nonzero(krn_i)[0]
            #if v_nz.size > 0:
            #    v_rows.extend([successful_row_idx] * v_nz.size)
            #    v_cols.extend(v_nz)
            #    v_data.extend(krn_i[v_nz])
            if len(hypo_list) != 0 and types_P[i] == 'e':
                phase_id = int(phase_listP_use.iloc[i, 0])
                if phase_id in hypo_id_to_idx:
                    event_index_i = hypo_id_to_idx[phase_id]
                    start_col = event_index_i * 4
                    for offset, val in enumerate(hypoder_i):
                        if val != 0:
                            hypo_rows.append(successful_row_idx)
                            hypo_cols.append(start_col + offset)
                            hypo_data.append(val)
            successful_row_idx += 1

    ttobs_P = copy(ttobsP_dum)
    phase_listP_use = pd.DataFrame(phase_listP_use_dum).reset_index(drop=True)
    vp_kernel = scsp.vstack(vp_kernel).tocsr()
    if len(hypo_list) != 0:
        hypo_kernel_P = scsp.csr_array((hypo_data, (hypo_rows, hypo_cols)),shape=(successful_row_idx, len(hypo_list) * 4))
    #save initial ray-tracing file
    ray_out = []
    for i, ray in enumerate(path_list_P):
        ray_idx = np.full((len(ray), 1), i)
        ray_out.append(np.hstack((ray, ray_idx)))
    ray_out = np.vstack(ray_out)
    np.savetxt(folder_name+'/_ray_initial',ray_out,delimiter=',',fmt='%.4f')

    ttcal_S = []
    path_list_S =[]
    phase_listS_use_dum=[]
    ttobsS_dum=[]

    if len(hypo_list) != 0:
        hypo_id_to_idx = {int(val): idx for idx, val in enumerate(hypo_list.iloc[:, 0])}
    vs_kernel=[]
    hypo_rows, hypo_cols, hypo_data = [], [], []
    successful_row_idx = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
        results = executor.map(velgridS.safe_forwardtwopoints,paths_S, chunksize=1)
        for i, result in enumerate(results):
            if isinstance(result, dict) and "error" in result:
                logger.info('not convergen in path S: ' +' '.join(phase_listS_use.iloc[i, :].astype(str)))
                logger.info(result['type'])
                logger.info(result['error'])
                logger.info(result['traceback'])
                continue
            ttcal_i = result[1]
            hypoder_i = result[3]
            path_list_S.append(result[0])
            vs_kernel.append(result[2])
            ttcal_S.append(float(ttcal_i))
            phase_listS_use_dum.append(phase_listS_use.iloc[i,:])
            ttobsS_dum.append(ttobs_S[i])
            if len(hypo_list) != 0 and types_S[i] == 'e':
                phase_id = int(phase_listS_use.iloc[i, 0])
                if phase_id in hypo_id_to_idx:
                    event_index_i = hypo_id_to_idx[phase_id]
                    start_col = event_index_i * 4
                    for offset, val in enumerate(hypoder_i):
                        if val != 0:
                            hypo_rows.append(successful_row_idx)
                            hypo_cols.append(start_col + offset)
                            hypo_data.append(val)
            successful_row_idx += 1


    ttobs_S = copy(ttobsS_dum)
    phase_listS_use = pd.DataFrame(phase_listS_use_dum).reset_index(drop=True)
    vs_kernel = scsp.vstack(vs_kernel).tocsr()
    if len(hypo_list) != 0:
        hypo_kernel_S = scsp.csr_array((hypo_data, (hypo_rows, hypo_cols)),shape=(successful_row_idx, len(hypo_list) * 4))


    #start adding node
    print('start removing node')
    logger.info('start removing node')
    hit_count = vp_kernel.count_nonzero(axis=0)
    hit_countPawal = vp_kernel.count_nonzero(axis=0)
    hit_countSawal = vs_kernel.count_nonzero(axis=0)
    print('RHC P awal',np.mean(hit_countPawal[hit_countPawal>0]),np.median(hit_countPawal[hit_countPawal>0]),
    np.std(hit_countPawal[hit_countPawal>0]))
    print('jumlah non-zero node awal P', len(hit_countPawal[hit_countPawal>0]))
    logger.info('<--- initial RHC P ---> \n'+'mean: '+str(np.mean(hit_countPawal[hit_countPawal>0]))+' || median: '+
                                                      str(np.median(hit_countPawal[hit_countPawal>0]))+' || deviation: '+
                                                      str(np.std(hit_countPawal[hit_countPawal>0])))
    logger.info('jumlah non-zero node awal P: '+str(len(hit_countPawal[hit_countPawal>0])))
    print('RHC S awal',np.mean(hit_countSawal[hit_countSawal>0]),np.median(hit_countSawal[hit_countSawal>0]),
    np.std(hit_countSawal[hit_countSawal>0]))
    print('jumlah non-zero node awal S', len(hit_countSawal[hit_countSawal > 0]))
    logger.info('<--- initial RHC S ---> \n'+'mean: '+str(np.mean(hit_countSawal[hit_countSawal>0]))+' || median: '+
                                                      str(np.median(hit_countSawal[hit_countSawal>0]))+' || deviation: '+
                                                      str(np.std(hit_countSawal[hit_countSawal>0])))
    logger.info('jumlah non-zero node awal S: '+str(len(hit_countPawal[hit_countPawal>0])))

    node=node[hit_count>0,:]
    vel_node_P=vel_node_P[hit_count>0]
    vel_node_S=vel_node_S[hit_count>0]
    velgridP.change_node(node)
    velgridS.change_node(node)

    vp_kernel=[]
    Ax=[]
    Ay=[]
    Az=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
        results = executor.map(velgridP.safe_crkernel_tensor,path_list_P,chunksize=1)
        for i, result in enumerate(results):
            vp_kernel.append(result[0])
            Ax.append(result[1])
            Ay.append(result[2])
            Az.append(result[3])
    vp_kernel = scsp.vstack(vp_kernel).tocsr()
    Ax = scsp.vstack(Ax).tocsr()
    Ay = scsp.vstack(Ay).tocsr()
    Az = scsp.vstack(Az).tocsr()


    print('start adding node')
    logger.info('start adding node')
    isadd = update_grid
    step_add=1
    while (isadd == True):
        print('add_node_step: '+str(step_add))
        logger.info('add_node_step: '+str(step_add))
        hit_count = vp_kernel.count_nonzero(axis=0)
        node_tetahedron = Delaunay(node)
        class_is_addnode = is_addnode(hit_count, up_threshold, node, deltn, interpP, interpS, nu_cpu=nu_cpu)
        sim=node_tetahedron.simplices
        added_node_list,added_velP_list,added_velS_list = class_is_addnode.add_simultanius(sim)
        if len(added_velP_list)==0:
            isadd=False
            break

        node=np.vstack((node,added_node_list))
        vel_node_P = np.hstack((vel_node_P.T, added_velP_list))
        vel_node_S = np.hstack((vel_node_S.T, added_velS_list))
        velgridP.change_node(node)
        velgridS.change_node(node)

        print('calculating kernel node')
        logger.info('calculating kernel node')
        vp_kernel=[]
        Ax=[]
        Ay=[]
        Az=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
            results = executor.map(velgridP.safe_crkernel_tensor, path_list_P, chunksize=1)
            for i, result in enumerate(results):
                vp_kernel.append(result[0])
                Ax.append(result[1])
                Ay.append(result[2])
                Az.append(result[3])
        vp_kernel = scsp.vstack(vp_kernel).tocsr()
        Ax = scsp.vstack(Ax).tocsr()
        Ay = scsp.vstack(Ay).tocsr()
        Az = scsp.vstack(Az).tocsr()
        step_add += 1


    # start removing node
    print('start removing node by density tensor')
    logger.info('start removing node by density tensor')
    dens=count_tensor_dens(Ax,Ay,Az)
    node=node[dens>=dens_thres,:]
    vel_node_P=vel_node_P[dens>=dens_thres]
    vel_node_S=vel_node_S[dens>=dens_thres]
    velgridP.change_node(node)
    velgridS.change_node(node)
    vp_kernel=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
        results = executor.map(velgridP.safe_crkernel,path_list_P,chunksize=1)
        for i, result in enumerate(results):
            vp_kernel.append(result)
    vp_kernel = scsp.vstack(vp_kernel).tocsr()

    # start removing node
    print('start removing node by RHC')
    logger.info('start removing node by RHC')
    hit_count = vp_kernel.count_nonzero(axis=0)
    node=node[hit_count>low_threshold,:]
    vel_node_P=vel_node_P[hit_count>low_threshold]
    vel_node_S=vel_node_S[hit_count>low_threshold]
    velgridP.change_node(node)
    velgridS.change_node(node)

    vp_kernel=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
        results = executor.map(velgridP.safe_crkernel,path_list_P,chunksize=1)
        for i, result in enumerate(results):
            vp_kernel.append(result)
    vp_kernel = scsp.vstack(vp_kernel).tocsr()
    vs_kernel = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
        results = executor.map(velgridS.safe_crkernel,path_list_S,chunksize=1)
        for i, result in enumerate(results):
            vs_kernel.append(result)
    vs_kernel = scsp.vstack(vs_kernel).tocsr()


    ttobs_P=np.array(ttobs_P)
    ttcal_P=np.array(ttcal_P)
    t_res_P_awal=ttobs_P-ttcal_P
    phase_listP_use['t_res']= abs(t_res_P_awal)

    ttobs_S=np.array(ttobs_S)
    ttcal_S=np.array(ttcal_S)
    t_res_S_awal=ttobs_S-ttcal_S
    phase_listS_use['t_res'] = abs(t_res_S_awal)

    #update_data_use
    vp_kernel=vp_kernel[abs(t_res_P_awal)<r_time_P,:]
    ttcal_P=ttcal_P[abs(t_res_P_awal)<r_time_P]
    if len(hypo_list)!=0:
        hypo_kernel_P=hypo_kernel_P[abs(t_res_P_awal)<r_time_P,:]
    phase_listP_use=phase_listP_use.loc[phase_listP_use['t_res']<r_time_P].reset_index(drop=True)
    t_res_P_awal=t_res_P_awal[abs(t_res_P_awal)<r_time_P]

    vs_kernel=vs_kernel[abs(t_res_S_awal)<r_time_S,:]
    ttcal_S=ttcal_S[abs(t_res_S_awal)<r_time_S]
    if len(hypo_list)!=0:
        hypo_kernel_S=hypo_kernel_S[abs(t_res_S_awal)<r_time_S,:]
    phase_listS_use=phase_listS_use.loc[phase_listS_use['t_res']<r_time_S].reset_index(drop=True)
    t_res_S_awal=t_res_S_awal[abs(t_res_S_awal)<r_time_S]

    print('create DD kernel')
    logger.info('create DD kernel')

    phase_keys = pd.MultiIndex.from_arrays([phase_listP_use.iloc[:, 0].astype(str),phase_listP_use.iloc[:, 1].astype(str)])
    first = ~phase_keys.duplicated(keep="first")
    phase_lookup = pd.Series(np.arange(len(phase_listP_use))[first],index=phase_keys[first])

    ev1 = DDphase_P_use["ev1"].astype(str)
    ev2 = DDphase_P_use["ev2"].astype(str)
    sta = DDphase_P_use["id_sta"].astype(str)

    keys1 = pd.MultiIndex.from_arrays([ev1, sta])
    keys2 = pd.MultiIndex.from_arrays([ev2, sta])

    idx1_all = phase_lookup.reindex(keys1).to_numpy()
    idx2_all = phase_lookup.reindex(keys2).to_numpy()

    source_coords = source_list[["id", "easting", "northing", "depth"]].copy()
    source_coords["id"] = source_coords["id"].astype(str)
    source_coords = (source_coords.drop_duplicates(subset="id", keep="first").set_index("id"))
    coord1 = source_coords.reindex(ev1)[["easting", "northing", "depth"]].to_numpy(dtype=float)
    coord2 = source_coords.reindex(ev2)[["easting", "northing", "depth"]].to_numpy(dtype=float)
    ev_dist = np.linalg.norm(coord1 - coord2, axis=1)
    dt_obs_all = pd.to_numeric(DDphase_P_use["delta_tt"], errors="coerce").to_numpy(dtype=float)

    valid = (np.isfinite(idx1_all)& np.isfinite(idx2_all)& np.isfinite(ev_dist)& (ev_dist <= max_sep)& np.isfinite(dt_obs_all))
    selected = np.flatnonzero(valid)
    idx1 = idx1_all[selected].astype(np.int64)
    idx2 = idx2_all[selected].astype(np.int64)
    ttcal = np.asarray(ttcal_P, dtype=float).ravel()
    dt_cal = ttcal[idx1] - ttcal[idx2]
    dt_obs = dt_obs_all[selected]
    valid_residual = (np.isfinite(dt_cal)& (np.abs(dt_obs - dt_cal) <= r_time_P))
    selected = selected[valid_residual]
    idx1 = idx1[valid_residual]
    idx2 = idx2[valid_residual]

    vp_kernel_DD = (vp_kernel[idx1, :] - vp_kernel[idx2, :]).tocsr()
    if len(hypo_list)!=0:
        hypo_kernel_P_DD = (hypo_kernel_P[idx1, :] - hypo_kernel_P[idx2, :]).tocsr()

    ttcal_DD_P = ttcal[idx1] - ttcal[idx2]
    ttobs_DD_P = dt_obs_all[selected]
    DDphase_P_use = (DDphase_P_use.iloc[selected].reset_index(drop=True))

    phase_keys = pd.MultiIndex.from_arrays([phase_listS_use.iloc[:, 0].astype(str),phase_listS_use.iloc[:, 1].astype(str)])
    first = ~phase_keys.duplicated(keep="first")
    phase_lookup = pd.Series(np.arange(len(phase_listS_use))[first],index=phase_keys[first])

    ev1 = DDphase_S_use["ev1"].astype(str)
    ev2 = DDphase_S_use["ev2"].astype(str)
    sta = DDphase_S_use["id_sta"].astype(str)

    keys1 = pd.MultiIndex.from_arrays([ev1, sta])
    keys2 = pd.MultiIndex.from_arrays([ev2, sta])

    idx1_all = phase_lookup.reindex(keys1).to_numpy()
    idx2_all = phase_lookup.reindex(keys2).to_numpy()

    source_coords = source_list[["id", "easting", "northing", "depth"]].copy()
    source_coords["id"] = source_coords["id"].astype(str)
    source_coords = (source_coords.drop_duplicates(subset="id", keep="first").set_index("id"))
    coord1 = source_coords.reindex(ev1)[["easting", "northing", "depth"]].to_numpy(dtype=float)
    coord2 = source_coords.reindex(ev2)[["easting", "northing", "depth"]].to_numpy(dtype=float)
    ev_dist = np.linalg.norm(coord1 - coord2, axis=1)
    dt_obs_all = pd.to_numeric(DDphase_S_use["delta_tt"], errors="coerce").to_numpy(dtype=float)

    valid = (np.isfinite(idx1_all)& np.isfinite(idx2_all)& np.isfinite(ev_dist)& (ev_dist <= max_sep)& np.isfinite(dt_obs_all))
    selected = np.flatnonzero(valid)
    idx1 = idx1_all[selected].astype(np.int64)
    idx2 = idx2_all[selected].astype(np.int64)
    ttcal = np.asarray(ttcal_S, dtype=float).ravel()
    dt_cal = ttcal[idx1] - ttcal[idx2]
    dt_obs = dt_obs_all[selected]
    valid_residual = (np.isfinite(dt_cal)& (np.abs(dt_obs - dt_cal) <= r_time_S))
    selected = selected[valid_residual]
    idx1 = idx1[valid_residual]
    idx2 = idx2[valid_residual]

    vs_kernel_DD = (vs_kernel[idx1, :] - vs_kernel[idx2, :]).tocsr()
    if len(hypo_list)!=0:
        hypo_kernel_S_DD = (hypo_kernel_S[idx1, :] - hypo_kernel_S[idx2, :]).tocsr()

    ttcal_DD_S = ttcal[idx1] - ttcal[idx2]
    ttobs_DD_S = dt_obs_all[selected]
    DDphase_S_use = (DDphase_S_use.iloc[selected].reset_index(drop=True))

    t_resDD_P_awal=ttobs_DD_P-ttcal_DD_P
    t_resDD_S_awal=ttobs_DD_S-ttcal_DD_S

    t_res_awal=np.hstack((t_res_P_awal,t_resDD_P_awal*weightDD,t_res_S_awal,t_resDD_S_awal*weightDD))
    t_res_awal_noweight=np.hstack((t_res_P_awal,t_resDD_P_awal,t_res_S_awal,t_resDD_S_awal))
    rms1 = np.sqrt(np.mean(t_res_awal_noweight ** 2))

    vp_kernel_awal=vp_kernel.copy()
    vs_kernel_awal=vs_kernel.copy()
    vp_kernel_DD_awal=vp_kernel_DD.copy()
    vs_kernel_DD_awal=vs_kernel_DD.copy()
    if len(hypo_list)!=0:
        hypo_kernel_P_awal=hypo_kernel_P.copy()
        hypo_kernel_S_awal=hypo_kernel_S.copy()
        hypo_kernel_P_DD_awal=hypo_kernel_P_DD.copy()
        hypo_kernel_S_DD_awal=hypo_kernel_S_DD.copy()

    phase_listP_use_awal=phase_listP_use.copy()
    phase_listS_use_awal=phase_listS_use.copy()
    DDphase_P_use_awal=DDphase_P_use.copy()
    DDphase_S_use_awal=DDphase_S_use.copy()

    rms_list=[rms1]
    dv_var=[]
    ds_var=[]
    dis_var=[]
    dt_var=[]
    rnorm_list=[]
    xnorm_list=[]
    CND_list=[]


    print('initial rms:', rms1)
    logger.info('initial rms: ' + str(rms1))


    #mulai iterasi:
    for iter in range(0, len(damping_list)):
        source_list_invers = source_list.copy()
        source_list_invers['to_update'] = 0
        source_list_invers['to_update_i'] = 0
        print('start iteration:', iter)
        logger.info('start iteration: ' + str(iter))
        weight = np.sqrt(vor_volumes(node))
        weight[weight == 0] = np.max(weight)
        weight_inv = scsp.diags(1.0 / weight,format="csr")

        vp_kernel=vp_kernel_awal@weight_inv
        vs_kernel=vs_kernel_awal@weight_inv
        Dp = damping_list.iloc[iter,1] * weight_inv
        Ds = damping_list.iloc[iter,2] * weight_inv

        vp_kernel_DD=weightDD*(vp_kernel_DD_awal@weight_inv)
        vs_kernel_DD=weightDD*(vs_kernel_DD_awal@weight_inv)
        L = scsp.csr_array(smooth_matrix(node))
        L_smooth = damping_list.iloc[iter,3] * (L @ weight_inv)

        if len(hypo_list)!=0:
            n_h=len(hypo_list)*4
            hypo_kernel_P_DD=weightDD*(hypo_kernel_P_DD_awal)
            hypo_kernel_S_DD=weightDD*(hypo_kernel_S_DD_awal)
            G_data = scsp.block_array([
        [hypo_kernel_P_awal,    vp_kernel,       None],
        [hypo_kernel_P_DD, vp_kernel_DD,    None],
        [hypo_kernel_S_awal,    None,     vs_kernel],
        [hypo_kernel_S_DD, None,     vs_kernel_DD]], format="csr")
            Dh = damping_list.iloc[iter,0]* scsp.eye(n_h, format="csr")
            G_regularization = scsp.block_array([
        [Dh,   None,     None],
        [None, Dp,       None],
        [None, None,     Ds],
        [None, L_smooth, None],
        [None, None,     L_smooth]], format="csr")    
        else:
            G_data = scsp.block_array([
        [vp_kernel,    None],
        [vp_kernel_DD, None],
        [None,  vs_kernel],
        [None,  vs_kernel_DD]], format="csr")    
            G_regularization = scsp.block_array([
        [Dp,       None],
        [None,     Ds],
        [L_smooth, None],
        [None,     L_smooth]], format="csr")

        
        inv_matrix_hypo = scsp.vstack([G_data,G_regularization], format="csr")
        t_res_inv = np.concatenate([np.asarray(t_res_awal, dtype=float).ravel(),np.zeros(G_regularization.shape[0])])
        ## add resolution matrix
        #i_damp=np.identity(inv_matrix_hypo.shape[1])*damping_1
        #G_full=np.vstack((inv_matrix_hypo,i_damp))
        #G_inv = np.linalg.pinv(G_full)
        #res_matrix=G_inv@G_full
        #diag_res=np.trace(res_matrix)/inv_matrix_hypo.shape[1]
        # print(diag_res)
        # logger.info('diagonal resolution matrix: ' + str(diag_res))
        ## resolution matrix end here
        col_norm = np.sqrt(np.asarray(inv_matrix_hypo.power(2).sum(axis=0)).ravel())
        D_inv = scsp.diags(1.0 / col_norm, format="csr")
        A_norm = (inv_matrix_hypo @ D_inv).tocsr()
        inversion_result = lsmr(A_norm, t_res_inv)
        dm_result = inversion_result[0] / col_norm

        if len(hypo_list)!=0:
            ds_P = dm_result[hypo_kernel_P.shape[1]:hypo_kernel_P.shape[1]+vp_kernel.shape[1]]*weight_inv.diagonal()
            ds_S = dm_result[hypo_kernel_P.shape[1]+vp_kernel.shape[1]:]*weight_inv.diagonal()
        else:
            ds_P = dm_result[:vp_kernel.shape[1]]*weight_inv.diagonal()
            ds_S = dm_result[vp_kernel.shape[1]:]*weight_inv.diagonal()

        #acond=inversion_result[6]
        print("CND: "+str(inversion_result[6]))
        print("istop: "+str(inversion_result[1]))
        rnorm_list.append(inversion_result[3])
        xnorm_list.append(inversion_result[7])
        CND_list.append(inversion_result[6])
        logger.info('CND value: ' + str(inversion_result[6]))
        logger.info('istop: ' + str(inversion_result[1]))
        logger.info('rnorm: ' + str(inversion_result[3]))
        logger.info('xnorm: ' + str(inversion_result[7]))
        logger.info('no_ite: ' + str(inversion_result[2]))

        ds_var_i = np.sqrt((np.mean(np.hstack((ds_P,ds_S)) ** 2)))
        ds_var.append(ds_var_i)

        #updateVp
        vel_awal_P = copy(vel_node_P)
        vawal_P = copy(vel_node_P)
        dsawal_P = 1 / vawal_P
        dsakhir_P = ds_P + dsawal_P
        vel_node_P = 1 / dsakhir_P

        #updateVs
        vel_awal_S = copy(vel_node_S)
        vawal_S = copy(vel_node_S)
        dsawal_S = 1 / vawal_S
        dsakhir_S = ds_S + dsawal_S
        vel_node_S = 1 / dsakhir_S

        dv_P=vel_node_P-vel_awal_P
        dv_S=vel_node_S-vel_awal_S
        dv_var_i = np.sqrt((np.mean(np.hstack((dv_P,dv_S)) ** 2)))
        dv_var.append(dv_var_i)

        print('velocity variation in parameter ', iter, ' :', dv_var_i)
        logger.info('velocity variant in parameter '+ str(iter)+' :'+str(dv_var_i))
        #update_hypo
        if len(hypo_list)!=0:
            d_hypo = dm_result[:(len(hypo_list) * 4)].reshape(len(hypo_list), 4)
            cols = ['to_update', 'easting', 'northing', 'elevation']
            hypo_list[cols] = (hypo_list[cols].to_numpy(dtype=float) + d_hypo)
            hypo_list['to_update_i'] = d_hypo[:, 0]
            source_list_invers=pd.concat([hypo_list, blast_list], ignore_index=True)
        else:
            source_list_invers=blast_list
        #update phase use (delete outside treshold)
        paths_P = []
        ttobs_P = []
        phase_listP_use_dum = []
        types_P=[]
        for i in range(len(phase_listP_use_awal)):
            source = np.array(source_list_invers[source_list_invers['id'] == phase_listP_use_awal.iloc[i, 0]][['easting', 'northing', 'elevation']])
            receiver = np.array(receiver_list[receiver_list['id'] == phase_listP_use_awal.iloc[i, 1]][['easting', 'northing', 'elevation']])
            t_update =(source_list_invers[source_list_invers['id'] == phase_listP_use_awal.iloc[i, 0]][['to_update']])
            if (source.size == 0) or (receiver.size == 0):
                continue
            if (float(phase_listP_use_awal.iloc[i,2])) <= 0:
                continue
            phase_listP_use_dum.append(phase_listP_use_awal.iloc[i, :])
            ttobs_P.append(float(phase_listP_use_awal.iloc[i,2]) - float(t_update.iloc[0, 0]))
            path = np.vstack((source, receiver))
            paths_P.append(path)
            types_P.append(str((source_list_invers[source_list_invers['id'] == phase_listP_use_awal.iloc[i, 0]]['type']).item()))

        phase_listP_use = pd.DataFrame(phase_listP_use_dum).reset_index(drop=True)

        paths_S= []
        ttobs_S = []
        phase_listS_use_dum = []
        types_S=[]
        for i in range(len(phase_listS_use_awal)):
            source = np.array(source_list_invers[source_list_invers['id'] == phase_listS_use_awal.iloc[i, 0]][['easting', 'northing', 'elevation']])
            receiver = np.array(receiver_list[receiver_list['id'] == phase_listS_use_awal.iloc[i, 1]][['easting', 'northing', 'elevation']])
            t_update = (source_list_invers[source_list_invers['id'] == phase_listS_use_awal.iloc[i, 0]][['to_update']])
            if (source.size == 0) or (receiver.size == 0):
                continue
            if (float(phase_listS_use_awal.iloc[i,2])) <= 0:
                continue
            phase_listS_use_dum.append(phase_listS_use_awal.iloc[i, :])
            ttobs_S.append(float(phase_listS_use_awal.iloc[i,2]) - float(t_update.iloc[0, 0]))
            path = np.vstack((source, receiver))
            paths_S.append(path)
            types_S.append(str((source_list_invers[source_list_invers['id'] == phase_listS_use_awal.iloc[i, 0]]['type']).item()))

        phase_listS_use = pd.DataFrame(phase_listS_use_dum).reset_index(drop=True)

        node_all = np.vstack((node, node_outer))
        vel_allP = np.hstack((vel_node_P, modvel_outer.Vp))
        interpP = RBFInterpolator(node_all, vel_allP, kernel='linear', neighbors=8)
        gridVp = (np.round(interpP(np.column_stack((Xinter.flatten(), Yinter.flatten(), Zinter.flatten()))),
                           decimals=4).reshape(len(xnode), len(ynode), len(znode))).astype(np.float32)
        velgridP = VelocityGrid(node, xnode, ynode, znode, gridVp, deltn, delt, xfac, iter1, iter2, tmin, if_art)

        vel_allS = np.hstack((vel_node_S, modvel_outer.Vs))
        interpS = RBFInterpolator(node_all, vel_allS, kernel='linear', neighbors=8)
        gridVs = (np.round(interpS(np.column_stack((Xinter.flatten(), Yinter.flatten(), Zinter.flatten()))),
                           decimals=4).reshape(len(xnode), len(ynode), len(znode))).astype(np.float32)
        velgridS = VelocityGrid(node, xnode, ynode, znode, gridVs, deltn, delt, xfac, iter1, iter2, tmin, if_art)

        print('start forward after iteration: ',iter)
        logger.info('start forward after iteration: ' + str(iter))

        # P Phase forward
        ttcal_P = []
        path_list_P = []
        phase_listP_use_dum = []
        ttobsP_dum = []
        if len(hypo_list) != 0:
            hypo_id_to_idx = {int(val): idx for idx, val in enumerate(hypo_list.iloc[:, 0])}
        vp_kernel=[]
        hypo_rows, hypo_cols, hypo_data = [], [], []
        successful_row_idx = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
            results=executor.map(velgridP.safe_forwardtwopoints,paths_P,chunksize=1)
            for i, result in enumerate(results):
                if isinstance(result, dict) and "error" in result:
                    logger.info('not convergen in path P: ' + ' '.join(phase_listP_use.iloc[i, :].astype(str)))
                    logger.info(result['type'])
                    logger.info(result['error'])
                    logger.info(result['traceback'])
                    continue

                ttcal_i = result[1]
                hypoder_i = result[3]
                path_list_P.append(result[0])
                vp_kernel.append(result[2])
                ttcal_P.append(float(ttcal_i))
                phase_listP_use_dum.append(phase_listP_use.iloc[i, :])
                ttobsP_dum.append(ttobs_P[i])
                if len(hypo_list) != 0 and types_P[i] == 'e':
                    phase_id = int(phase_listP_use.iloc[i, 0])
                    if phase_id in hypo_id_to_idx:
                        event_index_i = hypo_id_to_idx[phase_id]
                        start_col = event_index_i * 4
                        for offset, val in enumerate(hypoder_i):
                            if val != 0:
                                hypo_rows.append(successful_row_idx)
                                hypo_cols.append(start_col + offset)
                                hypo_data.append(val)
                successful_row_idx += 1

        ttobs_P=copy(ttobsP_dum)
        phase_listP_use = pd.DataFrame(phase_listP_use_dum).reset_index(drop=True)
        vp_kernel = scsp.vstack(vp_kernel).tocsr()
        if len(hypo_list) != 0:
            hypo_kernel_P = scsp.csr_array((hypo_data, (hypo_rows, hypo_cols)),shape=(successful_row_idx, len(hypo_list) * 4))

        #forward S
        ttcal_S = []
        path_list_S = []
        phase_listS_use_dum = []
        ttobsS_dum = []
        if len(hypo_list) != 0:
            hypo_id_to_idx = {int(val): idx for idx, val in enumerate(hypo_list.iloc[:, 0])}
        vs_kernel=[]
        hypo_rows, hypo_cols, hypo_data = [], [], []
        successful_row_idx = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
            results = executor.map(velgridS.safe_forwardtwopoints, paths_S, chunksize=1)
            for i, result in enumerate(results):
                if isinstance(result, dict) and "error" in result:
                    logger.info('not convergen in path S: ' + ' '.join(phase_listS_use.iloc[i, :].astype(str)))
                    logger.info(result['type'])
                    logger.info(result['error'])
                    logger.info(result['traceback'])
                    continue
                ttcal_i = result[1]
                hypoder_i = result[3]
                path_list_S.append(result[0])
                vs_kernel.append(result[2])
                ttcal_S.append(float(ttcal_i))
                phase_listS_use_dum.append(phase_listS_use.iloc[i, :])
                ttobsS_dum.append(ttobs_S[i])
                if len(hypo_list) != 0 and types_S[i] == 'e':
                    phase_id = int(phase_listS_use.iloc[i, 0])
                    if phase_id in hypo_id_to_idx:
                        event_index_i = hypo_id_to_idx[phase_id]
                        start_col = event_index_i * 4
                        for offset, val in enumerate(hypoder_i):
                            if val != 0:
                                hypo_rows.append(successful_row_idx)
                                hypo_cols.append(start_col + offset)
                                hypo_data.append(val)
                successful_row_idx += 1


        ttobs_S=copy(ttobsS_dum)
        phase_listS_use = pd.DataFrame(phase_listS_use_dum).reset_index(drop=True)
        vs_kernel = scsp.vstack(vs_kernel).tocsr()
        if len(hypo_list) != 0:
            hypo_kernel_S = scsp.csr_array((hypo_data, (hypo_rows, hypo_cols)),shape=(successful_row_idx, len(hypo_list) * 4))



        ttobs_P = np.array(ttobs_P).flatten()
        ttcal_P = np.array(ttcal_P)
        t_res_P = ttobs_P - ttcal_P

        ttobs_S = np.array(ttobs_S).flatten()
        ttcal_S = np.array(ttcal_S)
        t_res_S = ttobs_S - ttcal_S


        print('create DD kernel')
        logger.info('create DD kernel')
        
        phase_keys = pd.MultiIndex.from_arrays([phase_listP_use.iloc[:, 0].astype(str),phase_listP_use.iloc[:, 1].astype(str)])
        first = ~phase_keys.duplicated(keep="first")
        phase_lookup = pd.Series(np.arange(len(phase_listP_use))[first],index=phase_keys[first])

        ev1 = DDphase_P_use_awal["ev1"].astype(str)
        ev2 = DDphase_P_use_awal["ev2"].astype(str)
        sta = DDphase_P_use_awal["id_sta"].astype(str)

        keys1 = pd.MultiIndex.from_arrays([ev1, sta])
        keys2 = pd.MultiIndex.from_arrays([ev2, sta])

        idx1_all = phase_lookup.reindex(keys1).to_numpy()
        idx2_all = phase_lookup.reindex(keys2).to_numpy()

        source_coords = source_list[["id", "easting", "northing", "depth"]].copy()
        source_coords["id"] = source_coords["id"].astype(str)
        source_coords = (source_coords.drop_duplicates(subset="id", keep="first").set_index("id"))
        coord1 = source_coords.reindex(ev1)[["easting", "northing", "depth"]].to_numpy(dtype=float)
        coord2 = source_coords.reindex(ev2)[["easting", "northing", "depth"]].to_numpy(dtype=float)
        ev_dist = np.linalg.norm(coord1 - coord2, axis=1)
        dt_obs_all = update_ttobs_DD(DDphase_P_use_awal,source_list_invers)

        valid = (np.isfinite(idx1_all)& np.isfinite(idx2_all)& np.isfinite(ev_dist)& (ev_dist <= max_sep)& np.isfinite(dt_obs_all))
        selected = np.flatnonzero(valid)
        idx1 = idx1_all[selected].astype(np.int64)
        idx2 = idx2_all[selected].astype(np.int64)
        ttcal = np.asarray(ttcal_P, dtype=float).ravel()
        dt_cal = ttcal[idx1] - ttcal[idx2]
        dt_obs = dt_obs_all[selected]
        valid_residual = (np.isfinite(dt_cal)& (np.abs(dt_obs - dt_cal) <= r_time_P))
        selected = selected[valid_residual]
        idx1 = idx1[valid_residual]
        idx2 = idx2[valid_residual]

        vp_kernel_DD = (vp_kernel[idx1, :] - vp_kernel[idx2, :]).tocsr()
        if len(hypo_list)!=0:
            hypo_kernel_P_DD = (hypo_kernel_P[idx1, :] - hypo_kernel_P[idx2, :]).tocsr()

        ttcal_DD_P = ttcal[idx1] - ttcal[idx2]
        ttobs_DD_P = dt_obs_all[selected]
        DDphase_P_use = (DDphase_P_use_awal.iloc[selected].reset_index(drop=True))

        phase_keys = pd.MultiIndex.from_arrays([phase_listS_use.iloc[:, 0].astype(str),phase_listS_use.iloc[:, 1].astype(str)])
        first = ~phase_keys.duplicated(keep="first")
        phase_lookup = pd.Series(np.arange(len(phase_listS_use))[first],index=phase_keys[first])

        ev1 = DDphase_S_use_awal["ev1"].astype(str)
        ev2 = DDphase_S_use_awal["ev2"].astype(str)
        sta = DDphase_S_use_awal["id_sta"].astype(str)

        keys1 = pd.MultiIndex.from_arrays([ev1, sta])
        keys2 = pd.MultiIndex.from_arrays([ev2, sta])

        idx1_all = phase_lookup.reindex(keys1).to_numpy()
        idx2_all = phase_lookup.reindex(keys2).to_numpy()

        source_coords = source_list[["id", "easting", "northing", "depth"]].copy()
        source_coords["id"] = source_coords["id"].astype(str)
        source_coords = (source_coords.drop_duplicates(subset="id", keep="first").set_index("id"))
        coord1 = source_coords.reindex(ev1)[["easting", "northing", "depth"]].to_numpy(dtype=float)
        coord2 = source_coords.reindex(ev2)[["easting", "northing", "depth"]].to_numpy(dtype=float)
        ev_dist = np.linalg.norm(coord1 - coord2, axis=1)
        dt_obs_all = update_ttobs_DD(DDphase_S_use_awal, source_list_invers)

        valid = (np.isfinite(idx1_all)& np.isfinite(idx2_all)& np.isfinite(ev_dist)& (ev_dist <= max_sep)& np.isfinite(dt_obs_all))
        selected = np.flatnonzero(valid)
        idx1 = idx1_all[selected].astype(np.int64)
        idx2 = idx2_all[selected].astype(np.int64)
        ttcal = np.asarray(ttcal_S, dtype=float).ravel()
        dt_cal = ttcal[idx1] - ttcal[idx2]
        dt_obs = dt_obs_all[selected]
        valid_residual = (np.isfinite(dt_cal)& (np.abs(dt_obs - dt_cal) <= r_time_S))
        selected = selected[valid_residual]
        idx1 = idx1[valid_residual]
        idx2 = idx2[valid_residual]

        vs_kernel_DD = (vs_kernel[idx1, :] - vs_kernel[idx2, :]).tocsr()
        if len(hypo_list) != 0:
            hypo_kernel_S_DD = (hypo_kernel_S[idx1, :] - hypo_kernel_S[idx2, :]).tocsr()

        ttcal_DD_S = ttcal[idx1] - ttcal[idx2]
        ttobs_DD_S = dt_obs_all[selected]
        DDphase_S_use = (DDphase_S_use_awal.iloc[selected].reset_index(drop=True))

        t_resDD_P = ttobs_DD_P - ttcal_DD_P
        t_resDD_S = ttobs_DD_S - ttcal_DD_S

        t_res = np.hstack((t_res_P, t_resDD_P*weightDD, t_res_S, t_resDD_S*weightDD))
        t_res_noweight = np.hstack((t_res_P, t_resDD_P, t_res_S, t_resDD_S))
        rms2 = np.sqrt(np.mean(t_res_noweight ** 2))

        dt_var.append(rms2)
        print('rms in parameter ', iter, ' :', rms2)
        logger.info('rms in parameter '+ str(iter)+ ' :'+ str(rms2))


    damping_list['t_res']=np.array(dt_var)
    damping_list['slow_variance'] = np.array(ds_var)
    if len(hypo_list)!=0:
        damping_list['disp_variance'] = np.array(dis_var)
    else:
        damping_list['disp_variance'] = np.zeros(len(ds_var))
    damping_list['vel_variance']=np.array(dv_var)
    damping_list['rnorm']=np.array(rnorm_list)
    damping_list['xnorm']=np.array(xnorm_list)
    damping_list['CND']=np.array(CND_list)
    damping_list.to_csv(folder_name+'/param_opt.csv', index=False)
    #plot figure1
    labels = (damping_list['damp_hypo'].astype(str)+ ','+ damping_list['damp_p'].astype(str)+
              ','+damping_list['damp_s'].astype(str)+ ','+ damping_list['damp_smooth'].astype(str))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(damping_list['rnorm'],damping_list['xnorm'])
    for x, y, label in zip(damping_list['rnorm'],damping_list['xnorm'],labels):
        ax.annotate(label,(x, y),xytext=(5, 5),textcoords='offset points',fontsize=8)

    ax.set_xlabel('rnorm')
    ax.set_ylabel('xnorm')
    ax.set_title('trade-off curve')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig1_path = os.path.join(folder_name, 'fig1.png')
    fig.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # plot figure2
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(damping_list['t_res'],damping_list['slow_variance'])
    for x, y, label in zip(damping_list['t_res'],damping_list['slow_variance'],labels):
        ax.annotate(label,(x, y),xytext=(5, 5),textcoords='offset points',fontsize=8
        )
    ax.set_xlabel('t_res')
    ax.set_ylabel('slow_variance')
    ax.set_title('trade-off curve')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig2_path = os.path.join(folder_name, 'fig2.png')
    fig.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # plot figure3
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(damping_list['t_res'],damping_list['disp_variance'])
    for x, y, label in zip(damping_list['t_res'],damping_list['disp_variance'],labels):
        ax.annotate(label,(x, y),xytext=(5, 5),textcoords='offset points',fontsize=8
        )
    ax.set_xlabel('t_res')
    ax.set_ylabel('disp_variance')
    ax.set_title('trade-off curve')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig3_path = os.path.join(folder_name, 'fig3.png')
    fig.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    with ZipFile(folder_name + '/compress.zip', 'w') as zf:
        zf.write(folder_name + '/param_optDD.csv')
        zf.write(folder_name + '/fig1.png')
        zf.write(folder_name + '/fig2.png')
        zf.write(folder_name + '/fig3.png')
    print('process is done')
    logger.info('process is done')
