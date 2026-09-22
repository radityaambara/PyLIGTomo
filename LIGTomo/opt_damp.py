from .all_func import VelocityGrid, vor_volumes, smooth_matrix, is_addnode
from copy import copy
from scipy.sparse.linalg import lsmr
import scipy.sparse as scsp
import concurrent.futures
from scipy.spatial import Delaunay
from scipy.interpolate import RBFInterpolator
import pandas as pd
import numpy as np
from zipfile import ZipFile
import logging
import os
def run_opt_param(modvel, modvel_outer, source_list, receiver_list, phase_list, damping_list, delt, deltn, xfac, iter1, iter2, tmin,
                 up_threshold, low_threshold, r_time_P, r_time_S,
                 update_grid,folder_name,nu_cpu=os.cpu_count()-1):
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
    print('run optimization damping parameter')
    logger.info('run optimization damping parameter')
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
    r_time_P = float(r_time_P)
    r_time_S = float(r_time_S)
    damping_list = damping_list.apply(pd.to_numeric)
    update_grid = bool(update_grid)
    logger.info("\n delt: "+str(delt)+
                "\n deltn: "+str(deltn)+
                "\n xfac: "+str(xfac)+
                "\n iter1: "+str(iter1)+
                "\n iter2: "+str(iter2)+
                "\n tmin: "+str(tmin)+
                "\n up_threshold: "+str(up_threshold)+
                "\n low_threshold: "+str(low_threshold)+
                "\n r_time_P: "+str(r_time_P)+
                "\n r_time_S: "+str(r_time_S)+
                "\n update_grid: "+str(update_grid))

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
    types_P = []
    paths_S=[]
    ttobs_S=[]
    types_S = []
    hypo_list=source_list[source_list['type']=='e']
    hypo_list['to_update']=0
    hypo_list['to_update_i']=0
    blast_list=source_list[source_list['type']=='b']
    blast_list['to_update']=0
    blast_list['to_update_i']=0


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
        types_P.append(str((source_list[source_list['id'] == phase_listP.iloc[i, 0]]['type']).item()))
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
        types_S.append(str((source_list[source_list['id'] == phase_listS.iloc[i, 0]]['type']).item()))
    phase_listS_use = pd.DataFrame(phase_listS_use)


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
            if len(hypo_list) != 0 and types_P[i] == 0:
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
    phase_listP_use = pd.DataFrame(phase_listP_use_dum)
    vp_kernel = scsp.vstack(vp_kernel).tocsr()
    if len(hypo_list) != 0:
        hypo_kernel_P = scsp.csr_array((hypo_data, (hypo_rows, hypo_cols)),shape=(successful_row_idx, len(hypo_list) * 4))


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
            if len(hypo_list) != 0 and types_S[i] == 0:
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
    phase_listS_use = pd.DataFrame(phase_listS_use_dum)
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
    print('RHC S awal',np.mean(hit_countSawal[hit_countSawal>0]),np.median(hit_countSawal[hit_countSawal>0]),
    np.std(hit_countSawal[hit_countSawal>0]))
    print('jumlah non-zero node awal S', len(hit_countSawal[hit_countSawal > 0]))
    logger.info('<--- initial RHC P ---> \n'+'mean: '+str(np.mean(hit_countPawal[hit_countPawal>0]))+' || median: '+
                                                      str(np.median(hit_countPawal[hit_countPawal>0]))+' || deviation: '+
                                                      str(np.std(hit_countPawal[hit_countPawal>0])))
    logger.info('jumlah non-zero node awal P: '+str(len(hit_countPawal[hit_countPawal>0])))
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
    successful_rows_P = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
        results = executor.map(velgridP.safe_crkernel,path_list_P,chunksize=1)
        for i, result in enumerate(results):
            vp_kernel.append(result)
            successful_rows_P += 1
    vp_kernel = scsp.vstack(vp_kernel).tocsr()



    print('start adding node')
    logger.info('start adding node')
    isadd = update_grid
    step_add = 1
    while (isadd == True):
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
        successful_rows_P = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
            results = executor.map(velgridP.safe_crkernel, path_list_P, chunksize=1)
            for i, result in enumerate(results):
                vp_kernel.append(result)
                successful_rows_P += 1
        vp_kernel = scsp.vstack(vp_kernel).tocsr()
        step_add += 1

    # start removing node
    print('start removing node')
    logger.info('start removing node')
    hit_count = vp_kernel.count_nonzero(axis=0)
    node=node[hit_count>low_threshold,:]
    vel_node_P=vel_node_P[hit_count>low_threshold]
    vel_node_S=vel_node_S[hit_count>low_threshold]
    velgridP.change_node(node)
    velgridS.change_node(node)

    vp_kernel=[]
    successful_rows_P = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
        results = executor.map(velgridP.safe_crkernel,path_list_P,chunksize=1)
        for i, result in enumerate(results):
            vp_kernel.append(result)
            successful_rows_P += 1
    vp_kernel = scsp.vstack(vp_kernel).tocsr()
    vs_kernel = []
    successful_rows_S = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
        results = executor.map(velgridS.safe_crkernel,path_list_S,chunksize=1)
        for i, result in enumerate(results):
            vs_kernel.append(result)
            successful_rows_S += 1
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
    if len(hypo_list)!=0:
        hypo_kernel_P=hypo_kernel_P[abs(t_res_P_awal)<r_time_P,:]
    phase_listP_use=phase_listP_use.loc[phase_listP_use['t_res']<r_time_P]
    t_res_P_awal=t_res_P_awal[abs(t_res_P_awal)<r_time_P]

    vs_kernel=vs_kernel[abs(t_res_S_awal)<r_time_S,:]
    if len(hypo_list)!=0:
        hypo_kernel_S=hypo_kernel_S[abs(t_res_S_awal)<r_time_S,:]
    phase_listS_use=phase_listS_use.loc[phase_listS_use['t_res']<r_time_S]
    t_res_S_awal=t_res_S_awal[abs(t_res_S_awal)<r_time_S]

    t_res_awal=np.hstack((t_res_P_awal,t_res_S_awal))
    rms1 = np.sqrt(np.mean(t_res_awal ** 2))

    rms_list=[]
    rms_list.append(rms1)
    dv_var=[]
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
        weight = np.sqrt(vor_volumes(node))
        weight[weight == 0] = np.max(weight)
        vp_kernel_inv = vp_kernel[:, weight != 0]
        vs_kernel_inv = vs_kernel[:, weight != 0]
        vp_kernel_zeros = np.zeros((vs_kernel_inv.shape[0],vp_kernel_inv.shape[1]))
        vs_kernel_zeros = np.zeros((vp_kernel_inv.shape[0],vs_kernel_inv.shape[1]))

        weight_inv = scsp.diags(1 / (weight[weight != 0]))
        smooth_damp=scsp.csr_array(smooth_matrix(node[weight != 0,:]))*damping_list.iloc[iter,1]
        zeros_smooth=scsp.csr_array(smooth_damp.shape)

        v_stack_P = scsp.vstack([vp_kernel_inv, vp_kernel_zeros, smooth_damp, zeros_smooth])
        v_stack_S = scsp.vstack([vs_kernel_zeros, vs_kernel_inv, zeros_smooth, smooth_damp])
        inv_matrix_P = v_stack_P.dot(weight_inv)
        inv_matrix_S = v_stack_S.dot(weight_inv)

        if len(hypo_list)!=0:
            hypo_zero=scsp.csr_array((smooth_damp.shape[0]*2, hypo_kernel_P.shape[1]))
            hypo_kernel_inv=scsp.vstack((hypo_kernel_P,hypo_kernel_S,hypo_zero))
            inv_matrix_hypo = scsp.hstack((hypo_kernel_inv, inv_matrix_P, inv_matrix_S))
        else:
            inv_matrix_hypo=scsp.hstack((inv_matrix_P, inv_matrix_S))
        t_smooth = np.zeros((smooth_damp.shape[0] * 2))
        t_res_inv = np.hstack((t_res_awal, t_smooth))
        inversion_result = lsmr(inv_matrix_hypo, t_res_inv, damp=damping_list.iloc[iter,0])

        if len(hypo_list)!=0:
            ds_P=inversion_result[0][hypo_kernel_P.shape[1]:hypo_kernel_P.shape[1]+vp_kernel_inv.shape[1]]*weight_inv.diagonal()
            ds_S=inversion_result[0][hypo_kernel_P.shape[1]+vp_kernel_inv.shape[1]:]*weight_inv.diagonal()
        else:
            ds_P = inversion_result[0][:vp_kernel_inv.shape[1]]*weight_inv.diagonal()
            ds_S = inversion_result[0][vp_kernel_inv.shape[1]:]*weight_inv.diagonal()
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

        #updateVp
        vel_awal_P = copy(vel_node_P)
        vel_node_P_akhir = copy(vel_node_P)
        vawal_P = copy(vel_node_P[weight!=0])
        dsawal_P = 1 / vawal_P
        dsakhir_P = ds_P + dsawal_P
        vel_node_P_akhir[weight!=0] = 1 / dsakhir_P

        #updateVs
        vel_awal_S = copy(vel_node_S)
        vel_node_S_akhir = copy(vel_node_S)
        vawal_S = copy(vel_node_S[weight!=0])
        dsawal_S = 1 / vawal_S
        dsakhir_S = ds_S + dsawal_S
        vel_node_S_akhir[weight!=0] = 1 / dsakhir_S

        dv_P=vel_node_P_akhir-vel_awal_P
        dv_S=vel_node_S_akhir-vel_awal_S
        dv_var_i = np.sqrt((np.mean(np.hstack((dv_P,dv_S)) ** 2)))
        dv_var.append(dv_var_i)

        print('velocity variation in parameter ', iter, ' :', dv_var_i)
        logger.info('velocity variant in parameter '+ str(iter)+' :'+str(dv_var_i))
        #update_hypo
        if len(hypo_list)!=0:
            for i in range(0,len(hypo_list)):
                hypo_list.loc[i,'to_update'] += inversion_result[0][i*4]
                hypo_list.loc[i,'easting'] += inversion_result[0][i*4+1]
                hypo_list.loc[i,'northing'] += inversion_result[0][i*4+2]
                hypo_list.loc[i,'elevation'] += inversion_result[0][i*4+3]
                hypo_list.loc[i,'to_update_i'] = inversion_result[0][i * 4]
            source_list_invers=pd.concat([hypo_list, blast_list], ignore_index=True)
        else:
            source_list_invers=blast_list

        #update phase use (hapus diluar treshold)
        paths_P = []
        ttobs_P = []
        phase_listP_use_dum = []
        types_P = []
        for i in range(len(phase_listP_use)):
            source = np.array(
                source_list_invers[source_list_invers['id'] == phase_listP_use.iloc[i, 0]][['easting', 'northing', 'elevation']])
            receiver = np.array(
                receiver_list[receiver_list['id'] == phase_listP_use.iloc[i, 1]][['easting', 'northing', 'elevation']])
            t_update =(source_list_invers[source_list_invers['id'] == phase_listP_use.iloc[i, 0]][['to_update']])
            if (source.size == 0) or (receiver.size == 0):
                continue
            if (float(phase_listP_use.iloc[i,2])) <= 0:
                continue
            phase_listP_use_dum.append(phase_listP_use.iloc[i, :])
            ttobs_P.append(float(phase_listP_use.iloc[i,2]) - float(t_update.to_numpy().flatten()))
            path = np.vstack((source, receiver))
            paths_P.append(path)
            types_P.append(str((source_list_invers[source_list_invers['id'] == phase_listP_use.iloc[i, 0]]['type']).item()))
        phase_listP_use = pd.DataFrame(phase_listP_use_dum)

        paths_S= []
        ttobs_S = []
        phase_listS_use_dum = []
        types_S = []
        for i in range(len(phase_listS_use)):
            source = np.array(
                source_list_invers[source_list_invers['id'] == phase_listS_use.iloc[i, 0]][['easting', 'northing', 'elevation']])
            receiver = np.array(
                receiver_list[receiver_list['id'] == phase_listS_use.iloc[i, 1]][['easting', 'northing', 'elevation']])
            t_update = (source_list_invers[source_list_invers['id'] == phase_listS_use.iloc[i, 0]][['to_update']])
            if (source.size == 0) or (receiver.size == 0):
                continue
            if (float(phase_listS_use.iloc[i,2])) <= 0:
                continue
            phase_listS_use_dum.append(phase_listS_use.iloc[i, :])
            ttobs_S.append(float(phase_listS_use.iloc[i,2]) - float(t_update.to_numpy().flatten()))
            path = np.vstack((source, receiver))
            paths_S.append(path)
            types_S.append(str((source_list_invers[source_list_invers['id'] == phase_listS_use.iloc[i, 0]]['type']).item()))
        phase_listS_use = pd.DataFrame(phase_listS_use_dum)

        node_all = np.vstack((node, node_outer))
        vel_allP = np.hstack((vel_node_P_akhir, modvel_outer.Vp))
        interpP = RBFInterpolator(node_all, vel_allP, kernel='linear', neighbors=8)
        gridVp = np.round(interpP(np.column_stack((Xinter.flatten(), Yinter.flatten(), Zinter.flatten()))),
                           decimals=4).reshape(len(xnode), len(ynode), len(znode))
        velgridP = VelocityGrid(node, xnode, ynode, znode, gridVp, deltn, delt, xfac, iter1, iter2, tmin)

        vel_allS = np.hstack((vel_node_S_akhir, modvel_outer.Vs))
        interpS = RBFInterpolator(node_all, vel_allS, kernel='linear', neighbors=8)
        gridVs = np.round(interpS(np.column_stack((Xinter.flatten(), Yinter.flatten(), Zinter.flatten()))),
                           decimals=4).reshape(len(xnode), len(ynode), len(znode))
        velgridS = VelocityGrid(node, xnode, ynode, znode, gridVs, deltn, delt, xfac, iter1, iter2, tmin)

        print('start forward after iteration: ',iter)
        logger.info('start forward after iteration: '+str(iter))
        # P Phase forward
        ttcal_P = []
        ttobsP_dum = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
            results=executor.map(velgridP.safe_ttime_only,paths_P,chunksize=1)
            for i, result in enumerate(results):
                if isinstance(result, dict) and "error" in result:
                    logger.info('not convergen in path P: ' + ' '.join(phase_listP_use.iloc[i, :].astype(str)))
                    logger.info(result['type'])
                    logger.info(result['error'])
                    logger.info(result['traceback'])
                    continue

                ttcal_i = result[1]
                ttcal_P.append(float(ttcal_i))
                ttobsP_dum.append(ttobs_P[i])
        ttobs_P=copy(ttobsP_dum)


        # S Phase forward
        ttcal_S = []
        ttobsS_dum = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=nu_cpu) as executor:
            results=executor.map(velgridS.safe_ttime_only,paths_S,chunksize=1)
            for i, result in enumerate(results):
                if isinstance(result, dict) and "error" in result:
                    logger.info('not convergen in path S: ' + ' '.join(phase_listS_use.iloc[i, :].astype(str)))
                    logger.info(result['type'])
                    logger.info(result['error'])
                    logger.info(result['traceback'])
                    continue

                ttcal_i = result[1]
                ttcal_S.append(float(ttcal_i))
                ttobsS_dum.append(ttobs_S[i])
        ttobs_S=copy(ttobsS_dum)


        ttobs_P = np.array(ttobs_P).flatten()
        ttcal_P = np.array(ttcal_P)
        t_res_P = ttobs_P - ttcal_P

        ttobs_S = np.array(ttobs_S).flatten()
        ttcal_S = np.array(ttcal_S)
        t_res_S = ttobs_S - ttcal_S

        t_res = np.hstack((t_res_P, t_res_S))
        rms2 = np.sqrt(np.mean(t_res ** 2))

        dt_var.append(rms2)
        print('rms in parameter ', iter, ' :', rms2)
        logger.info('rms in parameter '+ str(iter)+ ' :'+ str(rms2))
    #save_var=np.vstack((dt_var,dv_var)).T
    #np.savetxt("var_opt_list",save_var,delimiter=',',fmt='%.4f',header='X,Y,Z,Vp,Vs,Vp/Vs,dws_p,dws_s',comments='')
    damping_list['t_res']=np.array(dt_var)
    damping_list['vel_variance']=np.array(dv_var)
    damping_list['rnorm']=np.array(rnorm_list)
    damping_list['xnorm']=np.array(xnorm_list)
    damping_list['CND']=np.array(CND_list)
    damping_list.to_csv(folder_name+'/param_opt.csv', index=False)
    with ZipFile(folder_name + '/compress.zip', 'w') as zip:
        zip.write(folder_name + '/param_opt.csv')
    print('process is done')
    logger.info('process is done')