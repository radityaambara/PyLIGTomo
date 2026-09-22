from .all_func import VelocityGrid, vor_volumes, smooth_matrix, is_addnode
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

def run_invers(modvel, modvel_outer, source_list, receiver_list, phase_list, delt, deltn, xfac, iter1, iter2, tmin,
                 iteration_number, up_threshold, low_threshold, d_rms, r_time_P, r_time_S, damping_1, damping_2,
               update_grid,folder_name,update_grid_after=False):
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
                "\n iteration_number: "+str(iteration_number)+
                "\n up_threshold: "+str(up_threshold)+
                "\n low_threshold: "+str(low_threshold)+
                "\n d_rms: "+str(d_rms)+
                "\n r_time_P: "+str(r_time_P)+
                "\n r_time_S: "+str(r_time_S)+
                "\n damping_1: "+str(damping_1)+
                "\n damping_2: "+str(damping_2)+
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
    iteration_number = int(iteration_number)
    up_threshold = float(up_threshold)
    low_threshold = float(low_threshold)
    d_rms=float(d_rms)
    r_time_P = float(r_time_P)
    r_time_S = float(r_time_S)
    damping_1 = float(damping_1)
    damping_2 = float(damping_2)
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
    types_P=[]
    paths_S=[]
    ttobs_S=[]
    types_S=[]

    #source_list_invers=source_list.copy()
    #source_list_invers['to_update']=0
    #source_list_invers['to_update_i']=0
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
        types_P.append(str((source_list[source_list['id']==phase_listP.iloc[i,0]]['type']).item()))

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
    print('forward P phase')
    logger.info('forward P phase')
    ttcal_P = []
    path_list_P =[]
    vp_kernel = np.zeros((len(paths_P), node.shape[0]))
    if len(hypo_list)!=0:
        hypo_kernel_P = np.zeros((len(paths_P),len(hypo_list)*4))
    phase_listP_use_dum=[]
    ttobsP_dum=[]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_iterator = executor.map(velgridP.forwardtwopoints, paths_P, chunksize=1)
        for i, grpli in enumerate(results_iterator):
            try:
                krn_i = grpli[2]
                ttcal_i = grpli[1]
                hypoder_i = grpli[3]
                path_list_P.append(grpli[0])
                vp_kernel[i, :] = (np.array(krn_i))
                if types_P[i]==0:
                    event_index_i=int((hypo_list.index[hypo_list.iloc[:,0]==int(phase_listP_use.iloc[i,0])]).tolist()[0])
                    hypo_kernel_P[i,(event_index_i*4):(event_index_i*4)+4]=hypoder_i
                ttcal_P.append(float(ttcal_i))
                phase_listP_use_dum.append(phase_listP_use.iloc[i,:])
                ttobsP_dum.append(ttobs_P[i])
            except Exception as e:
                logger.info('not convergen in path P: ' + ' '.join(phase_listP_use.iloc[i, :].astype(str)))
                logger.info(f"error = {e}")
                continue

    ttobs_P = copy(ttobsP_dum)
    vp_kernel = vp_kernel[~np.all(vp_kernel == 0, axis=1)]
    if len(hypo_list)!=0:
        hypo_kernel_P = hypo_kernel_P[~np.all(hypo_kernel_P == 0, axis=1)]
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
    np.savetxt(folder_name+'/_ray_initial',ray_out,delimiter=',',fmt='%.4f')

    print('forward S phase')
    logger.info('forward S phase')
    ttcal_S = []
    path_list_S =[]
    vs_kernel = np.zeros((len(paths_S), node.shape[0]))
    if len(hypo_list)!=0:
        hypo_kernel_S = np.zeros((len(paths_S),len(source_list)*4))
    phase_listS_use_dum = []
    ttobsS_dum = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_iterator = executor.map(velgridS.forwardtwopoints, paths_S, chunksize=1)
        for i, grpli in enumerate(results_iterator):
            try:
                krn_i = grpli[2]
                ttcal_i = grpli[1]
                hypoder_i = grpli[3]
                path_list_S.append(grpli[0])
                vs_kernel[i, :] = (np.array(krn_i))
                if types_S[i]==0:
                    event_index_i=int((hypo_list.index[hypo_list.iloc[:,0]==float(phase_listS_use.iloc[i,0])]).tolist()[0])
                    hypo_kernel_S[i,(event_index_i*4):(event_index_i*4)+4]=hypoder_i
                ttcal_S.append(float(ttcal_i))
                phase_listS_use_dum.append(phase_listS_use.iloc[i, :])
                ttobsS_dum.append(ttobs_S[i])
            except Exception as e:
                logger.info('not convergen in path S: ' + ' '.join(phase_listS_use.iloc[i, :].astype(str)))
                logger.info(f"error = {e}")
                continue

    ttobs_S = copy(ttobsS_dum)
    vs_kernel = vs_kernel[~np.all(vs_kernel == 0, axis=1)]
    if len(hypo_list)!=0:
        hypo_kernel_S = hypo_kernel_S[~np.all(hypo_kernel_S == 0, axis=1)]
    phase_listS_use = pd.DataFrame(phase_listS_use_dum)

    #start adding node
    print('start removing node')
    logger.info('start removing node')
    hit_count=np.count_nonzero(vp_kernel,axis=0)
    hit_countPawal=np.count_nonzero(vp_kernel,axis=0)
    hit_countSawal=np.count_nonzero(vs_kernel,axis=0)
    fig2=plt.figure(figsize=[12,7])
    ax21=fig2.add_subplot(1,2,1)
    ax21.hist(hit_countPawal[hit_countPawal>0],bins=100)
    ax21.set_title('Initial RHC distribution of P wave',fontsize=15,fontweight='bold')
    ax21.set_xlabel("RayHitCount", fontsize=15)
    ax21.set_ylabel("frequency", fontsize=15)
    ax21.tick_params(axis='x', labelsize=15)
    ax21.tick_params(axis='y', labelsize=15)
    print('RHC P awal',np.mean(hit_countPawal[hit_countPawal>0]),np.median(hit_countPawal[hit_countPawal>0]),
    np.std(hit_countPawal[hit_countPawal>0]))
    print('jumlah non-zero node awal P', len(hit_countPawal[hit_countPawal>0]))
    logger.info('<--- initial RHC P ---> \n'+'mean: '+str(np.mean(hit_countPawal[hit_countPawal>0]))+' || median: '+
                                                      str(np.median(hit_countPawal[hit_countPawal>0]))+' || deviation: '+
                                                      str(np.std(hit_countPawal[hit_countPawal>0])))
    logger.info('jumlah non-zero node awal P: '+str(len(hit_countPawal[hit_countPawal>0])))
    fig3=plt.figure(figsize=[12,7])
    ax31=fig3.add_subplot(1,2,1)
    ax31.hist(hit_countSawal[hit_countSawal>0],bins=100)
    ax31.set_title('Initial RHC distribution of S wave',fontsize=15,fontweight='bold')
    ax31.set_xlabel("RayHitCount", fontsize=15)
    ax31.set_ylabel("frequency", fontsize=15)
    ax31.tick_params(axis='x', labelsize=15)
    ax31.tick_params(axis='y', labelsize=15)
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

    vp_kernel = np.zeros((len(path_list_P), node.shape[0]))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_iterator = executor.map(velgridP.crkernel, path_list_P, chunksize=1)

        for i, krn_i in enumerate(results_iterator):
            vp_kernel[i,:]=krn_i

    vs_kernel = np.zeros((len(path_list_S), node.shape[0]))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_iterator = executor.map(velgridS.crkernel, path_list_S, chunksize=1)

        for i, krn_i in enumerate(results_iterator):
            vs_kernel[i,:]=krn_i


    print('start adding node')
    logger.info('start adding node')
    isadd = update_grid
    while (isadd == True):
        hit_count = np.count_nonzero(vp_kernel, axis=0)
        node_tetahedron = Delaunay(node)
        class_is_addnode = is_addnode(hit_count, up_threshold, node, deltn, interpP, interpS)
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

        vp_kernel = np.zeros((len(path_list_P), node.shape[0]))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results_iterator = executor.map(velgridP.crkernel, path_list_P, chunksize=1)

            for i, krn_i in enumerate(results_iterator):
                vp_kernel[i, :] = krn_i

        vs_kernel = np.zeros((len(path_list_S), node.shape[0]))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results_iterator = executor.map(velgridS.crkernel, path_list_S, chunksize=1)

            for i, krn_i in enumerate(results_iterator):
                vs_kernel[i, :] = krn_i

    # start removing node
    print('start removing node')
    logger.info('start removing node')
    hit_count = np.count_nonzero(vp_kernel, axis=0)
    node=node[hit_count>low_threshold,:]
    vel_node_P=vel_node_P[hit_count>low_threshold]
    vel_node_S=vel_node_S[hit_count>low_threshold]
    velgridP.change_node(node)
    velgridS.change_node(node)

    vp_kernel = np.zeros((len(path_list_P), node.shape[0]))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_iterator = executor.map(velgridP.crkernel, path_list_P, chunksize=1)

        for i, krn_i in enumerate(results_iterator):
            vp_kernel[i,:]=krn_i

    vs_kernel = np.zeros((len(path_list_S), node.shape[0]))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_iterator = executor.map(velgridS.crkernel, path_list_S, chunksize=1)

        for i, krn_i in enumerate(results_iterator):
            vs_kernel[i,:]=krn_i


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
    vp_kernel=scsp.csr_matrix(vp_kernel)
    if len(hypo_list)!=0:
        hypo_kernel_P=hypo_kernel_P[abs(t_res_P_awal)<r_time_P,:]
        hypo_kernel_P=scsp.csr_matrix(hypo_kernel_P)
    phase_listP_use=phase_listP_use.loc[phase_listP_use['t_res']<r_time_P]
    t_res_P_awal=t_res_P_awal[abs(t_res_P_awal)<r_time_P]

    vs_kernel=vs_kernel[abs(t_res_S_awal)<r_time_S,:]
    vs_kernel=scsp.csr_matrix(vs_kernel)
    if len(hypo_list)!=0:
        hypo_kernel_S=hypo_kernel_S[abs(t_res_S_awal)<r_time_S,:]
        hypo_kernel_S=scsp.csr_matrix(hypo_kernel_S)
    phase_listS_use=phase_listS_use.loc[phase_listS_use['t_res']<r_time_S]
    t_res_S_awal=t_res_S_awal[abs(t_res_S_awal)<r_time_S]

    t_res_awal=np.hstack((t_res_P_awal,t_res_S_awal))
    rms1 = np.sqrt(np.mean(t_res_awal ** 2))
    t_res=copy(t_res_awal)

    fig1=plt.figure(figsize=[9.5,10])
    ax1=fig1.add_subplot(2,2,1)
    n1, _, _ =ax1.hist(t_res_awal,bins=30)
    ax1.set_xlabel('Residual (s)')
    ax1.set_ylabel('Count')
    ax1.title.set_text('Initial (before inversion)')

    rms_list=[rms1]

    print('initial rms:', rms1)
    logger.info('initial rms: ' + str(rms1))

    #mulai iterasi:
    for iter in range(0, iteration_number):
        print('start iteration:', iter)
        logger.info('start iteration: ' + str(iter))
        weight = np.sqrt(vor_volumes(node))
        weight[weight == 0] = np.max(weight)
        vp_kernel_inv = vp_kernel[:, weight != 0]
        vs_kernel_inv = vs_kernel[:, weight != 0]
        vp_kernel_zeros = np.zeros((vs_kernel_inv.shape[0],vp_kernel_inv.shape[1]))
        vs_kernel_zeros = np.zeros((vp_kernel_inv.shape[0],vs_kernel_inv.shape[1]))

        weight_inv = scsp.diags(1 / (weight[weight != 0]))
        smooth_damp=scsp.csr_matrix(smooth_matrix(node[weight != 0,:]))*damping_2
        zeros_smooth=scsp.csr_matrix(smooth_damp.shape)

        v_stack_P = scsp.vstack([vp_kernel_inv, vp_kernel_zeros, smooth_damp, zeros_smooth])
        v_stack_S = scsp.vstack([vs_kernel_zeros, vs_kernel_inv, zeros_smooth, smooth_damp])
        inv_matrix_P = v_stack_P.dot(weight_inv)
        inv_matrix_S = v_stack_S.dot(weight_inv)

        if len(hypo_list)!=0:
            hypo_zero=scsp.csr_matrix((smooth_damp.shape[0]*2, hypo_kernel_P.shape[1]))
            hypo_kernel_inv=scsp.vstack((hypo_kernel_P,hypo_kernel_S,hypo_zero))
            inv_matrix_hypo = scsp.hstack((hypo_kernel_inv, inv_matrix_P, inv_matrix_S))
        else:
            inv_matrix_hypo=scsp.hstack((inv_matrix_P, inv_matrix_S))
        t_smooth = np.zeros((smooth_damp.shape[0] * 2))
        t_res_inv = np.hstack((t_res, t_smooth))
        ## add resolution matrix
        #i_damp=np.identity(inv_matrix_hypo.shape[1])*damping_1
        #G_full=np.vstack((inv_matrix_hypo,i_damp))
        #G_inv = np.linalg.pinv(G_full)
        #res_matrix=G_inv@G_full
        #diag_res=np.trace(res_matrix)/inv_matrix_hypo.shape[1]
        # print(diag_res)
        # logger.info('diagonal resolution matrix: ' + str(diag_res))
        ## resolution matrix end here
        inversion_result = lsmr(inv_matrix_hypo, t_res_inv, damp=damping_1)

        if len(hypo_list)!=0:
            ds_P=np.matmul(inversion_result[0][hypo_kernel_P.shape[1]:hypo_kernel_P.shape[1]+vp_kernel_inv.shape[1]],weight_inv)
            ds_S=np.matmul(inversion_result[0][hypo_kernel_P.shape[1]+vp_kernel_inv.shape[1]:],weight_inv)
        else:
            ds_P = np.matmul(inversion_result[0][:vp_kernel_inv.shape[1]], weight_inv)
            ds_S = np.matmul(inversion_result[0][vp_kernel_inv.shape[1]:], weight_inv)

        #acond=inversion_result[6]
        print("CND: "+str(inversion_result[6]))
        print("istop: "+str(inversion_result[1]))

        logger.info('CND value: ' + str(inversion_result[6]))
        logger.info('istop: ' + str(inversion_result[1]))
        logger.info('rnorm: ' + str(inversion_result[3]))
        logger.info('xnorm: ' + str(inversion_result[7]))
        logger.info('no_ite: ' + str(inversion_result[2]))



        #updateVp
        #vel_awal_P = copy(vel_node_P)
        vawal_P = copy(vel_node_P[weight!=0])
        dsawal_P = 1 / vawal_P
        dsakhir_P = ds_P + dsawal_P
        vel_node_P[weight!=0] = 1 / dsakhir_P

        #updateVs
        #vel_awal_S = copy(vel_node_S)
        vawal_S = copy(vel_node_S[weight!=0])
        dsawal_S = 1 / vawal_S
        dsakhir_S = ds_S + dsawal_S
        vel_node_S[weight!=0] = 1 / dsakhir_S

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
        #update phase use (delete outside treshold)
        paths_P = []
        ttobs_P = []
        phase_listP_use_dum = []
        types_P=[]
        for i in range(len(phase_listP_use)):
            source = np.array(source_list_invers[source_list_invers['id'] == phase_listP_use.iloc[i, 0]][['easting', 'northing', 'elevation']])
            receiver = np.array(receiver_list[receiver_list['id'] == phase_listP_use.iloc[i, 1]][['easting', 'northing', 'elevation']])
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
        types_S=[]
        for i in range(len(phase_listS_use)):
            source = np.array(source_list_invers[source_list_invers['id'] == phase_listS_use.iloc[i, 0]][['easting', 'northing', 'elevation']])
            receiver = np.array(receiver_list[receiver_list['id'] == phase_listS_use.iloc[i, 1]][['easting', 'northing', 'elevation']])
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
        vel_allP = np.hstack((vel_node_P, modvel_outer.Vp))
        interpP = RBFInterpolator(node_all, vel_allP, kernel='linear', neighbors=8)
        gridVp = np.round(interpP(np.column_stack((Xinter.flatten(), Yinter.flatten(), Zinter.flatten()))),
                           decimals=4).reshape(len(xnode), len(ynode), len(znode))
        velgridP = VelocityGrid(node, xnode, ynode, znode, gridVp, deltn, delt, xfac, iter1, iter2, tmin)

        vel_allS = np.hstack((vel_node_S, modvel_outer.Vs))
        interpS = RBFInterpolator(node_all, vel_allS, kernel='linear', neighbors=8)
        gridVs = np.round(interpS(np.column_stack((Xinter.flatten(), Yinter.flatten(), Zinter.flatten()))),
                           decimals=4).reshape(len(xnode), len(ynode), len(znode))
        velgridS = VelocityGrid(node, xnode, ynode, znode, gridVs, deltn, delt, xfac, iter1, iter2, tmin)

        print('start forward after iteration: ',iter)
        logger.info('start forward after iteration: ' + str(iter))

        # P Phase forward
        ttcal_P = []
        path_list_P = []
        vp_kernel = np.zeros((len(paths_P), node.shape[0]))
        if len(hypo_list)!=0:
            hypo_kernel_P = np.zeros((len(paths_P), len(hypo_list) * 4))
        phase_listP_use_dum=[]
        ttobsP_dum=[]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results_iterator = executor.map(velgridP.forwardtwopoints, paths_P, chunksize=1)

            for i, grpli in enumerate(results_iterator):
                try:
                    krn_i = grpli[2]
                    ttcal_i = grpli[1]
                    hypoder_i = grpli[3]
                    path_list_P.append(grpli[0])
                    vp_kernel[i, :] = (np.array(krn_i))
                    if types_P==0:
                        event_index_i=int((hypo_list.index[hypo_list.iloc[:,0]==float(phase_listP_use.iloc[i,0])]).tolist()[0])
                        hypo_kernel_P[i, (event_index_i * 4):(event_index_i * 4) + 4] = hypoder_i
                    ttcal_P.append(float(ttcal_i))
                    phase_listP_use_dum.append(phase_listP_use.iloc[i, :])
                    ttobsP_dum.append(ttobs_P[i])
                except Exception as e:
                    logger.info(f"not convergen in path P: {' '.join(phase_listP_use.iloc[i, :].astype(str))}")
                    logger.info(f"error = {e}")
                    continue

        ttobs_P=copy(ttobsP_dum)
        vp_kernel = vp_kernel[~np.all(vp_kernel == 0, axis=1)]
        if len(hypo_list)!=0:
            hypo_kernel_P = hypo_kernel_P[~np.all(hypo_kernel_P == 0, axis=1)]
        phase_listP_use = pd.DataFrame(phase_listP_use_dum)

        ttcal_S = []
        path_list_S = []
        vs_kernel = np.zeros((len(paths_S), node.shape[0]))
        if len(hypo_list)!=0:
            hypo_kernel_S = np.zeros((len(paths_S), len(source_list) * 4))
        phase_listS_use_dum=[]
        ttobsS_dum=[]
        # S Phase forward
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results_iterator = executor.map(velgridS.forwardtwopoints, paths_S, chunksize=1)

            for i, grpli in enumerate(results_iterator):
                try:
                    krn_i = grpli[2]
                    ttcal_i = grpli[1]
                    hypoder_i = grpli[3]
                    path_list_S.append(grpli[0])
                    vs_kernel[i, :] = (np.array(krn_i))
                    if types_S==0:
                        event_index_i=int((hypo_list.index[hypo_list.iloc[:,0]==float(phase_listS_use.iloc[i,0])]).tolist()[0])
                        hypo_kernel_S[i, (event_index_i * 4):(event_index_i * 4) + 4] = hypoder_i
                    ttcal_S.append(float(ttcal_i))
                    phase_listS_use_dum.append(phase_listS_use.iloc[i, :])
                    ttobsS_dum.append(ttobs_S[i])
                except:
                    logger.info('not convergen in path S: ' + ' '.join(phase_listS_use.iloc[i, :].astype(str)))
                    logger.info(f"error = {e}")
                    continue

        ttobs_S=copy(ttobsS_dum)
        vs_kernel = vs_kernel[~np.all(vs_kernel == 0, axis=1)]
        if len(hypo_list)!=0:
            hypo_kernel_S = hypo_kernel_S[~np.all(hypo_kernel_S == 0, axis=1)]
        phase_listS_use = pd.DataFrame(phase_listS_use_dum)


        ttobs_P = np.array(ttobs_P).flatten()
        ttcal_P = np.array(ttcal_P)
        t_res_P = ttobs_P - ttcal_P
        phase_listP_use['t_res'] = abs(t_res_P)

        ttobs_S = np.array(ttobs_S).flatten()
        ttcal_S = np.array(ttcal_S)
        t_res_S = ttobs_S - ttcal_S
        phase_listS_use['t_res'] = abs(t_res_S)

        #update data
        vp_kernel = vp_kernel[abs(t_res_P) < r_time_P, :]
        vp_kernel = scsp.csr_matrix(vp_kernel)
        if len(hypo_list)!=0:
            hypo_kernel_P = hypo_kernel_P[abs(t_res_P) < r_time_P, :]
            hypo_kernel_P = scsp.csr_matrix(hypo_kernel_P)
        phase_listP_use = phase_listP_use.loc[phase_listP_use['t_res'] < r_time_P]
        path_list_P=list(itertools.compress(path_list_P, abs(t_res_P) < r_time_P))
        t_res_P = t_res_P[abs(t_res_P) < r_time_P]

        vs_kernel = vs_kernel[abs(t_res_S) < r_time_S, :]
        vs_kernel = scsp.csr_matrix(vs_kernel)
        if len(hypo_list)!=0:
            hypo_kernel_S = hypo_kernel_S[abs(t_res_S) < r_time_S, :]
            hypo_kernel_S = scsp.csr_matrix(hypo_kernel_S)
        phase_listS_use = phase_listS_use.loc[phase_listS_use['t_res'] < r_time_S]
        path_list_S = list(itertools.compress(path_list_S, abs(t_res_S) < r_time_S))
        t_res_S = t_res_S[abs(t_res_S) < r_time_S]

        t_res = np.hstack((t_res_P, t_res_S))
        rms2 = np.sqrt(np.mean(t_res ** 2))

        rms_list.append(rms2)
        print('rms after iteration ', iter, ' :', rms2)
        logger.info('rms after iteration ' + str(iter) + ' : ' + str(rms2))

        if rms2>rms1 and iter>0:
            break
        if abs(rms2-rms1)<d_rms and iter>0:
            break
        if iter == iteration_number-1:
            break
        rms1 = np.copy(rms2)

        print('start adding node')
        logger.info('start adding node')

        isadd = update_grid_after
        while isadd == True:
            hit_count = np.count_nonzero(vp_kernel, axis=0)
            node_tetahedron = Delaunay(node)
            class_is_addnode = is_addnode(hit_count, up_threshold, node, deltn, interpP, interpS)
            sim = node_tetahedron.simplices
            added_node_list, added_velP_list, added_velS_list = class_is_addnode.add_simultanius(sim)
            if len(added_velP_list) == 0:
                isadd = False
                break

            node = np.vstack((node, added_node_list))
            vel_node_P = np.hstack((vel_node_P.T, added_velP_list))
            vel_node_S = np.hstack((vel_node_S.T, added_velS_list))
            velgridP.change_node(node)
            velgridS.change_node(node)

            vp_kernel = np.zeros((len(path_list_P), node.shape[0]))
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results_iterator = executor.map(velgridP.crkernel, path_list_P, chunksize=1)

                for i, krn_i in enumerate(results_iterator):
                    vp_kernel[i, :] = krn_i

            vs_kernel = np.zeros((len(path_list_S), node.shape[0]))
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results_iterator = executor.map(velgridS.crkernel, path_list_S, chunksize=1)

                for i, krn_i in enumerate(results_iterator):
                    vs_kernel[i, :] = krn_i

        if update_grid_after==True:
            # start removing node
            print('start removing node')
            logger.info('start removing node')
            hit_count = np.count_nonzero(vp_kernel, axis=0)
            node = node[hit_count > low_threshold, :]
            vel_node_P = vel_node_P[hit_count > low_threshold]
            vel_node_S = vel_node_S[hit_count > low_threshold]
            velgridP.change_node(node)
            velgridS.change_node(node)

            vp_kernel = np.zeros((len(path_list_P), node.shape[0]))
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results_iterator = executor.map(velgridP.crkernel, path_list_P, chunksize=1)

                for i, krn_i in enumerate(results_iterator):
                    vp_kernel[i, :] = krn_i

            vs_kernel = np.zeros((len(path_list_S), node.shape[0]))
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results_iterator = executor.map(velgridS.crkernel, path_list_S, chunksize=1)

                for i, krn_i in enumerate(results_iterator):
                    vs_kernel[i, :] = krn_i

        vp_kernel=scsp.csr_matrix(vp_kernel)
        vs_kernel=scsp.csr_matrix(vs_kernel)

    ax2=fig1.add_subplot(2,2,2)
    n2, _, _ =ax2.hist(t_res,bins=30)
    max_ylim = max(max(n1), max(n2)) * 1.1
    ax1.set_ylim(ymin=0,ymax=max_ylim)
    ax2.set_ylim(ymin=0,ymax=max_ylim)
    ax2.set_xlabel('Residual (s)')
    ax2.set_ylabel('Count')
    ax2.title.set_text('Final (after inversion)')

    hit_countP = np.count_nonzero(vp_kernel_inv, axis=0)
    hit_countS = np.count_nonzero(vs_kernel_inv, axis=0)
    ax22=fig2.add_subplot(1,2,2)
    ax22.hist(hit_countP,bins=100)
    ax22.set_title('Final RHC distribution of P wave',fontsize=15,fontweight='bold')
    ax22.set_xlabel("RayHitCount", fontsize=15)
    ax22.set_ylabel("frequency", fontsize=15)
    ax22.tick_params(axis='x', labelsize=15)
    ax22.tick_params(axis='y', labelsize=15)
    print('RHC P final',np.mean(hit_countP),np.median(hit_countP),np.std(hit_countP))
    print('jumlah non-zero node akhir P', len(hit_countP))
    fig2.savefig(folder_name+'/RHC hist P.png')
    logger.info('<--- RHC P final ---> \n'+'mean: '+ str(np.mean(hit_countP[hit_countP>0]))+' || median: '+
                                                     str(np.median(hit_countP[hit_countP>0]))+' || deviation: '+
                                                     str(np.std(hit_countP[hit_countP>0])))
    logger.info('jumlah non-zero node P: '+str(len(hit_countP[hit_countP>0])))

    ax32=fig3.add_subplot(1,2,2)
    ax32.hist(hit_countS,bins=100)
    ax32.set_title('Final RHC distribution of S wave',fontsize=15,fontweight='bold')
    ax32.set_xlabel("RayHitCount", fontsize=15)
    ax32.set_ylabel("frequency", fontsize=15)
    ax32.tick_params(axis='x', labelsize=15)
    ax32.tick_params(axis='y', labelsize=15)
    print('RHC S final',np.mean(hit_countS),np.median(hit_countS),np.std(hit_countS))
    print('jumlah non-zero node akhir S', len(hit_countS))
    fig3.savefig(folder_name+'/RHC hist S.png')
    logger.info('<--- RHC S final ---> \n'+'mean: '+ str(np.mean(hit_countS[hit_countS>0]))+' || median: '+
                                                     str(np.median(hit_countS[hit_countS>0]))+' || deviation: '+
                                                     str(np.std(hit_countS[hit_countS>0])))
    logger.info('jumlah non-zero node S: '+str(len(hit_countS[hit_countS>0])))

    df_grid = pd.DataFrame(node, columns=['X', 'Y', 'Z'])
    df_grid['Vp'] = vel_node_P
    df_grid['Vs'] = vel_node_S
    df_grid['hcP'] = np.count_nonzero(vp_kernel, axis=0)
    df_grid['hcS'] = np.count_nonzero(vs_kernel, axis=0)
    df_grid.to_csv(folder_name+'/vel_invers.csv', index=False)
    source_list_invers.to_csv(folder_name+'/source_invers.csv', index=False)


    ax3=fig1.add_subplot(2,2,(3,4))
    ax3.plot(np.array(rms_list))
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('RMS (s)')
    ax3.title.set_text('RMS per iteration')
    fig1.savefig(folder_name+'/statistical.png')
    #plt.show()

    #save ray-tracing file
    ray1=path_list_P[0]
    ray2=np.ones((len(ray1),1))*0
    ray_out=np.hstack((ray1,ray2))
    for i in range(1,len(path_list_P)):
        ray1=path_list_P[i]
        ray2=np.ones((len(ray1),1))*i
        ray3=np.hstack((ray1,ray2))
        ray_out = np.vstack((ray_out, ray3))
    np.savetxt(folder_name+'/_ray_final',ray_out,delimiter=',',fmt='%.4f')

    with ZipFile(folder_name+'/compress.zip', 'w') as zips:
        zips.write(folder_name+'/_ray_final')
        zips.write(folder_name+'/statistical.png')
        zips.write(folder_name+'/source_invers.csv')
        zips.write(folder_name+'/vel_invers.csv')
        zips.write(folder_name+'/RHC hist S.png')
        zips.write(folder_name+'/RHC hist P.png')
        zips.write(folder_name+'/_ray_initial')

    print('process is done')
    logger.info('process is done')
