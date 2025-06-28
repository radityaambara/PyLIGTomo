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
    modvel = pd.read_csv("modawal_inner_agu20.csv")
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
    update_grid = True
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
    phase_list=pd.read_csv('synthetic_time.csv', header=0)
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
        if (float(phase_listP.iloc[i,4]))<=0:
            continue

        phase_listP_use.append(phase_listP.iloc[i,:])
        path = np.vstack((source, receiver))
        paths_P.append(path)
        ttobs_P.append(float(phase_listP.iloc[i][4]))

    phase_listP_use=pd.DataFrame(phase_listP_use)


    phase_listS_use=[]
    for i in range (len(phase_listS)):
        source=np.array(source_list[source_list['id']==phase_listS.iloc[i,0]][['easting','northing','elevation']])
        receiver=np.array(receiver_list[receiver_list['id']==phase_listS.iloc[i,1]][['easting','northing','elevation']])
        if (source.size == 0) or (receiver.size == 0):
            continue
        if (float(phase_listS.iloc[i,4]))<=0:
            continue
        phase_listS_use.append(phase_listS.iloc[i,:])
        path = np.vstack((source, receiver))
        paths_S.append(path)
        ttobs_S.append(float(phase_listS.iloc[i][4]))

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
        f1 = [executor.submit(velgridP.crkernel, pat) for pat in path_list_P]

    for i in range(len(path_list_P)):
        krn_i=f1[i]._result
        vp_kernel[i,:]=krn_i

    vs_kernel = np.zeros((len(path_list_S), node.shape[0]))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f1 = [executor.submit(velgridS.crkernel, pat) for pat in path_list_S]

    for i in range(len(path_list_S)):
        krn_i=f1[i]._result
        vs_kernel[i,:]=krn_i


    print('start adding node')
    logger.info('start adding node')
    isadd = update_grid
    while (isadd == True):
        hit_count = np.count_nonzero(vp_kernel, axis=0)
        node_tetahedron = Delaunay(node)
        added_node_list = []
        added_velP_list = []
        added_velS_list = []
        class_is_addnode = is_addnode(hit_count, up_threshold, node, deltn, interpP, interpS)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            f1 = [executor.submit(class_is_addnode.cek_add, sim) for sim in node_tetahedron.simplices]
        for i in range(len(node_tetahedron.simplices)):
            add_n = f1[i]._result
            if add_n[1] != -99:
                added_node_list.append(add_n[0])
                added_velP_list.append(add_n[1])
                added_velS_list.append(add_n[2])


        if len(added_velP_list)==0:
            isadd=False
            break

        node=np.vstack((node,np.array(added_node_list)))
        vel_node_P = np.hstack((vel_node_P.T, (np.array(added_velP_list))))
        vel_node_S = np.hstack((vel_node_S.T, (np.array(added_velS_list))))
        velgridP.change_node(node)
        velgridS.change_node(node)

        vp_kernel = np.zeros((len(path_list_P), node.shape[0]))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            f1 = [executor.submit(velgridP.crkernel, pat) for pat in path_list_P]

        for i in range(len(path_list_P)):
            krn_i = f1[i]._result
            vp_kernel[i, :] = krn_i

        vs_kernel = np.zeros((len(path_list_S), node.shape[0]))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            f1 = [executor.submit(velgridS.crkernel, pat) for pat in path_list_S]

        for i in range(len(path_list_S)):
            krn_i = f1[i]._result
            vs_kernel[i, :] = krn_i

    # start removing node
    print('start removing node')
    logger.info('start removing node')
    hit_count=np.count_nonzero(vp_kernel,axis=0)
    node=node[hit_count>low_threshold,:]
    vel_node_P=vel_node_P[hit_count>low_threshold]
    vel_node_S=vel_node_S[hit_count>low_threshold]
    velgridP.change_node(node)
    velgridS.change_node(node)

    vp_kernel = np.zeros((len(path_list_P), node.shape[0]))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f1 = [executor.submit(velgridP.crkernel, pat) for pat in path_list_P]

    for i in range(len(path_list_P)):
        krn_i = f1[i]._result
        vp_kernel[i, :] = krn_i

    vs_kernel = np.zeros((len(path_list_S), node.shape[0]))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f1 = [executor.submit(velgridS.crkernel, pat) for pat in path_list_S]

    for i in range(len(path_list_S)):
        krn_i = f1[i]._result
        vs_kernel[i, :] = krn_i


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
    hypo_kernel_P=hypo_kernel_P[abs(t_res_P_awal)<r_time_P,:]
    phase_listP_use=phase_listP_use.loc[phase_listP_use['t_res']<r_time_P]
    t_res_P_awal=t_res_P_awal[abs(t_res_P_awal)<r_time_P]

    vs_kernel=vs_kernel[abs(t_res_S_awal)<r_time_S,:]
    hypo_kernel_S=hypo_kernel_S[abs(t_res_S_awal)<r_time_S,:]
    phase_listS_use=phase_listS_use.loc[phase_listS_use['t_res']<r_time_S]
    t_res_S_awal=t_res_S_awal[abs(t_res_S_awal)<r_time_S]

    t_res_awal=np.hstack((t_res_P_awal,t_res_S_awal))
    rms1 = np.sqrt(np.mean(t_res_awal ** 2))
    t_res=copy(t_res_awal)

    fig1=plt.figure(figsize=[9.5,10])
    ax1=fig1.add_subplot(2,2,1)
    ax1.hist(t_res_awal,bins=30)
    ax1.set_ylim(ymin=0,ymax=6000)
    ax1.set_xlabel('Residual (s)')
    ax1.set_ylabel('Count')
    ax1.title.set_text('Initial (before inversion)')

    rms_list=[]
    rms_list.append(rms1)

    print('initial rms:', rms1)
    logger.info('initial rms: '+str(rms1))
    #mulai iterasi:
    for iter in range(0, iteration_number):
        print('start iteration:', iter)
        logger.info('start iteration: '+str(iter))
        weight = np.sqrt(vor_volumes(node))
        weight[weight==0]=np.max(weight)
        vp_kernel_inv = vp_kernel[:, weight != 0]
        vs_kernel_inv = vs_kernel[:, weight != 0]
        vp_kernel_zeros = np.zeros((vs_kernel_inv.shape[0],vp_kernel_inv.shape[1]))
        vs_kernel_zeros = np.zeros((vp_kernel_inv.shape[0],vs_kernel_inv.shape[1]))

        weight_inv = np.diag(1 / (weight[weight != 0]))
        smooth_damp=smooth_matrix(node[weight != 0,:])*damping_2
        zeros_smooth=np.zeros((smooth_damp.shape))

        inv_matrix_P=np.vstack((np.matmul(vp_kernel_inv,weight_inv),vp_kernel_zeros,smooth_damp,zeros_smooth))
        inv_matrix_S=np.vstack((vs_kernel_zeros,np.matmul(vs_kernel_inv,weight_inv),zeros_smooth,smooth_damp))

        hypo_zero=np.zeros((smooth_damp.shape[0]*2, hypo_kernel_P.shape[1]))
        hypo_kernel_inv=np.vstack((hypo_kernel_P,hypo_kernel_S,hypo_zero))

        t_smooth=np.zeros((smooth_damp.shape[0]*2))



        inv_matrix_hypo=np.hstack((hypo_kernel_inv,inv_matrix_P,inv_matrix_S))
        t_res_inv=np.hstack((t_res,t_smooth))


        inversion_result = lsqr(inv_matrix_hypo, t_res_inv, damp=damping_1)
        ds_P=np.matmul(inversion_result[0][hypo_kernel_P.shape[1]:hypo_kernel_P.shape[1]+vp_kernel_inv.shape[1]],weight_inv)
        ds_S=np.matmul(inversion_result[0][hypo_kernel_P.shape[1]+vp_kernel_inv.shape[1]:],weight_inv)
        acond=inversion_result[6]
        print(inversion_result[6])
        logger.info('CND value: '+str(inversion_result[6]))

        #updateVp
        vel_awal_P = copy(vel_node_P)
        vawal_P = copy(vel_node_P[weight!=0])
        dsawal_P = 1 / vawal_P
        dsakhir_P = ds_P + dsawal_P
        vakhir_P = 1/dsakhir_P
        delta_vp=vakhir_P-vawal_P
        vakhir_P[delta_vp>0.3]=vawal_P[delta_vp>0.3]+0.3
        vakhir_P[delta_vp<-0.3]=vawal_P[delta_vp<-0.3]-0.3
        vakhir_P[vakhir_P < 2]=2
        vakhir_P[vakhir_P > 8.5]=8.5
        vel_node_P[weight!=0] = vakhir_P

        #updateVs
        vel_awal_S = copy(vel_node_S)
        vawal_S = copy(vel_node_S[weight!=0])
        dsawal_S = 1 / vawal_S
        dsakhir_S = ds_S + dsawal_S
        vakhir_S = 1/dsakhir_S
        delta_vs=vakhir_S-vawal_S
        vakhir_S[delta_vs>0.175]=vawal_S[delta_vs>0.175]+0.175
        vakhir_S[delta_vs<-0.175]=vawal_S[delta_vs<-0.175]-0.175
        vakhir_S[vakhir_S < 1.1]=1.1
        vakhir_S[vakhir_S > 5]=5
        vel_node_S[weight!=0] = vakhir_S

        #update_hypo
        for i in range(0,len(source_list_invers)):
            source_list_invers.loc[i,'to_update'] += inversion_result[0][i*4]
            source_list_invers.loc[i,'easting'] += inversion_result[0][i*4+1]
            source_list_invers.loc[i,'northing'] += inversion_result[0][i*4+2]
            source_list_invers.loc[i,'elevation'] += inversion_result[0][i*4+3]
            source_list_invers.loc[i,'to_update_i'] = inversion_result[0][i * 4]

        #update phase use (hapus diluar treshold)
        paths_P = []
        ttobs_P = []
        phase_listP_use_dum = []
        for i in range(len(phase_listP_use)):
            source = np.array(
                source_list_invers[source_list_invers['id'] == phase_listP_use.iloc[i, 0]][['easting', 'northing', 'elevation']])
            receiver = np.array(
                receiver_list[receiver_list['id'] == phase_listP_use.iloc[i, 1]][['easting', 'northing', 'elevation']])
            t_update =(source_list_invers[source_list_invers['id'] == phase_listP_use.iloc[i, 0]][['to_update']])
            if (source.size == 0) or (receiver.size == 0):
                continue
            if (float(phase_listP_use.iloc[i][2])) <= 0:
                continue
            phase_listP_use_dum.append(phase_listP_use.iloc[i, :])
            ttobs_P.append(float(phase_listP_use.iloc[i][4]) - float(t_update.to_numpy().flatten()))
            path = np.vstack((source, receiver))
            paths_P.append(path)
        phase_listP_use = pd.DataFrame(phase_listP_use_dum)

        paths_S= []
        ttobs_S = []
        phase_listS_use_dum = []
        for i in range(len(phase_listS_use)):
            source = np.array(
                source_list_invers[source_list_invers['id'] == phase_listS_use.iloc[i, 0]][['easting', 'northing', 'elevation']])
            receiver = np.array(
                receiver_list[receiver_list['id'] == phase_listS_use.iloc[i, 1]][['easting', 'northing', 'elevation']])
            t_update = (source_list_invers[source_list_invers['id'] == phase_listS_use.iloc[i, 0]][['to_update']])
            if (source.size == 0) or (receiver.size == 0):
                continue
            if (float(phase_listS_use.iloc[i][2])) <= 0:
                continue
            phase_listS_use_dum.append(phase_listS_use.iloc[i, :])
            ttobs_S.append(float(phase_listS_use.iloc[i][4]) - float(t_update.to_numpy().flatten()))
            path = np.vstack((source, receiver))
            paths_S.append(path)
        phase_listS_use = pd.DataFrame(phase_listS_use_dum)

        node_all = np.vstack((node, node_outer))
        vel_allP = np.hstack((vel_node_P, modvel_outer.Vp))
        interpP = LinearNDInterpolator(node_all, vel_allP)
        gridVp = np.round_(interpP(Xinter,Yinter,Zinter),decimals=4)
        velgridP = VelocityGrid(node, xnode, ynode, znode, gridVp, deltn, delt, xfac, iter1, iter2, tmin)

        vel_allS = np.hstack((vel_node_S, modvel_outer.Vs))
        interpS = LinearNDInterpolator(node_all, vel_allS)
        gridVs = np.round_(interpS(Xinter,Yinter,Zinter),decimals=4)
        velgridS = VelocityGrid(node, xnode, ynode, znode, gridVs, deltn, delt, xfac, iter1, iter2, tmin)

        print('start forward after iteration: ',iter)
        logger.info('start forward after iteration: '+str(iter))
        # P Phase forward
        with concurrent.futures.ProcessPoolExecutor() as executor:
            f1 = [executor.submit(velgridP.forwardtwopoints, pat) for pat in paths_P]

        ttcal_P = []
        path_list_P = []
        vp_kernel = np.zeros((len(paths_P), node.shape[0]))
        hypo_kernel_P = np.zeros((len(paths_P), len(source_list) * 4))
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
                event_index_i=int((source_list.index[source_list.iloc[:,0]==float(phase_listP_use.iloc[i,0])]).tolist()[0])
                hypo_kernel_P[i, (event_index_i * 4):(event_index_i * 4) + 4] = hypoder_i
                ttcal_P.append(float(ttcal_i))
                phase_listP_use_dum.append(phase_listP_use.iloc[i, :])
                ttobsP_dum.append(ttobs_P[i])
            except:
                continue

        ttobs_P=copy(ttobsP_dum)
        vp_kernel = vp_kernel[~np.all(vp_kernel == 0, axis=1)]
        hypo_kernel_P = hypo_kernel_P[~np.all(hypo_kernel_P == 0, axis=1)]
        phase_listP_use = pd.DataFrame(phase_listP_use_dum)

        # S Phase forward
        with concurrent.futures.ProcessPoolExecutor() as executor:
            f1 = [executor.submit(velgridS.forwardtwopoints, pat) for pat in paths_S]

        ttcal_S = []
        path_list_S = []
        vs_kernel = np.zeros((len(paths_S), node.shape[0]))
        hypo_kernel_S = np.zeros((len(paths_S), len(source_list) * 4))
        phase_listS_use_dum=[]
        ttobsS_dum=[]
        for i in range(0, len(paths_S)):
            try:
                grpli = f1[i]._result
                krn_i = grpli[2]
                ttcal_i = grpli[1]
                hypoder_i = grpli[3]
                path_list_S.append(grpli[0])
                vs_kernel[i, :] = (np.array(krn_i))
                event_index_i=int((source_list.index[source_list.iloc[:,0]==float(phase_listS_use.iloc[i,0])]).tolist()[0])
                hypo_kernel_S[i, (event_index_i * 4):(event_index_i * 4) + 4] = hypoder_i
                ttcal_S.append(float(ttcal_i))
                phase_listS_use_dum.append(phase_listS_use.iloc[i, :])
                ttobsS_dum.append(ttobs_S[i])
            except:
                continue

        ttobs_S=copy(ttobsS_dum)
        vs_kernel = vs_kernel[~np.all(vs_kernel == 0, axis=1)]
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
        hypo_kernel_P = hypo_kernel_P[abs(t_res_P) < r_time_P, :]
        phase_listP_use = phase_listP_use.loc[phase_listP_use['t_res'] < r_time_P]
        path_list_P=list(itertools.compress(path_list_P, abs(t_res_P) < r_time_P))
        t_res_P = t_res_P[abs(t_res_P) < r_time_P]

        vs_kernel = vs_kernel[abs(t_res_S) < r_time_S, :]
        hypo_kernel_S = hypo_kernel_S[abs(t_res_S) < r_time_S, :]
        phase_listS_use = phase_listS_use.loc[phase_listS_use['t_res'] < r_time_S]
        path_list_S = list(itertools.compress(path_list_S, abs(t_res_S) < r_time_S))
        t_res_S = t_res_S[abs(t_res_S) < r_time_S]

        t_res = np.hstack((t_res_P, t_res_S))
        rms2 = np.sqrt(np.mean(t_res ** 2))

        rms_list.append(rms2)
        print('rms after iteration ', iter, ' :', rms2)
        logger.info('rms after iteration '+str(iter)+' : '+str(rms2))
        if iter == iteration_number-1 or rms2>rms1 or abs(rms2-rms1)<d_rms:
            break
        rms1 = np.copy(rms2)

        print('start adding node')
        logger.info('start adding node')
        isadd = False
        while (isadd == True):
            hit_count = np.count_nonzero(vp_kernel, axis=0)
            node_tetahedron = Delaunay(node)
            added_node_list = []
            added_velP_list = []
            added_velS_list = []
            class_is_addnode = is_addnode(hit_count, up_threshold, node, deltn, interpP, interpS)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                f1 = [executor.submit(class_is_addnode.cek_add, sim) for sim in node_tetahedron.simplices]
            for i in range(len(node_tetahedron.simplices)):
                add_n = f1[i]._result
                if add_n[1] != -99:
                    added_node_list.append(add_n[0])
                    added_velP_list.append(add_n[1])
                    added_velS_list.append(add_n[2])

            if len(added_velP_list) == 0:
                isadd = False
                break

            node = np.vstack((node, np.array(added_node_list)))
            vel_node_P = np.hstack((vel_node_P.T, (np.array(added_velP_list))))
            vel_node_S = np.hstack((vel_node_S.T, (np.array(added_velS_list))))
            velgridP.change_node(node)
            velgridS.change_node(node)

            vp_kernel = np.zeros((len(path_list_P), node.shape[0]))
            with concurrent.futures.ProcessPoolExecutor() as executor:
                f1 = [executor.submit(velgridP.crkernel, pat) for pat in path_list_P]

            for i in range(len(path_list_P)):
                krn_i = f1[i]._result
                vp_kernel[i, :] = krn_i

            vs_kernel = np.zeros((len(path_list_S), node.shape[0]))
            with concurrent.futures.ProcessPoolExecutor() as executor:
                f1 = [executor.submit(velgridS.crkernel, pat) for pat in path_list_S]

            for i in range(len(path_list_S)):
                krn_i = f1[i]._result
                vs_kernel[i, :] = krn_i

        # start removing node
        print('start removing node')
        hit_count = np.count_nonzero(vp_kernel, axis=0)
        logger.info('start removing node')
        node = node[hit_count > low_threshold, :]
        vel_node_P = vel_node_P[hit_count > low_threshold]
        vel_node_S = vel_node_S[hit_count > low_threshold]
        velgridP.change_node(node)
        velgridS.change_node(node)

        vp_kernel = np.zeros((len(path_list_P), node.shape[0]))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            f1 = [executor.submit(velgridP.crkernel, pat) for pat in path_list_P]

        for i in range(len(path_list_P)):
            krn_i = f1[i]._result
            vp_kernel[i, :] = krn_i

        vs_kernel = np.zeros((len(path_list_S), node.shape[0]))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            f1 = [executor.submit(velgridS.crkernel, pat) for pat in path_list_S]

        for i in range(len(path_list_S)):
            krn_i = f1[i]._result
            vs_kernel[i, :] = krn_i

    ax2=fig1.add_subplot(2,2,2)
    ax2.hist(t_res,bins=30)
    ax2.set_ylim(ymin=0,ymax=6000)
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
    fig2.savefig('RHC hist P')
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
    fig3.savefig('RHC hist S')
    logger.info('<--- RHC S final ---> \n'+'mean: '+ str(np.mean(hit_countS[hit_countS>0]))+' || median: '+
                                                     str(np.median(hit_countS[hit_countS>0]))+' || deviation: '+
                                                     str(np.std(hit_countS[hit_countS>0])))
    logger.info('jumlah non-zero node S: '+str(len(hit_countS[hit_countS>0])))

    df_grid = pd.DataFrame(node, columns=['X', 'Y', 'Z'])
    df_grid['Vp'] = vel_node_P
    df_grid['Vs'] = vel_node_S
    df_grid['hcP'] = np.count_nonzero(vp_kernel, axis=0)
    df_grid['hcS'] = np.count_nonzero(vs_kernel, axis=0)
    df_grid.to_csv('vel_result.csv', index=False)
    source_list_invers.to_csv('source_result.csv', index=False)


    ax3=fig1.add_subplot(2,2,(3,4))
    ax3.plot(np.array(rms_list))
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('RMS (s)')
    ax3.title.set_text('RMS per iteration')
    fig1.savefig('statistik result')
    plt.show()

    #simpan ray-tracing file
    ray1=path_list_P[0]
    ray2=np.ones((len(ray1),1))*0
    ray_out=np.hstack((ray1,ray2))
    for i in range(1,len(path_list_P)):
        ray1=path_list_P[i]
        ray2=np.ones((len(ray1),1))*i
        ray3=np.hstack((ray1,ray2))
        ray_out = np.vstack((ray_out, ray3))
    np.savetxt('ray_akhir',ray_out,delimiter=',',fmt='%.4f')
    #plot


    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #for path in path_list_P:
    #    ax.plot(path[:,0],path[:,1],path[:,2],alpha=0.1)

    #ax.scatter(node[:,0],node[:,1],node[:,2])
    #plt.show()
