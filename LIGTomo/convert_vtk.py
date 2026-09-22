from pyevtk.hl import polyLinesToVTK,pointsToVTK,gridToVTK
import numpy as np
import pandas as pd
from scipy.interpolate import  RBFInterpolator
from scipy.spatial import  Delaunay

def ray_tovtk(file, scalling):
    ray = np.genfromtxt(file, delimiter=',')
    id_num = ray[:, 3]
    values, counts = np.unique(id_num, return_counts=True)

    polyLinesToVTK(file+'_ray_vtk', ray[:, 0].flatten() * scalling, ray[:, 1].flatten() * scalling,
                   ray[:, 2].flatten() * scalling, pointsPerLine=counts, pointData={'data': ray[:, 3].flatten()})


def event_tovtk(file,scalling):
    source = pd.read_csv(file)
    pointsToVTK(file+'_event.vtk', np.array(source.easting) * scalling, np.array(source.northing) * scalling,
                np.array(source.elevation) * scalling,
                data={"id": np.array(source.id), 'Z_elev': np.array(source.elevation) * scalling})


def vel_tovtk(file,file_init,scalling=1000,dx=250,dy=250,dz=250):
    def create_grid(X_coor, Y_coor, Z_coor,dx,dy,dz):
        X_i, X_f = min(X_coor) - dx, max(X_coor) + dx
        Y_i, Y_f = min(Y_coor) - dy, max(Y_coor) + dy
        Z_i, Z_f = min(Z_coor) - dz, max(Z_coor) + dz
        Xinter, Yinter, Zinter = np.mgrid[X_i:X_f+dx:dx, Y_i:Y_f+dy:dy, Z_i:Z_f+dz:dz]
        return Xinter.flatten(), Yinter.flatten(), Zinter.flatten(),Xinter.shape

    def in_hull(p, hull):
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        return hull.find_simplex(p) >= 0

    mod_init = pd.read_csv(file_init)
    mod_final = pd.read_csv(file)
    mod_init.X = mod_init.X * scalling
    mod_init.Y = mod_init.Y * scalling
    mod_init.Z = mod_init.Z * scalling
    mod_final.X = mod_final.X * scalling
    mod_final.Y = mod_final.Y * scalling
    mod_final.Z = mod_final.Z * scalling
    interpolp = RBFInterpolator(list(zip(mod_init.X, mod_init.Y,mod_init.Z)), mod_init.Vp, neighbors=8, kernel='linear')
    vel_init_p = interpolp(list(zip(mod_final.X, mod_final.Y, mod_final.Z)))
    interpols = RBFInterpolator(list(zip(mod_init.X, mod_init.Y, mod_init.Z)), mod_init.Vs, neighbors=8, kernel='linear')
    vel_init_s = interpols(list(zip(mod_final.X, mod_final.Y, mod_final.Z)))

    vel_pertub_p = (mod_final.Vp - vel_init_p) / vel_init_p * 100
    vel_pertub_s = (mod_final.Vs - vel_init_s) / vel_init_s * 100
    vpvs = mod_final.Vp / mod_final.Vs

    Xgrid, Ygrid, Zgrid, shape= create_grid(mod_final.X, mod_final.Y, mod_final.Z, dx,dy,dz)
    coor_flat = np.hstack((Xgrid.reshape(-1, 1), Ygrid.reshape(-1, 1), Zgrid.reshape(-1, 1)))

    #select point inside coordinate only
    coor_select = coor_flat[in_hull(coor_flat, np.array(mod_final[['X', 'Y', 'Z']]))]
    vpi = RBFInterpolator(list(zip(mod_final.X, mod_final.Y, mod_final.Z)), mod_final.Vp, neighbors=8,
                          kernel='linear')(coor_select)
    vsi = RBFInterpolator(list(zip(mod_final.X, mod_final.Y, mod_final.Z)), mod_final.Vs, neighbors=8,
                          kernel='linear')(coor_select)
    dvpi = RBFInterpolator(list(zip(mod_final.X, mod_final.Y, mod_final.Z)), vel_pertub_p, neighbors=8,
                          kernel='linear')(coor_select)
    dvsi = RBFInterpolator(list(zip(mod_final.X, mod_final.Y, mod_final.Z)), vel_pertub_s, neighbors=8,
                          kernel='linear')(coor_select)
    psi = RBFInterpolator(list(zip(mod_final.X, mod_final.Y, mod_final.Z)), vpvs, neighbors=8, kernel='linear')(
        coor_select)
    hcp = RBFInterpolator(list(zip(mod_final.X, mod_final.Y, mod_final.Z)), mod_final.hcP, neighbors=8,
                          kernel='linear')(coor_select)
    hcs = RBFInterpolator(list(zip(mod_final.X, mod_final.Y, mod_final.Z)), mod_final.hcS, neighbors=8,
                          kernel='linear')(coor_select)
    hcp[np.isnan(hcp)] = 0
    hcs[np.isnan(hcs)] = 0

    pointsToVTK(file+'_smooth.vtu', coor_select[:, 0].flatten(), coor_select[:, 1].flatten(),
                coor_select[:, 2].flatten(),
                data={"Vp(km/s)": vpi, "Vs(km/s)": vsi, "dVp(%)": dvpi, "dVs(%)": dvsi, "Vp/Vs": psi, "hcP": hcp, "hcS": hcs})  # "Vp/Vs":vpvs})

    pointsToVTK(file + '_ori_grid.vtu', np.array(mod_final.X), np.array(mod_final.Y), np.array(mod_final.Z),
                data={"Vp(km/s)": np.array(mod_final.Vp), "Vs(km/s)": np.array(mod_final.Vs), "dVp(%)": np.array(vel_pertub_p),
                      "dVs(%)":np.array(vel_pertub_s), "Vp/Vs": np.array(vpvs), "hcP": np.array(mod_final.hcP),
                      "hcS": np.array(mod_final.hcS)})  # "Vp/Vs":vpvs})


    vpi = RBFInterpolator(list(zip(mod_final.X, mod_final.Y, mod_final.Z)), mod_final.Vp, neighbors=8,
                          kernel='linear')(coor_flat)
    vsi = RBFInterpolator(list(zip(mod_final.X, mod_final.Y, mod_final.Z)), mod_final.Vs, neighbors=8,
                          kernel='linear')(coor_flat)
    dvpi = RBFInterpolator(list(zip(mod_final.X, mod_final.Y, mod_final.Z)), vel_pertub_p, neighbors=8,
                          kernel='linear')(coor_flat)
    dvsi = RBFInterpolator(list(zip(mod_final.X, mod_final.Y, mod_final.Z)), vel_pertub_s, neighbors=8,
                          kernel='linear')(coor_flat)
    psi = RBFInterpolator(list(zip(mod_final.X, mod_final.Y, mod_final.Z)), vpvs, neighbors=8, kernel='linear')(
        coor_flat)
    hcp = RBFInterpolator(list(zip(mod_final.X, mod_final.Y, mod_final.Z)), mod_final.hcP, neighbors=8,
                          kernel='linear')(coor_flat)
    hcs = RBFInterpolator(list(zip(mod_final.X, mod_final.Y, mod_final.Z)), mod_final.hcS, neighbors=8,
                          kernel='linear')(coor_flat)
    Xi= Xgrid.reshape(shape)
    Yi = Ygrid.reshape(shape)
    Zi = Zgrid.reshape(shape)
    vpgrid = vpi.reshape(shape)
    vsgrid = vsi.reshape(shape)
    dvpgrid = dvpi.reshape(shape)
    dvsgrid = dvsi.reshape(shape)
    vpvsgrid = psi.reshape(shape)
    hcpgrid = hcp.reshape(shape)
    hcsgrid = hcs.reshape(shape)
    gridToVTK(file + '_smooth', Xi, Yi, Zi,
              pointData={"Vp(km/s)": vpgrid, "Vs(km/s)": vsgrid, "dVp(%)": dvpgrid, "dVs(%)":dvsgrid,
                      "Vp/Vs": vpvsgrid, "hcP": hcpgrid, "hcS": hcsgrid})  # "Vp/Vs":vpvs})
