import matplotlib.pyplot as plt
import numpy as np
import torch
import healpy as hp
import os
import time
import lenspyx
from lenspyx.utils import camb_clfile
import gc
import argparse
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main(minv, maxv, num, times):
    # work_dir = os.getcwd()
    # start = time.time()
    cls_path = 'input_spectra'
    lmax = 4096  # target maximum multipole of the lensed field
    # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
    dlmax = 1024
    nside = 2048  # The lensed alm's are computed with healpy map2alm from a lensed map at resolution 'nside'
    facres = 0
    # print(filename_list)
    # Desired size of output maps in square degrees
    A_map = 160
    x = np.sqrt(A_map)

    # Desired number of pixels of output maps (out_pix x out_pix)
    out_pix = 224

    cl_file_r000 = 'planck_2018_r000_lenspotentialCls.dat'
    # Angular power spectra for r = rstar = 0.1, assuming tensor spectral index n_t = 0.
    cl_file_r010 = 'planck_2018_r010_nt0_lenspotentialCls.dat'
    rstar = 0.1  # do not change; this is not a variable

    cl_r000 = camb_clfile(os.path.join(cls_path, cl_file_r000))
    cl_r010 = camb_clfile(os.path.join(cls_path, cl_file_r010))

    for ttt in range(times):
        ttt += 1
        ttt += ppp
        print('ttt is equal to', ttt)
        for r in np.linspace(minv, maxv, num):

            
            filename = 'r = ' + format(r, '0.4f') + f'<start>160_sqdeg_224x224_pix_{ttt}.pt'
            # filename_list = []
            if filename not in filename_list:
                seed = int(r*500)+ttt*1000
                np.random.seed(seed)

                print(f'r is equal to {r}')
                # Initialise final array
                cl = camb_clfile(os.path.join(cls_path, cl_file_r000))
                for i in ('tt', 'ee', 'bb', 'te'):
                    cl[i] = cl_r000[i] + r/rstar*(cl_r010[i] - cl_r000[i])
                gc.collect()

                # Define the vanishing cross-power spectra (TB, EB, PB) for completeness
                cl['tb'] = np.zeros_like(cl['tt'])
                cl['eb'] = np.zeros_like(cl['tt'])
                cl['pb'] = np.zeros_like(cl['tt'])
                # Generate a random realisation of T, E, B and P, taking into account the correct correlations
                alm = hp.synalm((cl['pp'], cl['tt'], cl['ee'], cl['bb'], cl['pt'], cl['te'],
                                cl['eb'], cl['pe'], cl['tb'], cl['pb']), lmax=lmax + dlmax, new=True)
                (p, t, e, b) = (0, 1, 2, 3)

                # Construct unlensed T, E, B maps and lensing potential map
                Tunl = hp.alm2map(alm[t], nside)
                Eunl = hp.alm2map(alm[e], nside)
                Bunl = hp.alm2map(alm[b], nside)
                # print(Bunl.shape)
                # np.save('Bunl.npy', Bunl)
                Pmap = hp.alm2map(alm[p], nside)

                dlm = hp.almxfl(alm[p], np.sqrt(
                    np.arange(lmax+1, dtype=float) * np.arange(1, lmax+2)))
                Red, Imd = hp.alm2map_spin(
                    [dlm, np.zeros_like(dlm)], nside, 1, hp.Alm.getlmax(dlm.size))
                # Compute lensed temperature map
                # Tlen = lenspyx.alm2lenmap(
                #     alm[t], [Red, Imd], nside, facres=facres, verbose=False)
                # Compute lensed Q and U maps
                Qlen, Ulen = lenspyx.alm2lenmap_spin(
                    [alm[e], alm[b]], [Red, Imd], nside, 2, facres=facres, verbose=False)
                # print(Qlen.shape)
                # np.save('Qlen.npy', Qlen)
                elm, blm = hp.map2alm_spin([Qlen, Ulen], 2, lmax=lmax)
                Elen = hp.alm2map(elm, nside)
                Blen = hp.alm2map(blm, nside)
                nside_proj = 1
                if (hp.nside2pixarea(nside_proj, degrees=True) < A_map):
                    print(
                        'Warning! A_map is too large! Cannot produce more than one independent patch per full-sky simulation.')
                while (hp.nside2pixarea(nside_proj, degrees=True) > 2*4*A_map):
                    nside_proj = nside_proj*2

                npatch = hp.nside2npix(nside_proj)
                out_path = 'output/inference_map/'
                B_all = torch.tensor([])
                Q_all = torch.tensor([])
                U_all = torch.tensor([])
                B_len_all = torch.tensor([])
                for j in range(npatch):  # npatch = 48
                    # Cartesian projection
                    rotation = hp.pix2ang(nside_proj, j, lonlat=True)
                    # Tunl_patch = hp.cartview(Tunl, rot=rotation, lonra=[0,x], latra=[-x/2,x/2], xsize=out_pix, return_projected_map=True)
                    # Eunl_patch = hp.cartview(Eunl, rot=rotation, lonra=[0,x], latra=[-x/2,x/2], xsize=out_pix, return_projected_map=True)
                    Bunl_patch = hp.cartview(Bunl, rot=rotation, lonra=[
                        0, x], latra=[-x/2, x/2], xsize=out_pix, return_projected_map=True)
                    # Tlen_patch = hp.cartview(Tlen, rot=rotation, lonra=[0,x], latra=[-x/2,x/2], xsize=out_pix, return_projected_map=True)
                    # Elen_patch = hp.cartview(Elen, rot=rotation, lonra=[0,x], latra=[-x/2,x/2], xsize=out_pix, return_projected_map=True)
                    Blen_patch = hp.cartview(Blen, rot=rotation, lonra=[0,x], latra=[-x/2,x/2], xsize=out_pix, return_projected_map=True)
                    Qlen_patch = hp.cartview(Qlen, rot=rotation, lonra=[
                        0, x], latra=[-x/2, x/2], xsize=out_pix, return_projected_map=True)
                    Ulen_patch = hp.cartview(Ulen, rot=rotation, lonra=[
                        0, x], latra=[-x/2, x/2], xsize=out_pix, return_projected_map=True)
                    # Pmap_patch = hp.cartview(Pmap, rot=rotation, lonra=[0,x], latra=[-x/2,x/2], xsize=out_pix, return_projected_map=True)
                    outsuffix = '<start>' + str(A_map) + '_sqdeg_' + str(out_pix) + 'x' + str(
                        out_pix) + '_pix_patch_' + str(j) + '_' + str(ttt) + '.txt'
                    # Save maps to disk
                    # np.savetxt(os.path.join(out_path,  'T_unlensed_' , 'r = ' + format(r,'0.4f') + outsuffix), Tunl_patch)
                    # np.savetxt(os.path.join(out_path,  'E_unlensed_' , 'r = ' + format(r,'0.4f') + outsuffix), Eunl_patch)
                    # np.savetxt(os.path.join(out_path,  'B_unlensed_',
                    #                         'r = ' + format(r, '0.4f') + outsuffix), Bunl_patch)
                    # # np.savetxt(os.path.join(out_path,  'T_lensed_'   , 'r = ' + format(r,'0.4f') + outsuffix), Tlen_patch)
                    # # np.savetxt(os.path.join(out_path,  'E_lensed_'   , 'r = ' + format(r,'0.4f') + outsuffix), Elen_patch)
                    # # np.savetxt(os.path.join(out_path,  'B_lensed_'   , 'r = ' + format(r,'0.4f') + outsuffix), Blen_patch)
                    # np.savetxt(os.path.join(out_path,  'Q_lensed_',
                    #                         'r = ' + format(r, '0.4f') + outsuffix), Qlen_patch)
                    # np.savetxt(os.path.join(out_path,  'U_lensed_',
                    #                         'r = ' + format(r, '0.4f') + outsuffix), Ulen_patch)
                    # np.savetxt(os.path.join(out_path,  'P_'          , 'r = ' + format(r,'0.4f') + outsuffix), Pmap_patch)
                    B_all = torch.cat(
                        (torch.from_numpy(Bunl_patch).view(-1, 224, 224), B_all))
                    Q_all = torch.cat(
                        (torch.from_numpy(Qlen_patch).view(-1, 224, 224), Q_all))
                    U_all = torch.cat(
                        (torch.from_numpy(Ulen_patch).view(-1, 224, 224), U_all))
                    B_len_all = torch.cat(
                        (torch.from_numpy(Blen_patch).view(-1, 224, 224), B_len_all))

                all_data = torch.cat(
                    (Q_all.unsqueeze(0), U_all.unsqueeze(0), B_all.unsqueeze(0), B_len_all.unsqueeze(0)))
                new_filename = 'r = ' + \
                    format(r, '0.4f') + \
                    f'<start>160_sqdeg_224x224_pix_{ttt}.pt'
                torch.save(all_data, out_path+'all_map/'+new_filename)
            else:
                print(filename, 'already exist!')


parser = argparse.ArgumentParser(description='simulation')
parser.add_argument('--p', type=int, default=0, metavar='N',
                    help='input batch size for training (default: 128)')

if __name__ == '__main__':
    args = parser.parse_args()
    ppp = args.p
    s2 = time.time()
    # for ppp in [10, 11]:
    # if ppp in [40, 41]:
    #     filename_list = os.listdir('output/val_map/all_map/')
    # else:
    if not os.path.exists('output/inference_map/all_map/'):
        os.makedirs('output/inference_map/all_map/')
    filename_list = os.listdir('output/inference_map/all_map/')
    # main(0.2, 0.24, 3, 1)
    # break
    # print('It takes '+ str(s2-s1) +'second for one generation')
    import multiprocessing
    num = 1
    times = 1
    # main(0.002, 0.04, num, times)
    p1 = multiprocessing.Process(
        target=main, args=(0.002, 0.04, num, times))
    p2 = multiprocessing.Process(
        target=main, args=(0.042, 0.08, num, times))
    # p3 = multiprocessing.Process(
    #     target=main, args=(0.082, 0.12, num, times))
    # p4 = multiprocessing.Process(
    #     target=main, args=(0.122, 0.16, num, times))
    # p5 = multiprocessing.Process(
    #     target=main, args=(0.162, 0.20, num, times))
    # p6 = multiprocessing.Process(
    #     target=main, args=(0.202, 0.24, num, times))
    # p7 = multiprocessing.Process(
    #     target=main, args=(0.242, 0.28, num, times))
    # p8 = multiprocessing.Process(
    #     target=main, args=(0.282, 0.32, num, times))
    # p9 = multiprocessing.Process(
    #     target=main, args=(0.322, 0.36, num, times))
    # p10 = multiprocessing.Process(
    #     target=main, args=(0.362, 0.40, num, times))
    # p11 = multiprocessing.Process(
    #     target=main, args=(0.402, 0.44, num, times))
    # p12 = multiprocessing.Process(
    #     target=main, args=(0.442, 0.48, num, times))
    p1.start()
    p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    # p7.start()
    # p8.start()
    # p9.start()
    # p10.start()
    # p11.start()
    # p12.start()
    p1.join()
    p2.join()
    # p3.join()
    # p4.join()
    # p5.join()
    # p6.join()
    # p7.join()
    # p8.join()
    # p9.join()
    # p10.join()
    # p11.join()
    # p12.join()
    s3 = time.time()

    print('It takes ' + str(s3-s2) + 'second for three generation')
    # The dataset size is 48*num*times
