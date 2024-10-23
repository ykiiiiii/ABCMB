'''
source from Jan Hamann, edited by Kai Yi
'''

import os
import time
import argparse
import numpy as np
import torch
import healpy as hp
import lenspyx
from lenspyx.utils import camb_clfile
import multiprocessing
from typing import Tuple, Dict
import gc

class CMBDataGenerator:
    def __init__(self, 
                 cls_path: str = 'input_spectra',
                 output_dir: str = 'output/train/all_map/',
                 lmax: int = 4096,
                 dlmax: int = 1024,
                 nside: int = 2048,
                 output_size: int = 224,
                 map_size_deg: float = 160):
        """
        Initialize CMB data generator.
        
        Args:
            cls_path: Path to input spectra files
            lmax: Maximum multipole of the lensed field
            dlmax: Buffer for accurate lensing
            nside: HEALPix nside parameter
            output_size: Output map size in pixels
            map_size_deg: Output map size in square degrees
        """
        self.cls_path = cls_path
        self.lmax = lmax
        self.dlmax = dlmax
        self.nside = nside
        self.output_size = output_size
        self.map_size_deg = map_size_deg
        self.output_dir = output_dir
        
        # Load power spectra
        self.cl_r000 = self._load_cls('planck_2018_r000_lenspotentialCls.dat')
        self.cl_r010 = self._load_cls('planck_2018_r010_nt0_lenspotentialCls.dat')
        self.rstar = 0.1  # Reference value, do not change
        
        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _load_cls(self, filename: str) -> Dict:
        """Load power spectra from file."""
        return camb_clfile(os.path.join(self.cls_path, filename))
    
    def _get_power_spectra(self, r: float) -> Dict:
        """
        Get power spectra for a given r value.
        
        Args:
            r: Tensor-to-scalar ratio
        Returns:
            Dictionary containing power spectra
        """
        cl = self._load_cls('planck_2018_r000_lenspotentialCls.dat')
        
        # Interpolate between r=0 and r=0.1 spectra
        for i in ('tt', 'ee', 'bb', 'te'):
            cl[i] = self.cl_r000[i] + r/self.rstar*(self.cl_r010[i] - self.cl_r000[i])
            
        # Add zero cross-power spectra
        for i in ('tb', 'eb', 'pb'):
            cl[i] = np.zeros_like(cl['tt'])
            
        return cl
        
    def generate_maps(self, r: float, batch_id: int) -> Tuple[torch.Tensor, ...]:
        """
        Generate CMB maps for a given r value.
        
        Args:
            r: Tensor-to-scalar ratio
            batch_id: Batch identifier for seeding
        Returns:
            Tuple of Q, U, B unlensed, and B lensed maps
        """
        # Set seed based on r value and batch_id as in original code
        seed = int(r*500) + batch_id*1000
        np.random.seed(seed)
        
        # Get power spectra
        cl = self._get_power_spectra(r)
        
        # Generate random realization
        alm = hp.synalm((cl['pp'], cl['tt'], cl['ee'], cl['bb'], cl['pt'], cl['te'],
                        cl['eb'], cl['pe'], cl['tb'], cl['pb']), 
                       lmax=self.lmax + self.dlmax, 
                       new=True)
        
        # Generate maps
        Bunl = hp.alm2map(alm[3], self.nside)  # B mode
        Pmap = hp.alm2map(alm[0], self.nside)  # Lensing potential
        
        # Generate lensed maps
        dlm = hp.almxfl(alm[0], np.sqrt(np.arange(self.lmax+1) * np.arange(1, self.lmax+2)))
        Red, Imd = hp.alm2map_spin([dlm, np.zeros_like(dlm)], self.nside, 1, 
                                  hp.Alm.getlmax(dlm.size))
        
        # Get lensed Q and U maps
        Qlen, Ulen = lenspyx.alm2lenmap_spin([alm[2], alm[3]], [Red, Imd], 
                                            self.nside, 2, facres=0, verbose=False)
        
        # Convert to B mode
        elm, blm = hp.map2alm_spin([Qlen, Ulen], 2, lmax=self.lmax)
        Blen = hp.alm2map(blm, self.nside)
        
        return self._process_maps(Bunl, Qlen, Ulen, Blen)
    
    def _process_maps(self, Bunl, Qlen, Ulen, Blen) -> Tuple[torch.Tensor, ...]:
        """Process full-sky maps into patches."""
        nside_proj = self._get_projection_nside()
        npatch = hp.nside2npix(nside_proj)
        
        B_all, Q_all, U_all, B_len_all = [], [], [], []
        
        for j in range(npatch):
            rotation = hp.pix2ang(nside_proj, j, lonlat=True)
            x = np.sqrt(self.map_size_deg)
            
            # Project maps
            patches = {}
            patches['B_unl'] = hp.cartview(Bunl, rot=rotation, lonra=[0, x], 
                                         latra=[-x/2, x/2], xsize=self.output_size, 
                                         return_projected_map=True,)
            patches['Q_len'] = hp.cartview(Qlen, rot=rotation, lonra=[0, x], 
                                         latra=[-x/2, x/2], xsize=self.output_size, 
                                         return_projected_map=True)
            patches['U_len'] = hp.cartview(Ulen, rot=rotation, lonra=[0, x], 
                                         latra=[-x/2, x/2], xsize=self.output_size, 
                                         return_projected_map=True)
            patches['B_len'] = hp.cartview(Blen, rot=rotation, lonra=[0, x], 
                                         latra=[-x/2, x/2], xsize=self.output_size, 
                                         return_projected_map=True)
            
            # Convert to tensors and reshape
            B_all.append(torch.from_numpy(patches['B_unl']).view(-1, self.output_size, self.output_size))
            Q_all.append(torch.from_numpy(patches['Q_len']).view(-1, self.output_size, self.output_size))
            U_all.append(torch.from_numpy(patches['U_len']).view(-1, self.output_size, self.output_size))
            B_len_all.append(torch.from_numpy(patches['B_len']).view(-1, self.output_size, self.output_size))
        
        # Stack all patches
        return (torch.cat(Q_all), torch.cat(U_all), 
                torch.cat(B_all), torch.cat(B_len_all))
    
    def _get_projection_nside(self) -> int:
        """Calculate appropriate nside for projection."""
        nside_proj = 1
        while (hp.nside2pixarea(nside_proj, degrees=True) > 2*4*self.map_size_deg):
            nside_proj *= 2
        return nside_proj

def generate_batch(params: Tuple[float, float, int, int,str] ) -> None:
    """
    Generate a batch of CMB maps.
    
    Args:
        params: Tuple of (min_r, max_r, num_samples, batch_id)
    """
    min_r, max_r, num_samples, batch_id,output_path = params
    generator = CMBDataGenerator()
    
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    for r in np.linspace(min_r, max_r, num_samples):
        # Use original filename format
        filename = f'r = {r:.4f}<start>160_sqdeg_224x224_pix_{batch_id}.pt'
        
        if filename not in os.listdir(output_path):
            maps = generator.generate_maps(r, batch_id)
            # Stack maps in original format: [Q, U, B_unl, B_len]
            all_data = torch.cat([m.unsqueeze(0) for m in maps])
            torch.save(all_data, os.path.join(output_path, filename))
            print(f'Generated maps for r={r:.4f}, batch {batch_id}')
        else:
            print(f'File {filename} already exists, skipping')
        
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description='CMB Map Generator')
    parser.add_argument('--p', type=int, default=79, help='Batch offset (default: 79)')
    parser.add_argument('--num_samples', type=int, default=1,
                      help='Number of r values to sample between min and max')
    parser.add_argument('--output_path', type=str, default='output/train/all_map/',
                      help='Output directory for generated maps')
    args = parser.parse_args()
    
    # Define parameter ranges for parallel processing
    # Using original r ranges
    param_ranges = [
        (0.0, 0.04, args.num_samples, args.p,args.output_path),
        (0.002, 0.12, args.num_samples, args.p, args.output_path),
    ]
    
    # Run parallel processes
    start_time = time.time()
    processes = []
    for params in param_ranges:
        p = multiprocessing.Process(target=generate_batch, args=(params,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print(f'Generation completed in {time.time() - start_time:.2f} seconds')

if __name__ == '__main__':
    main()