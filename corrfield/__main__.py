#!/usr/bin/env python

import sys, os, argparse, time
import numpy as np
import nibabel as nib
import torch

from . import corrfield, mindssc


def main(args):
    
    print('Run corrField registration ...')
    print()
    
    device = 'cuda:0'
    
    img_fix_path = args.fixed
    img_mov_path = args.moving
    mask_fix_path = args.mask
    output_path = args.output
    #segment_path = args.segment


    alpha = float(args.alpha)
    beta = float(args.beta)
    gamma = float(args.gamma)
    delta = int(args.delta)
    lambd = float(args.lambd)
    sigma = float(args.sigma)
    sigma1 = float(args.sigma1)
    
    print(' Fixed image: {}'.format(img_fix_path))
    print('Moving image: {}'.format(img_mov_path))
    print('  Fixed mask: {}'.format(mask_fix_path))
#    print('Moving segm.: {}'.format(segment_path))
    print('Output files: {}.csv/.nii.gz'.format(output_path))
    print('       alpha: {}'.format(alpha))
    print('        beta: {}'.format(beta))
    print('       gamma: {}'.format(gamma))
    print('      lambda: {}'.format(lambd))
    print('       delta: {}'.format(delta))
    print('       sigma: {}'.format(sigma))
    print('      sigma1: {}'.format(sigma1))
    print()
    
    L = [int(l) for l in args.search_radius.split('x')]
    N = [int(n) for n in args.length.split('x')]
    Q = [int(q) for q in args.quantisation.split('x')]
    R = [int(r) for r in args.patch_radius.split('x')]
    T = [t for t in args.transform.split('x')]
    
    mindssc.mindssc(torch.zeros(nib.load(img_fix_path).shape).unsqueeze(0).unsqueeze(0).to(device), delta, sigma1) # init GPU???
    
    img_fix = torch.from_numpy(nib.load(img_fix_path).get_fdata().astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    img_mov = torch.from_numpy(nib.load(img_mov_path).get_fdata().astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    mask_fix = torch.from_numpy(nib.load(mask_fix_path).get_fdata().astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    #seg_mov = torch.from_numpy(nib.load(segment_path).get_fdata().astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)


    torch.cuda.synchronize()
    t0 = time.time()
    
    img_mov_warped, kpts_fix, kpts_mov_corr = corrfield.corrfield(img_fix, mask_fix, img_mov, alpha, beta, gamma, delta, lambd, sigma, sigma1, L, N, Q, R, T)
    
    torch.cuda.synchronize()
    t1 = time.time()
    
    np.savetxt('{}.csv'.format(output_path), torch.cat([kpts_fix[0], kpts_mov_corr[0]], dim=1).cpu().numpy(), delimiter=",", fmt='%.3f')
    #seg_mov_warped = F.grid_sample(seg_mov, dense_flow1, align_corners=True,mode='nearest').squeeze().short()


    nib.save(nib.Nifti1Image(img_mov_warped.cpu().numpy(), np.eye(4)), '{}.nii.gz'.format(output_path))
    
    print('Files {}.csv and {}.nii.gz written.'.format(output_path, output_path))
    print('Total computation time: {:.1f} s'.format(t1-t0))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='corrField registration')

    parser.add_argument('-F',  '--fixed',         required='True', help="fixed image (*.nii/*.nii.gz)")
    parser.add_argument('-M',  '--moving',        required='True', help="moving image (*.nii/*.nii.gz)")
    parser.add_argument('-m',  '--mask',          required='True', help="mask for fixed image (*.nii/*.nii.gz)")
    parser.add_argument('-O',  '--output',        required='True', help="output name (no filename extension)")
    #parser.add_argument('-S',  '--segment',        required='True', help="input name moving segmentation labels")


    parser.add_argument('-a',  '--alpha',         default=2.5,     help="regularisation parameter (default: 2.5)")
    parser.add_argument('-b',  '--beta',          default=150,     help="intensity weighting (default: 150)")
    parser.add_argument('-g',  '--gamma',         default=5,       help="scaling factor for soft correspondeces (default: 5)")
    parser.add_argument('-d',  '--delta',         default=1,       help="step size for mind descriptor (default: 1)")
    parser.add_argument('-l',  '--lambd',         default=0,       help="regularistion parameter for TPS (default: 0)")
    parser.add_argument('-s',  '--sigma',         default=1.4,     help="sigma for foerstner operator (default: 1.4)")
    parser.add_argument('-s1', '--sigma1',        default=1,       help="sigma for mind descriptor (default: 1)")
    
    parser.add_argument('-L',  '--search_radius', default="16x8",  help="maximum search radius for each level (default: 16x8)")
    parser.add_argument('-N',  '--length',        default="6x3",   help="cube-length of non-maximum suppression (default: 6x3)")
    parser.add_argument('-Q',  '--quantisation',  default="2x1",   help="quantisation of search step size (default: 2x1)")
    parser.add_argument('-R',  '--patch_radius',  default="3x2",   help="patch radius for similarity seach (default: 3x2)")
    parser.add_argument('-T',  '--transform',     default="nxn",   help="rigid(r)/non-rigid(n) (default: nxn)")
    
    args = parser.parse_args()

    main(args)
