import numpy as np
import time
import torch
import torch.nn.functional as F

from . import (
    thin_plate_spline,
    foerstner,
    utils,
    mindssc,
    similarity,
    belief_propagation,
    graphs
)

#def compute_marginals(kpts_fix, img_fix, mind_fix, mind_mov, alpha, beta, disp_radius, disp_step, patch_radius):
#    cost = alpha * ssd(kpts_fix, mind_fix, mind_mov, disp_radius, disp_step, patch_radius)
#        
#    dist = kpts_dist(kpts_fix, img_fix, beta)
#    edges, level = minimum_spanning_tree(dist)
#    marginals = tbp(cost, edges, level, dist)
#    
#    return marginals


def compute_marginals(
    kpts_fix,
    img_fix,
    mind_fix,
    mind_mov,
    alpha,
    beta,
    disp_radius,
    disp_step,
    patch_radius
):
    cost = alpha * similarity.ssd(
        kpts_fix, mind_fix, mind_mov, disp_radius, disp_step, patch_radius
    )
    k = 32
    while True:
        k *= 2
        try:
            dist = graphs.kpts_dist(kpts_fix, img_fix, beta, k)
            edges, level = graphs.minimum_spanning_tree(dist)
            break
        except IndexError:
            pass

    marginals = belief_propagation.tbp(cost, edges, level, dist)
    return marginals


def corrfield(
    img_fix,
    mask_fix,
    img_mov,
    alpha=2.5,   # regularization parameter
    beta=150.0,  # intensity weighting factor
    gamma=5.0,   # scaling factor for soft correspondences  
    delta=1,     # step size for MIND descriptor
    lambd=0.0,   # regularization parameter for TPS
    sigma=1.4,   # sigma for Foerstner operator
    sigma1=1.0,  # sigma for MIND descriptor
    L=[16, 8],   # maximum search radius
    N=[6, 3],    # cube length for non-max suppression
    Q=[2, 1],    # quantization of search step size
    R=[3, 2],    # patch radius for similarity search
    T=['n', 'n'] # rigid(r) or non-rigid(n) transform
):
    '''
    Registration with CorrField.

    Args:
        img_fix: (1, 1, D, H, W) fixed image tensor
        mask_fix: (1, 1, D, H, W) fixed mask tensor
        img_mov: (1, 1, D, H, W) moving image tensor
        alpha: regularization parameter (default: 2.5)
        beta: intensity weighting factor (default: 150.0)
        gamma: scaling factor for soft correspondences (default: 5.0)
        delta: step size for MIND descriptor (default: 1)
        lambd: TPS regularization parameter (default: 0.0)
        sigma: sigma for Foerstner operator (default: 1.4)
        sigma1: sigma for MIND descriptor (default: 1.0)
        L: maximum search radius (default: [16, 8])
        N: cube length for non-max suppression (default: [6, 3])
        Q: quantization of search step size (default: [2, 1])
        R: patch radius for similarity search (default: [3, 2])
        T: rigid(r) or non-rigid(n) transform (default: ['n', 'n'])
    Returns:
        dense_flow: (1, D, H, W, 3) deformation tensor
        kpts_fix: (1, N, 3) fixed image keypoints
        kpts_fix_warped: (1, N, 3) warped keypoints
    '''
    device = img_fix.device
    _, _, D, H, W = img_fix.shape
    
    print('Compute fixed MIND features ...', end=" ")
    torch.cuda.synchronize()
    t0 = time.time()
    mind_fix = mindssc.mindssc(img_fix, delta, sigma1)
    torch.cuda.synchronize()
    t1 = time.time()
    print('finished ({:.2f} s).'.format(t1-t0))
        
    dense_flow = torch.zeros((1, D, H, W, 3), device=device)
    img_mov_warped = img_mov
    for i in range(len(L)):
        print('Stage {}/{}'.format(i + 1, len(L)))
        print('    search radius: {}'.format(L[i]))
        print('      cube length: {}'.format(N[i]))
        print('     quantisation: {}'.format(Q[i]))
        print('     patch radius: {}'.format(R[i]))
        print('        transform: {}'.format(T[i]))
        
        disp = utils.get_disp(Q[i], L[i], (D, H, W), device=device)
        
        print('    Compute moving MIND features ...', end=" ")
        torch.cuda.synchronize()
        t0 = time.time()
        mind_mov = mindssc.mindssc(img_mov_warped, delta, sigma1)
        torch.cuda.synchronize()
        t1 = time.time()
        print('finished ({:.2f} s).'.format(t1-t0))
        
        torch.cuda.synchronize()
        t0 = time.time()
        kpts_fix = foerstner.foerstner_kpts(img_fix, mask_fix, sigma, N[i])
        torch.cuda.synchronize()
        t1 = time.time()
        print('    {} fixed keypoints extracted ({:.2f} s).'.format(kpts_fix.shape[1], t1-t0))

        print('    Compute forward marginals ...', end =" ")
        torch.cuda.synchronize()
        t0 = time.time()
        marginalsf = compute_marginals(kpts_fix, img_fix, mind_fix, mind_mov, alpha, beta, L[i], Q[i], R[i])
        torch.cuda.synchronize()
        t1 = time.time()
        print('finished ({:.2f} s).'.format(t1-t0))

        flow = (
            F.softmax(-gamma * marginalsf.view(1, kpts_fix.shape[1], -1, 1), dim=2) *
            disp.view(1, 1, -1, 3)
        ).sum(2)
        
        kpts_mov = kpts_fix + flow

        print('    Compute symmetric backward marginals ...', end =" ")
        torch.cuda.synchronize()
        t0 = time.time()
        marginalsb = compute_marginals(kpts_mov, img_fix, mind_mov, mind_fix, alpha, beta, L[i], Q[i], R[i])
        torch.cuda.synchronize()
        t1 = time.time()
        print('finished ({:.2f} s).'.format(t1-t0))

        marginals = 0.5 * (marginalsf.view(1, kpts_fix.shape[1], -1) + marginalsb.view(1, kpts_fix.shape[1], -1).flip(2))
        
        flow = (F.softmax(-gamma * marginals.view(1, kpts_fix.shape[1], -1, 1), dim=2) * disp.view(1, 1, -1, 3)).sum(2)
        
        torch.cuda.synchronize()
        t0 = time.time()
        if  T[i] == 'r':
            print('    Find rigid transform ...', end =" ")
            rigid = utils.compute_rigid_transform(kpts_fix, kpts_fix + flow)
            dense_flow_ = F.affine_grid(rigid[:, :3, :] - torch.eye(3, 4, device=device).unsqueeze(0), (1, 1, D, H, W), align_corners=True)
        elif T[i] == 'n':
            print('    Dense thin plate spline interpolation ...', end =" ")
            dense_flow_ = thin_plate_spline.thin_plate_dense(kpts_fix, flow, (D, H, W), 3, lambd)
        torch.cuda.synchronize()
        t1 = time.time()
        print('finished ({:.2f} s).'.format(t1-t0))
        
        dense_flow += dense_flow_
        
        img_mov_warped = F.grid_sample(
            img_mov,
            F.affine_grid(
                torch.eye(3, 4, dtype=img_mov.dtype, device=device).unsqueeze(0),
                (1, 1, D, H, W),
                align_corners=True
            ) + dense_flow.to(img_mov.dtype),
            align_corners=True
        )

        print()
        
    flow = F.grid_sample(
        dense_flow.permute(0, 4, 1, 2, 3),
        kpts_fix.view(1, 1, 1, -1, 3),
        align_corners=True
    ).view(1, 3, -1).permute(0, 2, 1)
    
    ##dense_flow1 = F.affine_grid(torch.eye(3, 4, dtype=img_mov.dtype, device=device).unsqueeze(0), (1, 1, D, H, W), align_corners=True) + dense_flow.to(img_mov.dtype)

    return (
        utils.flow_world(
            dense_flow.view(1, -1, 3), (D, H, W), align_corners=True
        ).view(1, D, H, W, 3),
        utils.kpts_world(kpts_fix, (D, H, W), align_corners=True),
        utils.kpts_world(kpts_fix + flow, (D, H, W), align_corners=True)
    )
    #return img_mov_warped, kpts_world(kpts_fix, (D, H, W), align_corners=True), kpts_world(kpts_fix + flow, (D, H, W), align_corners=True)
