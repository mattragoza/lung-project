import sys, os, argparse, time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

import mycode as code


@code.utils.main
def train(
    out_name='ASDF',
    pde_name='poisson',
    image_size=128,
    n_nodes=128,
    n_epochs=100,
    batch_size=128,
    learning_rate=1e-5,
):
    # configuration
    n_samples = 10000
    n_freqs = 32
    n_filters = 16
    kernel_size = 5
    activ_fn = 'leaky_relu'
    device = 'cuda'
    benchmark = True

    print('Initialize PDE solver')
    assert pde_name.lower() == 'poisson'
    pde_solver = code.pde.PDESolver(n_nodes)

    print('Generating PDE dataset')
    dataset = code.data.PDEDataset.generate(
        n_samples, n_freqs, image_size, pde_solver, device=device
    )

    print('Splitting train and val set')
    train_set, val_set = dataset.split(n=0.9)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=True)

    print('Initializing model and optimizer')
    model = code.model.PDENet(n_freqs, n_filters, kernel_size, activ_fn, pde_solver)
    model.to(device, dtype=torch.float64)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = F.mse_loss

    plot = code.output.TrainingPlot(out_name)

    print('Start training loop')
    for epoch in range(n_epochs):

        model.train()
        train_progress = tqdm(train_loader, file=sys.stdout)
        for i, (a, mu, u, ub) in enumerate(train_progress):

            t0 = code.utils.timer(benchmark)
            mu_hat, u_hat = model.forward(a, ub)
            t1 = code.utils.timer(benchmark)
            mu_loss = loss_fn(mu_hat, mu)
            u_loss  = loss_fn(u_hat, u)
            t2 = code.utils.timer(benchmark)
            u_loss.backward()
            t3 = code.utils.timer(benchmark)
            optim.step()
            t4 = code.utils.timer(benchmark)

            times = {
                't_model': t1 - t0,
                't_loss':  t2 - t1,
                't_grad':  t3 - t2,
                't_optim': t4 - t3,
            }
            plot.update_train(epoch + i/len(train_loader), u_loss, mu_loss, **times)
            train_progress.set_description(
                f'[Epoch {epoch + 1}/{n_epochs}|train] u_loss = {u_loss.item():.4f}, mu_loss = {mu_loss.item():.4f}'
            )

        model.eval()
        val_progress = tqdm(val_loader, file=sys.stdout)
        for i, (a, mu, u, ub) in enumerate(val_progress):

            with torch.no_grad():
                t0 = code.utils.timer(benchmark)
                mu_hat, u_hat = model.forward(a, ub)
                t1 = code.utils.timer(benchmark)
                mu_loss = loss_fn(mu_hat, mu)
                u_loss  = loss_fn(u_hat, u)
                t2 = code.utils.timer(benchmark)
                times = {
                    't_model': t1 - t0,
                    't_loss':  t2 - t1
                }
                plot.update_test(epoch + 1, u_loss, mu_loss, **times)
                val_progress.set_description(
                    f'[Epoch {epoch + 1}/{n_epochs}|eval] u_loss = {u_loss.item():.4f}, mu_loss = {mu_loss.item():.4f}'
                )

        plot.write()

        x = np.linspace(0, 1, image_size)
        u = code.utils.dofs_to_image(u, pde_solver.V, image_size)
        u_hat = code.utils.dofs_to_image(u_hat, pde_solver.V, image_size)

        code.output.plot1d(
            x,
            mu=mu[:4].detach().cpu().numpy(),
            mu_hat=mu_hat[:4].detach().cpu().numpy(),
            u=u[:4].detach().cpu().numpy(),
            u_hat=u_hat[:4].detach().cpu().numpy()
        ).savefig(out_name + f'_mu_epoch_{epoch}.png', bbox_inches='tight')


    print('Done')
