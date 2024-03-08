import sys, os, argparse, time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

import mycode as code



def train(out_name, pde_name, image_size, n_nodes):

    # configuration
    n_samples = 10000
    n_freqs = 32
    n_filters = 16
    kernel_size = 5
    activ_fn = 'leaky_relu'
    learning_rate = 1e-5
    batch_size = 128
    n_epochs = 2
    device = 'cuda'

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
            mu_hat, u_hat = model.forward(a, ub)
            mu_loss = loss_fn(mu_hat, mu)
            u_loss  = loss_fn(u_hat, u)
            u_loss.backward()
            optim.step()
            plot.update_train(epoch + i/len(train_loader), u_loss, mu_loss)
            train_progress.set_description(
                f'[Epoch {epoch + 1}/{n_epochs}|train] u_loss = {u_loss.item():.4f}, mu_loss = {mu_loss.item():.4f}'
            )

        model.eval()
        val_progress = tqdm(val_loader, file=sys.stdout)
        for i, (a, mu, u, ub) in enumerate(val_progress):
            with torch.no_grad():
                mu_hat, u_hat = model.forward(a, ub)
                mu_loss = loss_fn(mu_hat, mu)
                u_loss  = loss_fn(u_hat, u)
                plot.update_test(epoch + 1, u_loss, mu_loss)
                val_progress.set_description(
                    f'[Epoch {epoch + 1}/{n_epochs}|eval] u_loss = {u_loss.item():.4f}, mu_loss = {mu_loss.item():.4f}'
                )

        plot.write()

        code.output.plot1d(
            np.linspace(0, 1, image_size),
            mu=mu[:4].detach().cpu().numpy(),
            mu_hat=mu_hat[:4].detach().cpu().numpy(),
            u=u[:4].detach().cpu().numpy(),
            u_hat=u_hat[:4].detach().cpu().numpy()
        ).savefig(out_name + f'_epoch_{epoch}.png', bbox_inches='tight')

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_name', type=str, default='ASDF')
    parser.add_argument('--pde_name', type=str, default='Poisson')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--n_nodes', type=int, default=128)
    args = parser.parse_args()
    train(args.out_name, args.pde_name, args.image_size, args.n_nodes)
