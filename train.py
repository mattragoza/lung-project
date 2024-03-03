import sys, os, argparse, time
from tqdm import tqdm
import torch

import code


def train(image_size, n_nodes):

    # configuration
    n_samples = 10000
    n_freqs = 32
    n_filters = 16
    kernel_size = 5
    activ_fn = 'leaky_relu'
    learning_rate = 1e-5
    batch_size = 100
    n_epochs = 10

    print('Initialize PDE solver')
    pde_solver = code.pde.PDESolver(n_nodes)

    print('Generating PDE dataset')
    dataset = code.data.PDEDataset.generate(n_samples, n_freqs, image_size, pde_solver)

    print('Splitting train and val set')
    train_set, val_set = dataset.split(n=0.9)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=True)

    print('Initializing model and optimizer')
    model = code.model.PDENet(n_freqs, n_filters, kernel_size, activ_fn, pde_solver)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = F.mse_loss

    plot = code.output.TrainingPlot()

    print('Start training loop')
    for epoch in range(n_epochs):
        print(f'[Epoch {epoch+1}/{n_epochs}]')

        print('Training...')
        model.train()
        for i, (a, mu, u, ub) in enumerate(tqdm(train_loader, file=sys.stdout)):
            mu_hat, u_hat = model.forward(a, ub)
            mu_loss = loss_fn(mu_hat, mu)
            u_loss  = loss_fn(u_hat, u)
            u_loss.backward()
            optim.step()
            plot.update_train(iteration, u_loss, mu_loss)

        print('Evaluating...')
        model.eval()
        for i, (a, mu, u, ub) in enumerate(tqdm(val_loader, file=sys.stdout)):
            with torch.no_grad():
                mu_hat, u_hat = model.forward(a, ub)
                mu_loss = loss_fn(mu_hat, mu)
                u_loss  = loss_fn(u_hat, u)
                plot.update_test(iteration, u_loss, mu_loss)

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_nodes', type=int, default=128)
    args = parser.parse_args()
    train(args.n_nodes)
