import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TrainingPlot(object):
    '''
    Interactive training plot.
    '''
    def __init__(self):
        
        # create subplots for loss and accuracy
        self.fig, ax = plt.subplots(1, 2, figsize=(8,4))
        ax[0].set_ylabel('u_loss')
        ax[1].set_ylabel('mu_loss')
        for ax_ in ax:
            ax_.set_axisbelow(True)
            ax_.grid(linestyle=':')
            ax_.set_xlabel('iteration')
        self.fig.tight_layout()
        
        # store data and artists for interactive ploting
        self.data = pd.DataFrame(columns=['iter', 'phase', 'u_loss', 'mu_loss'])

        self.train_u_loss_line = ax[0].plot([], [], label='train')[0]
        self.test_u_loss_line  = ax[0].plot([], [], label='test')[0]
        self.train_mu_loss_line = ax[1].plot([], [], label='train')[0]
        self.test_mu_loss_line  = ax[1].plot([], [], label='test')[0]
        
    def draw(self, pad=1e-8):
        ax = self.fig.get_axes()
        ax[0].set_xlim(0, self.data.iter.max() * 1.1 + pad)
        ax[0].set_ylim(0, self.data.u_loss.max() * 1.1 + pad)
        ax[1].set_xlim(0, self.data.iter.max() * 1.1 + pad)
        ax[1].set_ylim(0, self.data.mu_loss.max() * 1.1 + pad)
        self.fig.canvas.draw()
        
    def update_train(self, iteration, u_loss, mu_loss):
        self.data.loc[len(self.data)] = [
            iteration, 'train', u_loss.item(), mu_loss.item()
        ]
        
        data = self.data.groupby(['phase', 'iter']).mean()
        train = data.loc['train'].reset_index()
        if isinstance(train, pd.Series): # need > 1 rows
            return
        
        self.train_u_loss_line.set_xdata(train.iter)
        self.train_u_loss_line.set_ydata(train.u_loss)

        self.train_mu_loss_line.set_xdata(train.iter)
        self.train_mu_loss_line.set_ydata(train.mu_loss)

        self.draw()
        
    def update_test(self, iteration, u_loss, mu_loss):
        self.data.loc[len(self.data)] = [
            iteration, 'test', u_loss.item(), mu_loss.item()
        ]
        
        data = self.data.groupby(['phase', 'iter']).mean()
        test = data.loc['test'].reset_index()
        if isinstance(test, pd.Series): # need > 1 rows
            return
        
        self.test_u_loss_line.set_xdata(test.iter)
        self.test_u_loss_line.set_ydata(test.u_loss) 
        
        self.test_mu_loss_line.set_xdata(test.iter)
        self.test_mu_loss_line.set_ydata(test.mu_loss)
        
        self.draw()
