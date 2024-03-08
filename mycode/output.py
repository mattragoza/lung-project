import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TrainingPlot(object):
    '''
    Interactive training plot.
    '''
    def __init__(self, out_name):
        
        # create subplots for loss and accuracy
        self.fig, ax = plt.subplots(1, 2, figsize=(8,4))
        ax[0].set_ylabel('u_loss')
        ax[1].set_ylabel('mu_loss')
        for ax_ in ax:
            ax_.set_axisbelow(True)
            ax_.grid(linestyle=':')
            ax_.set_xlabel('epoch')

        self.fig.tight_layout()
        
        # store data and artists for interactive ploting
        self.data = pd.DataFrame(columns=['epoch', 'phase', 'u_loss', 'mu_loss'])

        self.train_u_loss_line = ax[0].plot([], [], label='train')[0]
        self.test_u_loss_line  = ax[0].plot([], [], label='test')[0]
        self.train_mu_loss_line = ax[1].plot([], [], label='train')[0]
        self.test_mu_loss_line  = ax[1].plot([], [], label='test')[0]

        self.out_name = out_name

    def __len__(self):
        return len(self.data)

    def write(self):
        self.data.to_csv(self.out_name + '_metrics.csv', sep='\t', index=False)
        self.fig.savefig(self.out_name + '_training.png', bbox_inches='tight')
        
    def draw(self, pad=1e-8):
        ax = self.fig.get_axes()
        ax[0].set_xlim(0, self.data.epoch.max() * 1.1 + pad)
        ax[0].set_ylim(0, self.data.u_loss.max() * 1.1 + pad)
        ax[1].set_xlim(0, self.data.epoch.max() * 1.1 + pad)
        ax[1].set_ylim(0, self.data.mu_loss.max() * 1.1 + pad)
        self.fig.canvas.draw()
        
    def update_train(self, epoch, u_loss, mu_loss):
        self.data.loc[len(self)] = [epoch, 'train', u_loss.item(), mu_loss.item()]
        
        data = self.data.groupby(['phase', 'epoch']).mean()
        train = data.loc['train'].reset_index()
        if isinstance(train, pd.Series): # need > 1 rows
            return
        
        self.train_u_loss_line.set_xdata(train.epoch)
        self.train_u_loss_line.set_ydata(train.u_loss)

        self.train_mu_loss_line.set_xdata(train.epoch)
        self.train_mu_loss_line.set_ydata(train.mu_loss)

        self.draw()
        
    def update_test(self, epoch, u_loss, mu_loss):
        self.data.loc[len(self)] = [epoch, 'test', u_loss.item(), mu_loss.item()]
        
        data = self.data.groupby(['phase', 'epoch']).mean()
        test = data.loc['test'].reset_index()
        if isinstance(test, pd.Series): # need > 1 rows
            return
        
        self.test_u_loss_line.set_xdata(test.epoch)
        self.test_u_loss_line.set_ydata(test.u_loss) 
        
        self.test_mu_loss_line.set_xdata(test.epoch)
        self.test_mu_loss_line.set_ydata(test.mu_loss)
        
        self.draw()
