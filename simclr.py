import torch
from resnet_model import ResNetModel
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss_function import NtXent
import os
import numpy as np

torch.manual_seed(0)

weight_decay = 10e-6
epochs = 100
load_weights_path = 'D:\\Projekte\\MSE\\SimCLR_Usage\\runs\\Mar31_22-09-49_LAPTOP-T1N7HK2E---'

class SimCLR:

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.number_of_training_steps = 0
        self.number_of_validation_steps = 0
        self.smallest_loss = np.inf
        self.tensorboard_writer = SummaryWriter()
        self.processor = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_function = NtXent(self.processor, self.batch_size)

    def step(self, res_net, xi, xj):
        # get base encoder + projection head
        hi, zi = res_net(xi)
        hj, zj = res_net(xj)
        # normalize
        zi = F.normalize(zi, dim=1)
        zj = F.normalize(zj, dim=1)
        # calculate loss
        loss = self.loss_function(zi, zj)
        return loss

    def train(self):
        train_loader, valid_loader = self.data.get_data_loaders()
        res_net = ResNetModel().to(self.processor)
        res_net = self.load_pre_trained_weights(res_net)
        adam_optimizer = torch.optim.Adam(res_net.parameters(), 3e-4, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam_optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

        model = os.path.join(self.tensorboard_writer.log_dir, 'model')
        if not os.path.exists(model):
            os.makedirs(model)

        for epoch_count in range(epochs):
            self.exec_training_steps(adam_optimizer, res_net, train_loader)
            self.validate_steps(epoch_count, model, res_net, valid_loader)
            if epoch_count >= 10:
                scheduler.step()  # warmup
            self.tensorboard_writer.add_scalar('cosine annealing lr decay', scheduler.get_lr()[0], global_step=self.number_of_training_steps)

    def validate_steps(self, epoch_counter, model_folder, res_net, valid_loader):
        if epoch_counter > 0:
            loss = self.validate(res_net, valid_loader)
            if loss < self.smallest_loss:
                self.smallest_loss = loss
                torch.save(res_net.state_dict(), os.path.join(model_folder, 'res_net.pth'))

            self.tensorboard_writer.add_scalar('validation loss', loss, global_step=self.number_of_validation_steps)
            self.number_of_validation_steps += 1

    def exec_training_steps(self, adam_optimizer, res_net, train_loader):
        for (xi, xj), _ in train_loader:
            adam_optimizer.zero_grad()
            xi = xi.to(self.processor)
            xj = xj.to(self.processor)
            loss = self.step(res_net, xi, xj)
            if self.number_of_training_steps % 50 == 0:  # Log every 50 steps
                self.tensorboard_writer.add_scalar('training loss', loss, global_step=self.number_of_training_steps)

            loss.backward()
            adam_optimizer.step()
            self.number_of_training_steps += 1

    def load_pre_trained_weights(self, res_net):
        try:
            model_folder = os.path.join(load_weights_path, 'model')
            state_dict = torch.load(os.path.join(model_folder, 'model.pth'))
            res_net.load_state_dict(state_dict)
        except FileNotFoundError:
            print("Start new training")
        return res_net

    def validate(self, res_net, valid_loader):
        # validation steps
        with torch.no_grad():
            res_net.eval()

            valid_loss = 0.0
            for counter, ((xi, xj), _) in enumerate(valid_loader):
                xi = xi.to(self.processor)
                xj = xj.to(self.processor)

                loss = self.step(res_net, xi, xj)
                valid_loss += loss.item()

            valid_loss /= counter
        res_net.train()
        return valid_loss
