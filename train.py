import torch
import glob

import os

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from UNet_model import UNet
from UNet_GAN_model import UNetGen, UNetDis
from GAN_model import NetGen, NetDis
from ResNet_model import ResNet

from colorize_data import ColorizeData


class Trainer:
    def __init__(self, train_paths, val_paths, epochs, batch_size, learning_rate, num_workers=2):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, model = UNet(), name='UNet'):             
        train_dataset = ColorizeData(paths=self.train_paths)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True)
        model = model.to(self.device)
        criterion = torch.nn.MSELoss(reduction='mean').to(self.device)
        optimizer = torch.optim.Adam(model.parameters(),lr=self.learning_rate, weight_decay=1e-6)

        Loss = []
        val_Loss = []
       

        # train loop
        for epoch in range(self.epochs):
            print("Starting Training Epoch " + str(epoch + 1))
            avg_loss = 0.0
            model.train()
            for i, data in enumerate(tqdm(train_dataloader)):                                                    #(train_dataloader, 0)?
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()                                                                       # Initialize optimizer 

                outputs = model(inputs)                                                                     # forward prop

                loss = torch.sqrt(criterion(outputs, targets))
                loss.backward()                                                                             # back prop
                optimizer.step()                                                                            # Update the weights

                avg_loss += loss.item()
            
            Loss.append(avg_loss/len(train_dataloader))

            print(f'Epoch {epoch + 1} \t\t Training Loss: {avg_loss / len(train_dataloader)}')
            
            if (epoch + 1) % 1 == 0:
                val_loss, val_len = self.validate(model, criterion)
                val_Loss.append(val_loss / val_len)
                print(f'Epoch {epoch + 1} \t\t Validation Loss: {val_loss / val_len}')

            p = os.getcwd()
            os.makedirs(p+'\\weights', exist_ok=True)     
            torch.save(model.state_dict(), p+'\\weights\\'+name+'_weights.pth')

        return Loss, val_Loss


    def validate(self, model, criterion):
        # Validation loop begin
        # ------
        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            val_dataset = ColorizeData(paths=self.val_paths)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
            for i, data in enumerate(val_dataloader):
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)

                loss = torch.sqrt(criterion(outputs, targets))

                valid_loss += loss.item()
        # model.train()
        return valid_loss, len(val_dataloader)

class GAN_Trainer:
    def __init__(self, train_paths, val_paths, epochs, batch_size, learning_rate, num_workers):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.train_paths = train_paths
        self.val_paths = val_paths        
        self.real_label = 1
        self.fake_label = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self, NetG=NetGen(), NetD=NetDis(), name = "GAN"):             
        train_dataset = ColorizeData(paths=self.train_paths)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True, drop_last = True)
        # Model
        
        model_G = NetG.to(self.device)
        model_D = NetD.to(self.device)
        model_G.apply(self.weights_init)
        model_D.apply(self.weights_init)

        optimizer_G = torch.optim.Adam(model_G.parameters(),
                             lr=self.learning_rate, betas=(0.5, 0.999),
                             eps=1e-8, weight_decay=0)
        optimizer_D = torch.optim.Adam(model_D.parameters(),
                             lr=self.learning_rate, betas=(0.5, 0.999),
                             eps=1e-8, weight_decay=0)
        
        criterion = nn.BCELoss()
        L1 = nn.L1Loss()

        model_G.train()
        model_D.train()

        Loss = []
        val_Loss = []

        # train loop
        for epoch in range(self.epochs):
            print("Starting Training Epoch " + str(epoch + 1))
            for i, data in enumerate(tqdm(train_dataloader)):                                                    #(train_dataloader, 0)?
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                model_D.zero_grad()
                label = torch.full((self.batch_size,), self.real_label, dtype=torch.float, device=self.device)
                output = model_D(targets)
                errD_real = criterion(torch.squeeze(output), label)
                errD_real.backward()

                fake = model_G(inputs)
                label.fill_(self.fake_label)
                output = model_D(fake.detach())
                errD_fake = criterion(torch.squeeze(output), label)
                errD_fake.backward()
                errD = errD_real + errD_fake
                optimizer_D.step()

                model_G.zero_grad()
                label.fill_(self.real_label)
                output = model_D(fake)
                errG = criterion(torch.squeeze(output), label)
                errG_L1 = L1(fake.view(fake.size(0),-1), targets.view(targets.size(0),-1))
                errG = errG + 100 * errG_L1
                errG.backward()
                optimizer_G.step()   
            Loss.append([errD.cpu().detach().numpy() / len(train_dataloader),errG.cpu().detach().numpy() / len(train_dataloader)])

            print(f'Training: Epoch {epoch + 1} \t\t Discriminator Loss: {errD / len(train_dataloader)}  \t\t Generator Loss: {errG / len(train_dataloader)}')
            
            if (epoch + 1) % 1 == 0:
                errD_val, errG_val, val_len = self.validate(model_D, model_G, criterion, L1)
                val_Loss.append([errD_val.cpu().numpy() / val_len, errG_val.cpu().numpy() / val_len])
                print(f'Validation: Epoch {epoch + 1} \t\t Discriminator Loss: {errD_val / val_len}  \t\t Generator Loss: {errG_val / val_len}')

            p = os.getcwd()
            os.makedirs(p+'\\weights', exist_ok=True) 
            os.makedirs(p+'\\weights\\'+name, exist_ok=True) 
            torch.save(model_G.state_dict(), p+'/weights/'+name+'/Generator.pth')
            torch.save(model_D.state_dict(), p+'/weights/'+name+'/Discriminator.pth')
        return Loss, val_Loss


    def validate(self, model_D, model_G, criterion, L1):
        model_G.eval()
        model_D.eval()
        with torch.no_grad():
            valid_loss = 0.0
            val_dataset = ColorizeData(paths=self.val_paths)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last = True)
            for i, data in enumerate(val_dataloader):
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                label = torch.full((self.batch_size,), self.real_label, dtype=torch.float, device=self.device)
                output = model_D(targets)
                errD_real = criterion(torch.squeeze(output), label)

                fake = model_G(inputs)
                label.fill_(self.fake_label)
                output = model_D(fake.detach())
                errD_fake = criterion(torch.squeeze(output), label)
                errD = errD_real + errD_fake

                label.fill_(self.real_label) 
                output = model_D(fake)
                errG = criterion(torch.squeeze(output), label)
                errG_L1 = L1(fake.view(fake.size(0),-1), targets.view(targets.size(0),-1))
                errG = errG + 100* errG_L1

        return errD, errG, len(val_dataloader)


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.realpath(__file__))

    parent_directory = os.path.dirname(current_directory)
    paths = np.array([])
    training_set_path = os.path.join(parent_directory, 'Dataset', 'training_set')
    for p in os.listdir(training_set_path):
        new_path = os.path.join(training_set_path,p)
        n_paths = np.array(glob.glob( new_path+ "/*.jpg"))
        paths = np.concatenate([paths,n_paths])
    
    val_paths = np.array([])
    validation_set_path = os.path.join(parent_directory, 'Dataset', 'validation_set')
    for p in os.listdir(validation_set_path):
        new_path = os.path.join(validation_set_path,p)
        n_paths = np.array(glob.glob( new_path+ "/*.jpg"))
        val_paths = np.concatenate([paths,n_paths])

    train_indices = np.random.permutation(len(paths))
    train_paths = paths[train_indices]
    val_indices = np.random.permutation(len(val_paths))
    val_paths = val_paths[val_indices]

    trainer = Trainer(train_paths, val_paths, epochs = 200, batch_size = 64, learning_rate = 0.01, num_workers = 2)
    trainer.train()