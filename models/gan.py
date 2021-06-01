import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Generator
class Generator(nn.Module):
    def __init__(self, kwargs):
        
        super().__init__() # Ez így elég vagy super(Generator, slef).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(kwargs['input_size'], kwargs['input_size']),
            nn.ReLU(),
            nn.Linear(kwargs['input_size'], kwargs['input_size']),
            nn.ReLU()
        )
        # initialize the weights
        self.net.apply(self.init_weights)

        
    def forward(self, input):
        return self.net(input)
    
    
    def init_weights(self, m): #hogy kell megoldani hogy ugyanaz legyen? Vagy ez így oké mert nem random?
        if type(m) == nn.Linear:
            nn.init.eye_(m.weight)
            m.bias.data.fill_(1.e-5)
            
# Discriminator
class Discriminator(nn.Module):
    def __init__(self, kwargs):
        super().__init__() # Ez így elég?
        self.net = nn.Sequential(
            nn.Linear((kwargs['input_size'], math.ceil(math.sqrt(kwargs['data_size']))),
            nn.ReLU(),
            nn.Linear(math.ceil(math.sqrt(kwargs['data_size'])), 1),
            nn.Sigmoid()
        )
            
        # initiliaze the weights
        self.net.apply(self.init_weights)

    def forward(self, input):
        return self.net(input)

    def init_weights(self, m): #Hogy kell megoldani hogy ugyanaz legyen?
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
            

def train_gan(model_cfg, data):
    
    dev = model_cfg['dev']
    
    loader_tr = model_cfg['loaders'][data]
            
    # create discriminator
    discriminator = model_cfg['discriminator']
    discriminator_optim = model_cfg['disc_optim']
    discriminator_criterion = model_cfg['loss']
    discriminator.to(dev) # ez kell?

    # Create generator
    generator = model_cfg['generator']
    generator_optim = model_cfg['gen_optim']
    generator_criterion = model_cfg['loss']
    generator.to(dev) #ez kell?
            
    for b in loader_tr:
        
        # Generate noise
        noise_size = len(b["x_data"]) # itt a batch size kellene nekem
        noise = np.random.uniform(0, 1, (int(noise_size), latent_size)) # Itt a latent size a b["x_data"].shape[1] ha jól értem? Lehet a normális eloszlás egyébként jobb lenne
        noise = torch.tensor(noise, dtype=torch.float32).to(dev)

        # Get training data
        data_batch, _ = data # ez volt eredetileg
        data_batch = data['x_data'] #???

        # Generate potential outliers
        generated_data = generator(noise)
            
            
        # Concatenate real data to generated data
        # X = torch.tensor(np.concatenate([data_batch, generated_data]), dtype=torch.float32)
        X = torch.cat((data_batch, generated_data))
        Y = torch.tensor(np.array([1] * batch_size + [0] * int(noise_size)),
                         dtype=torch.float32).unsqueeze(dim=1).to(dev)
        
        # Train discriminator
        # enable training mode
        discriminator.train()
        # getting the prediction
        discriminator_pred = discriminator(X)
        # compute the loss
        discriminator_loss = discriminator_criterion(discriminator_pred, Y)
        # reset the gradients to avoid gradients accumulation
        discriminator_optim.zero_grad()
        # compute the gradients of loss w.r.t weights
        discriminator_loss.backward(retain_graph=True)
        # update the weights
        discriminator_optim.step()

        # Train generator
        # create fake labels
        trick = torch.tensor(np.array([1] * noise_size), dtype=torch.float32).unsqueeze(dim=1).to(dev)
        discriminator.eval()  # freeze the discriminator

        generator.train()  # enable training mode for the generator
        generator_loss = generator_criterion(discriminator(generated_data), trick)
        generator_optim.zero_grad()
        generator_loss.backward(retain_graph=True)
        generator_optim.step()


        # unfreeze the discriminator's layers
        for param in discriminator.parameters():
            param.requires_grad = True
            
    return generator_loss, discriminator_loss
