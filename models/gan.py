import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

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
            nn.Linear(kwargs['input_size'], math.ceil(math.sqrt(kwargs['data_size']))),
            nn.ReLU(),
            nn.Linear(math.ceil(math.sqrt(kwargs['data_size'])), 1),
            nn.Sigmoid()
        )


    def forward(self, input):
        return self.net(input)

    def init_weights(self, m): #Hogy kell megoldani hogy ugyanaz legyen?
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
            
class Gan(nn.Module):
    def __init__(self, kwargs):
        super().__init__() # Ez így elég vagy super(Generator, slef).__init__()
        
        self.generator = Generator(kwargs)
        self.discriminator = Discriminator(kwargs)

    def init_weights(self, m): #Hogy kell megoldani hogy ugyanaz legyen?
        self.generator.apply(self.generator.init_weights)
        self.discriminator.apply(self.discriminator.init_weights)

    def init_opt_and_loss(self, m, optimizer, loss, kwargs):
        m['gen_optimizer'] = optimizer(m['net'].parameters(), lr=kwargs['gen_lr'], momentum=kwargs['momentum'], dampening=kwargs['dampening'])
        m['gen_loss'] = loss(reduction='mean')
        m['disc_loss'] = loss(reduction='mean')
        m['disc_optimizer'] = optimizer(m['net'].parameters(), lr=kwargs['disc_lr'], momentum=kwargs['momentum'], dampening=kwargs['dampening'])


def train_gan(model_cfg, data):
    
    dev = model_cfg['dev']
    
    loader_tr = model_cfg['loaders'][data]
            
    # create discriminator
    discriminator = model_cfg['net'].discriminator
    discriminator_optim = model_cfg['disc_optimizer']
    discriminator_criterion = F.binary_cross_entropy

    # Create generator
    generator = model_cfg['net'].generator
    generator_optim = model_cfg['gen_optimizer']
    generator_criterion = F.binary_cross_entropy

            
    for b in loader_tr:
        
        # Generate noise
        noise_size = len(b["x_data"]) # itt a batch size kellene nekem
        noise = np.random.uniform(0, 1, (int(noise_size), b["x_data"].shape[1])) # Itt a latent size a b["x_data"].shape[1] ha jól értem? Lehet a normális eloszlás egyébként jobb lenne
        noise = torch.tensor(noise, dtype=torch.float32).to(dev)

        # Get training data
        data_batch = b['x_data'].to(dev) #???

        # Generate potential outliers
        generated_data = generator(noise)
            
            
        # Concatenate real data to generated data
        # X = torch.tensor(np.concatenate([data_batch, generated_data]), dtype=torch.float32)
        X = torch.cat((data_batch, generated_data))
        Y = torch.tensor(np.array([1] * int(noise_size) + [0] * int(noise_size)),
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

        # Train generator
        # create fake labels
        trick = torch.tensor(np.array([1] * noise_size), dtype=torch.float32).unsqueeze(dim=1).to(dev)

        generator.train()  # enable training mode for the generator
        generator_loss = generator_criterion(discriminator(generated_data), trick)
        generator_optim.zero_grad()

        # compute the gradients of loss w.r.t weights
        discriminator_loss.backward(retain_graph=True)

        discriminator.eval()  # freeze the discriminator
        generator_loss.backward(retain_graph=True)

        # unfreeze the discriminator's layers
        for param in discriminator.parameters():
            param.requires_grad = True

        # update the weights
        discriminator_optim.step()
        generator_optim.step()

            
    return (generator_loss + discriminator_loss) / 2

def validate_gan(model_cfg, data, kwargs):
    net = model_cfg['net'].discriminator
    loader = model_cfg['loaders'][data]
    dev = model_cfg['dev']
    loss = F.binary_cross_entropy
    net.eval()
    
    y_true_list = []
    y_hat_list = []
    pred_list = []
    
    with torch.no_grad():
        for b in loader:
            X = b["x_data"].to(dev)
            y_data = b["y_data"].view(-1).to(dev)
            y_true_list.append(y_data)
            output = net(X)
            # output = loss(x_hat, X).mean(dim=1).view(-1)
            y_hat_list.append(output)
            output_sorted = output.sort().values
            th_value = output_sorted[int(output.shape[0] * kwargs['anomaly_trhold']) - 1].item()
            pred_list.append(torch.where(output > th_value, 1.0, 0.0))
    
        y_true = torch.cat(y_true_list, dim=0).cpu().numpy()
        y_hat = torch.cat(y_hat_list, dim=0).cpu().numpy()
        preds = torch.cat(pred_list, dim=0).cpu().numpy()
    
        eq = y_true == preds
        accuracy = np.mean(eq)
    
        auc = roc_auc_score(y_true, y_hat)

        return {
            'accuracy': accuracy,
            'auc': auc,
        }

def predict_gan(model_cfg, data, kwargs):
    net = model_cfg['net'].discriminator
    loader = model_cfg['loaders'][data]
    dev = model_cfg['dev']
    loss = F.binary_cross_entropy
    net.eval()

    preds = []
    with torch.no_grad():
        for b in loader:
            X = b["x_data"].to(dev)
            output = net(X)
            # output = loss(x_hat, X).mean(dim = 1).view(-1)
            output_sorted = output.sort().values
            th_value = output_sorted[int(output.shape[0] * kwargs['anomaly_trhold']) - 1].item()
            preds.append(torch.where(output > th_value, 1.0, 0.0))

        preds = torch.cat(preds, dim=0).cpu().numpy()
    
    
    return preds
