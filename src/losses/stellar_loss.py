import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STELLAR_Loss(nn.Module):
    def __init__(self, rec_coeff=None, reg_coeff=1.0):
        super(STELLAR_Loss, self).__init__()

        self.rec_coeff = rec_coeff
        self.reg_coeff = reg_coeff

    def forward(self, predictions, labels):

        # Reconstruction loss
        recon_loss = 0.0
        if "MSE" in predictions:
            recon_loss += predictions["MSE"]
            self.rec_coeff = 10 if self.rec_coeff is None else self.rec_coeff
        elif "VQ-CE" in predictions:
            recon_loss += predictions["VQ-CE"]
            self.rec_coeff = 0.05 if self.rec_coeff is None else self.rec_coeff
            

        reg_loss = 0.0
        for key in predictions:
            if "loss" in key:
                reg_loss += predictions[key]

        total_loss =  self.rec_coeff * recon_loss + self.reg_coeff * reg_loss
   
        return total_loss