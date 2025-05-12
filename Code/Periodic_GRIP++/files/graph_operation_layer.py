import torch
import torch.nn as nn

# f_graph(nctq) = f_conv(nctp) (einstein summation over p) G(npq)
class GraphOperation(nn.Module):
    def __init__(self, particles):
        super(GraphOperation, self).__init__()
        
        assert type(particles) == int
        
        self.PARTICLES = particles
        self.DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
        #self.DEVICE = "cpu"
        G0 = torch.randn((self.PARTICLES, self.PARTICLES), requires_grad=True, device=self.DEVICE)
        G1 = torch.randn((self.PARTICLES, self.PARTICLES), requires_grad=True, device=self.DEVICE)
        self.G0_train = nn.Parameter(G0)
        self.G1_train = nn.Parameter(G1)
        
    def forward(self, x, G_fixed):
        G = G_fixed + self.G0_train + self.G1_train
        x = torch.einsum('nctp, npq->nctq', (x, G))
        return x