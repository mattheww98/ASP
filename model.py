import torch.nn as nn

class ASPModel(nn.Module):
    def __init__(self,
                 in_dims=132, 
                 hidden_dims=[1024,512,256,128],
                 out_dims = 40,
                 compute_device=None
                ):
        super().__init__()
        self.input_dims = in_dims
        self.hidden_dims = hidden_dims
        self.out_dims = out_dims
        self.compute_device = compute_device
        dims = [in_dims] + hidden_dims
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.acts = nn.ModuleList([nn.LeakyReLU() for j in range(len(dims) - 1)])
        self.fc_out = nn.Linear(dims[-1], out_dims)

    def forward(self, fea):
        for fc, act in zip(self.fcs, self.acts):
            fea = act(fc(fea))
        output = self.fc_out(fea)
        return output

if __name__ == "__main__":
    model = ASPModel()
