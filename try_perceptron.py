import torch
import torch.nn as nn
import torch.optim as optim
#事实证明，自动微分是非常重要的
input=torch.randn(6,20)
targets=torch.randn(6,1)
print(input.mean())

class perceptron(nn.Module):
    def __init__(self,inputdim=10,outputdim=1):
        super().__init__()
        self.fc=nn.Linear(inputdim,outputdim)
    def forward(self,input):
        output=self.fc(input)
        return output
model=perceptron(20,1)
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
loss_record=[]
for epoch in range(100):
    model.zero_grad()
    output=model(input)
    loss=criterion(output,targets)
    loss_record.append(loss)
    loss.backward()
    optimizer.step()
print(model(input))
print(loss_record)


