class CNN(nn.Module):   # réseau à 2 couche convolution + pooling et 3 couches linéaires 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*53*53, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 17)
        
    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16*53*53)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
#         x = F.log_softmax(x, dim=1)   # pas de fonction d'activation car déjà présente dans la fonction de perte
        return x