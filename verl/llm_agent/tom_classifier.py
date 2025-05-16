from torch import nn
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], num_classes=5, dropout_rate=0):
        super(MLPClassifier, self).__init__()
        self.layers = nn.ModuleList()
        
        # 添加输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))
        
        # 添加隐藏层
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        self.layers.append(nn.Linear(hidden_dims[-1], num_classes))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x