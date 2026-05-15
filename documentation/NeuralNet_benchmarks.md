## This document contains some notes about how different neural network achitectures performed in terms of accuracy (%)

`
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
`

*Accuracy* **73%**

`
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 16),
    nn.LeakyReLU(),
    nn.Linear(16, 1)
)
`

*Accuracy* **64%**
