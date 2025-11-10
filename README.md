# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset


## DESIGN STEPS

### STEP 1:
Load MNIST dataset and convert to tensors.

### STEP 2:
Apply Gaussian noise to images for training.

### STEP 3:
Design encoder-decoder architecture for reconstruction.

### STEP 4:
Use MSE loss to measure reconstruction quality.

### STEP 5:
Train autoencoder using Adam optimizer efficiently.

### STEP 6:
Evaluate model on noisy and clean images.

### STEP 7:
Visualize results comparing original, noisy, denoised versions.

### STEP 8:
Improve performance by tuning hyperparameters carefully.

## PROGRAM
### Name: NITHIYANERANJAN S
### Register Number: 212223040136
```py
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Define your layers here
        # Example:
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # For reconstruction, sigmoid is often used
        )
    def forward(self, x):
        # Include your code here
        x = x.view(-1, 28*28)  # Flatten the input image
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, 28, 28)  # Reshape to image dimensions
        return x

#Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
summary(model, (1, 28, 28))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in loader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
```



## OUTPUT

### Model Summary

<img width="752" height="550" alt="image" src="https://github.com/user-attachments/assets/339eab33-6f01-4df5-9b45-96e058d1d43f" />

### Original vs Noisy Vs Reconstructed Image
<img width="401" height="480" alt="image" src="https://github.com/user-attachments/assets/3bd4dbdf-16f2-4733-93c7-0968269be6ac" />

<img width="1776" height="646" alt="image" src="https://github.com/user-attachments/assets/8219f088-6881-4d79-8027-3821c513d249" />




## RESULT
A convolutional autoencoder for image denoising application is developed successfully.
