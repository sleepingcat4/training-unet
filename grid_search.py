from sklearn.model_selection import GridSearchCV
import torch.optim as optim

hyperparameters = {
    'lr': [0.001, 0.01, 0.1],
    'weight_decay': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64]
}

model = UNet(num_classes=10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters, scoring='accuracy', cv=3)

grid_search.fit(input_image, targets)

print("Best Hyperparameters:", grid_search.best_params_)
