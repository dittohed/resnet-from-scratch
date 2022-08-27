import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

from resnet import resnet18
from callbacks import EarlyStopping, Checkpoint
from utils import split_dataset, evaluate, plot_history


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(),
        transforms.RandomHorizontalFlip()
        ])
    
    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        transform=train_transforms, 
                                        download=True)
    val_test_ds = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        transform=transforms.ToTensor())
    val_ds, test_ds = split_dataset(val_test_ds, seed=42)

    train_loader = torch.utils.data.DataLoader(train_ds, 
                                               batch_size=config['batch_size'], 
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, 
                                               batch_size=config['batch_size'], 
                                               shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, 
                                               batch_size=config['batch_size'], 
                                               shuffle=False)

    model = resnet18().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    early_stopping = EarlyStopping()
    checkpoint = Checkpoint('./resnet_cifar.pth.tar', model, optimizer)  
    n_epochs = config['n_epochs']
    history = {
        'acc': [], 'val_acc': [],
        'loss': [], 'val_loss': []
        }
    
    for epoch in range(n_epochs):
        n_correct = 0
        tqdm_it = tqdm(train_loader, total=len(train_loader), 
                       leave=True)
   
        model.train()
        for images, labels in tqdm_it:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                outputs = torch.argmax(outputs, dim=1)
                correct_batch = (outputs == labels).float().sum().item()
                n_correct += correct_batch
                accuracy = correct_batch / labels.shape[0]

            tqdm_it.set_description(f'Epoch:  [{epoch}/{n_epochs}]')
            tqdm_it.set_postfix(loss=loss.item(), acc=accuracy.item())

        model.eval()
        with torch.no_grad():
            train_loss, train_accuracy = evaluate(model, train_loader, 
                                              len(train_ds), criterion, device) 
            print(f'Training loss: {train_loss:.2f}, accuracy: {train_accuracy:.2f}')
            history['acc'].append(train_accuracy)
            history['loss'].append(train_loss)

            val_loss, val_accuracy = evaluate(model, val_loader, 
                                              len(val_ds), criterion, device)            
            print(f'Validation loss: {val_loss:.2f}, accuracy: {val_accuracy:.2f}')
            history['val_acc'].append(val_accuracy)
            history['val_loss'].append(val_loss)

        early_stopping.callback(val_accuracy)
        checkpoint.callback(val_accuracy)
        if early_stopping.stop_training:
            break

    plot_history(history)


if __name__ == '__main__':
    config = {
        'n_epochs': 50,
        'batch_size': 32,
        'lr': 1e-3}

    main(config)