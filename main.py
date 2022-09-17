import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

from resnet import resnet50
from callbacks import EarlyStopping, Checkpoint
from utils import split_dataset, evaluate, plot_history


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        transform=train_transforms, 
                                        download=True)
    val_test_ds = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        transform=test_transforms)
    val_ds, test_ds = split_dataset(val_test_ds, seed=42)

    train_loader = torch.utils.data.DataLoader(train_ds, 
                                               batch_size=config['batch_size'], 
                                               shuffle=True,
                                               num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, 
                                               batch_size=config['batch_size'], 
                                               shuffle=False,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, 
                                               batch_size=config['batch_size'], 
                                               shuffle=False,
                                               num_workers=4)

    model = resnet50().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'],
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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

            tqdm_it.set_description(f'Epoch: [{epoch+1}/{n_epochs}]')
            tqdm_it.set_postfix(loss=loss.item(), acc=accuracy)

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

        scheduler.step()
        early_stopping.callback(val_accuracy)
        checkpoint.callback(val_accuracy)
        if early_stopping.stop_training:
            break
            
    plot_history(history)


if __name__ == '__main__':
    config = {
        'n_epochs': 200,
        'batch_size': 128,
        'lr': 0.1}

    main(config)