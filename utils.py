import torch
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def evaluate(model, data_loader, n_samples, criterion, device):
  mean_loss = 0
  mean_acc = 0

  for images, labels in data_loader:
    images = images.to(device)
    labels = labels.to(device)
    batch_size = images.shape[0]

    outputs = model(images)
    loss = criterion(outputs, labels)

    mean_loss += ((batch_size / n_samples) * loss).item()

    outputs = torch.argmax(outputs, dim=1)
    correct_batch = (outputs == labels).float().sum()

    mean_acc += ((batch_size / n_samples) 
                  * (correct_batch / batch_size)).item()

  return mean_loss, mean_acc


def split_dataset(dataset, ratio=0.5, seed=None):
  n_samples = len(dataset)
  n_samples_1 = int(ratio*n_samples)
  n_samples_2 = n_samples - n_samples_1

  return torch.utils.data.random_split(dataset, [n_samples_1, n_samples_2],
                                       generator=torch.Generator().manual_seed(seed))


def plot_history(history):
  fig, axs = plt.subplots(2, 1, figsize=(10, 10))
  fig.tight_layout()

  axs[0].plot('train accuracy', 'r--', label='acc', data=history)
  axs[0].plot('val accuracy', 'k--', label='val_acc', data=history)

  axs[1].plot('train loss', 'r', label = 'loss', data=history)
  axs[1].plot('val loss', 'k', label = 'val_loss', data=history)

  axs[0].legend() 
  axs[1].legend()
  plt.show()
  plt.savefig('history.jpg', dpi=300)