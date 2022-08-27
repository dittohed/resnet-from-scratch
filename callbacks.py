import torch 


class EarlyStopping():
  def __init__(self, patience=15):
    self._patience = patience
    self._best_acc = 0
    self._counter = 0
    self.stop_training = False
    
  def callback(self, acc):
    if acc <= self._best_acc:
      self._counter += 1
      if self._counter == self._patience:
        print(f'Accuracy has not increased for {self._patience} epochs, terminating...')
        self.stop_training = True
    else:
      self._best_acc = acc
      self._counter = 0


class Checkpoint():
  def __init__(self, path, model, optimizer):
    self._path = path
    self._best_acc = 0
    self._model = model
    self._optimizer = optimizer

  def save(self):
    dict_to_save = {
        'state_dict': self._model.state_dict(),
        'optimizer': self._optimizer.state_dict()}
    torch.save(dict_to_save, self._path)
    print(f'Successfully saved state dicts to {self._path}.')

  def load(self):
    dict_to_load = torch.load(self._path)
    self._model.load_state_dict(dict_to_load['state_dict'])
    self._optimizer.load_state_dict(dict_to_load['optimizer'])
    print(f'Successfully loaded state dicts from {self._path}.')

  def callback(self, acc):
    if acc > self._best_acc:
      print(f'Accuracy has increased from {self._best_acc:.2f} to {acc:.2f}.')
      self.save()
      self._best_acc = acc