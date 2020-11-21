import numpy as np
import torch
import time
from sklearn.metrics import f1_score,accuracy_score
from utils.Preprocessing import to_numpy



def print_summary(epoch, i, nb_batch, loss, batch_time,
                  average_loss, average_time, logging_mode):
    mode = logging_mode

    summary = '[' + logging_mode + '] Epoch: [{0}][{1}/{2}] '.format(
        epoch, i, nb_batch)

    string = ''
    string += ('Loss {:.4f} ').format(loss)
    string += ('(Average {:.4f}) \t').format(average_loss)
    string += ('Batch Time {:.4f} ').format(batch_time)
    string += ('(Average {:.4f}) \t').format(average_time)

    summary += string

    print(summary)


def to_var(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.to(device)
    return x


def train_loop(loader, model, criterion, optimizer, scheduler, epoch, device, batch_size, print_frequency):
    logging_mode = 'Train' if model.training else 'Val'

    end = time.time()
    time_sum, loss_sum = 0, 0
    metric_sum = 0

    for i, sample in enumerate(loader, 1):
        # Take variable and put them to GPU
        (images, labels) = sample
        images = to_var(images.float(), device)
        labels = to_var(labels.long(), device)
        # compute output
        pred_labels = model(images)
        # Entropy loss
        loss = criterion(pred_labels, labels)
        pred_labels = torch.argmax(pred_labels, dim=1)
        metric = accuracy_score(to_numpy(labels), to_numpy(pred_labels))

        # compute gradient and do SGD step
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            scheduler.step(loss)

        # measure elapsed time
        batch_time = time.time() - end

        time_sum += batch_size * batch_time
        loss_sum += batch_size * loss
        metric_sum += batch_size * metric
        average_loss = loss_sum / (i * batch_size)
        average_metric = metric_sum / (i * batch_size)
        average_time = time_sum / (i * batch_size)
        end = time.time()
        if i % print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), loss, batch_time,
                          average_loss, average_time, logging_mode)

            print('*****metrics equal to:', metric)
    return average_loss, average_metric

def train_model(model,train_loader,valid_loader,criterion ,optimizer,scheduler ,batch_size,epochs,device,print_frequency):
  # TRAINING

  print("***************************")
  model.to(device)
  train_losses = []
  train_metrics= []
  val_losses = []
  val_metrics = []
  for epoch in range(epochs):  # loop over the dataset multiple times
      print('******** Epoch [{}/{}]  ********'.format(epoch+1, epochs+1))

      # train for one epoch
      model.train()
      print('Training')
      loss , metric =train_loop(train_loader, model, criterion, optimizer,scheduler, epoch, device, batch_size,print_frequency)
      train_losses.append(loss)
      train_metrics.append(metric)

      # evaluate on validation set
      print('Validation')
      with torch.no_grad():
          model.eval()
          val_loss, val_metric = train_loop(valid_loader, model, criterion,
                                optimizer,scheduler, epoch, device, batch_size,print_frequency)
          val_losses.append(val_loss)
          val_metrics.append(val_metric)
  return train_losses,train_metrics,val_losses, val_metrics

def predict(model,loader,device):
  preds = []
  tar = []

  for i, sample in enumerate(loader, 1):
      # Take variable and put them to GPU
      (images, labels) = sample
      images = to_var(images.float(), device)
      labels = to_var(labels.long(), device)
      # compute output
      pred_labels = model.predict(images)
      pred_labels = torch.argmax(pred_labels,dim=1)
      preds.append(to_numpy(pred_labels))
      tar.append(to_numpy(labels))
  preds = np.array(preds).flatten()
  tar = np.array(tar).flatten()
  metric = accuracy_score(preds,tar)
  f1 =  f1_score(tar, preds,average = None)
  return preds, metric, f1