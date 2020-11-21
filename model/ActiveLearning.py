import numpy as np
from utils.train_utils import to_var, to_numpy, train_model
import torch
from utils.Preprocessing import FashionDataset, complex_preprocess
from model.BayesianCNN import BayesianCNN

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def select_from_pool(model, D_train, N, nb_samples, aquis='BALD'):
    """Return indices of the N most uncertain predictions
    """

    if aquis == 'BALD':
        uncertainties = []
        for i, sample in enumerate(D_train, 1):
            # Take variable and put them to GPU
            (images, _) = sample
            images = to_var(images.float(), device)
            # compute uncertainty
            uncertainty = model.uncertainty(images)
            uncertainties.append(to_numpy(uncertainty))
        uncertainties = np.array(uncertainties).flatten()
        m = uncertainties.mean()
        indices = []
        new_x_train = []
        new_y_train = []
        for k in range(N):
            j = np.argmax(uncertainties)
            uncertainties[j] = -np.inf
            indices.append(j)
        return indices, m
    elif aquis == 'RAND':
        indices = np.random.choice(nb_samples, N)
        return indices, None

def active_training(init_model,X_train,y_train,X_pool,y_pool,valid_loader,N,aquis,iterations,epochs,learning_rate,batch_size,weight_decay,criterion):
  """Train model on new dataset after the aquisition of N data points from the new data
  """
  train_active_losses = []
  train_active_metrics = []
  val_active_losses = []
  val_active_metrics = []
  model = init_model

  for k in range(iterations):

    new_train_dataset = FashionDataset(X_pool,y_pool,transform = complex_preprocess())
    # DataLoaders
    new_train_loader = torch.utils.data.DataLoader(new_train_dataset,
                                              batch_size=batch_size, shuffle=False,sampler=None,
                                              drop_last=True)
    # We set model to train to activate the dropout layers
    model.train()
    print('*************')
    print('Extract training images from the pool with high uncertainty')
    indices,mean = select_from_pool(model,new_train_loader,N,X_pool.shape[0],aquis=aquis)
    print('mean uncertainty:',mean)
    print('Number of extracted training images:',len(indices))
    X_train = np.concatenate((X_train,X_pool[indices]),axis=0)
    y_train = np.concatenate((y_train,y_pool[indices]),axis=0)
    print('Size of new training dataset',np.shape(X_train))
    X_pool = np.delete(X_pool,indices,axis=0)
    y_pool = np.delete(y_pool,indices,axis=0)
    new_train_dataset = FashionDataset(X_train,y_train,transform = complex_preprocess())
    # DataLoaders
    new_train_loader = torch.utils.data.DataLoader(new_train_dataset,
                                              batch_size=batch_size, shuffle=True,sampler=None,
                                              drop_last=True)
    model = BayesianCNN(dropout_prob=0.25,num_samples=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.9,patience=5000,verbose=True)
    # TRAINING
    print('New training has began')
    new_train_losses, new_train_metrics,new_val_losses,new_val_metrics = train_model(model,new_train_loader,valid_loader,
                                                                  criterion,optimizer,scheduler,batch_size,epochs,device)
    filename = '/content/drive/MyDrive/BayesianActiveLearning/model_'+aquis + '_' +str(k)+'.pth.tar'
    torch.save(model.state_dict(),filename)
    print('Saving the model to ',filename)
    train_active_losses = train_active_losses + new_train_losses
    train_active_metrics = train_active_metrics + new_train_metrics
    val_active_losses = val_active_losses + new_val_losses
    val_active_metrics = val_active_metrics + new_val_metrics

  return model , train_active_losses , train_active_metrics ,val_active_losses , val_active_metrics