from tensorflow.keras.datasets import fashion_mnist
import torch
from sklearn.model_selection import train_test_split
from utils.Preprocessing import complex_preprocess, FashionDataset
from model.BayesianCNN import BayesianCNN
from utils.train_utils import train_model
from model.ActiveLearning import active_training

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

learning_rate = 1e-5
epochs = 100
batch_size = 32
print_frequency = 100
criterion = torch.nn.NLLLoss()
nb_sample = 10000
weight_decay = 0.00001

# Import the data from tensorflow
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preparing the DATA for the first training phase
X_train, X_valid, Y_train, Y_valid = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
train_dataset = FashionDataset(X_train[:nb_sample],Y_train[:nb_sample],transform = complex_preprocess())
valid_dataset = FashionDataset(X_valid,Y_valid,transform = complex_preprocess())
# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,sampler=None,
                                           drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=batch_size, shuffle=True,sampler=None,
                                           drop_last=True)
# initiate the model
init_model = BayesianCNN(dropout_prob=0.25,num_samples=5)
optimizer = torch.optim.Adam(init_model.parameters(), lr=learning_rate,weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=10,verbose=True)
# First TRAINING
train_losses, train_metrics,val_losses,val_metrics = train_model(init_model,train_loader,valid_loader,
                                                                 criterion,optimizer,scheduler,batch_size,epochs,device,print_frequency)

# Save and load the initial model
torch.save(init_model.state_dict(),'/content/initial_model.pth.tar')
init_model = BayesianCNN(dropout_prob=0.25,num_samples=5)
PATH =  '/content/drive/MyDrive/BayesianActiveLearning/initial_model.pth.tar'
init_model.load_state_dict(torch.load(PATH,map_location='cuda'),map)
init_model.to(device)

#ACTIVE TRAINING PHASE
x_train = X_train[:nb_sample]
y_train = Y_train[:nb_sample]
x_pool = X_train[nb_sample:]
y_pool = Y_train[nb_sample:]

learning_rate = 1e-5
batch_size=32
weight_decay = 0.00001
#Active learning
final_model , train_active_losses , train_active_metrics ,val_active_losses , val_active_metrics = active_training(init_model,x_train,y_train,x_pool,y_pool,10000
                                                                                                                   ,'RAND',3,90,learning_rate,batch_size,weight_decay,criterion)