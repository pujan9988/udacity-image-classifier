import torch
from torch import nn,optim 
from torch.utils.data import DataLoader
from torchvision import datasets,models
from torchvision.transforms import v2
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path


def arg_parse():
    parser = ArgumentParser(description="Image classifier")

    parser.add_argument('data_dir')
    parser.add_argument("--save_dir",help="directory to save checkpoints",
                        default="checkpoint.pth")
    parser.add_argument("--epochs",help="No of epochs to train",
                        type=int,default=5)
    parser.add_argument("--lr",help="Learning rate",type=float,default=0.001)
    parser.add_argument("--hidden_units",help="No of hidden units",
                        type=int,default=512)
    parser.add_argument("--output_units",help="No of output units",type=int,
                        default=102)
    parser.add_argument("--gpu",help="use GPU to train the model",default="cpu")

    parser.add_argument("--arch",help="Model architecture: write 1 for VGG and 2 for ResNET",type=int,default=2)

    return parser.parse_args()


def train_transform(train_dir):
    transforms = v2.Compose([v2.RandomRotation(30),
                            v2.RandomResizedCrop(224),
                            v2.RandomHorizontalFlip(),
                            v2.ToTensor(),
                            v2.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                                     ])
    
    train_data = datasets.ImageFolder(root= train_dir,transform= transforms)
    return train_data 

def val_test_transform(val_dir):
    transforms = v2.Compose([v2.Resize(256),
                            v2.CenterCrop(224),
                            v2.ToTensor(),
                            v2.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                            ])
    
    val_data = datasets.ImageFolder(root=val_dir,transform=transforms)
    return val_data 


def data_loader(data,batch_size=64,shuffle=True):
    return DataLoader(data,batch_size=batch_size,shuffle=shuffle)

def device_agnostic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device=="cuda":
        print(f"Using {device}")
    else:
        print(f"No cuda found ! Using cpu")
    return device 

def load_model(arch):
    if arch==1:
        model = models.vgg19(weights='DEFAULT')
    if arch==2:
        model = models.resnet101(weights='DEFAULT')
    else:
        raise ValueError("Invalid Option! Choose 1 for VGG or 2 for ResNet.")

    for param in model.parameters():
            param.requires_grad = False

    return model 

def modify_classifier(model,hidden_units,output_units):
    if hasattr(model,'classifier'):
        input_units = model.classifier[0].in_features
    else:
        input_units = model.fc.in_features

    classifier = nn.Sequential(
        nn.Linear(input_units,hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units,output_units),
        nn.LogSoftmax(dim=1)
    )
    return classifier 

def test_model(test_dataloader,device,model):

    accuracy = 0
    model.eval()
    with torch.inference_mode():
        for inputs,labels in test_dataloader:
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            ps = torch.exp(outputs)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        avg_accuracy = accuracy/len(test_dataloader)

    return avg_accuracy

def train_model(train_dataloader,val_dataloader,
                device,model,criterion,optimizer,epochs=5):
    
    for epoch in tqdm(range(epochs)):
        model.train()
        training_loss = 0
        for inputs,labels in train_dataloader:
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        avg_train_loss = training_loss/len(train_dataloader)

        model.eval()
        val_loss = 0
        accuracy=0
        with torch.inference_mode():
            for inputs,labels in val_dataloader:
                inputs,labels = inputs.to(device),labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs,labels)
                val_loss += loss.item()

                ps = torch.exp(outputs)
                _, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


        avg_val_loss = val_loss/len(val_dataloader)
        avg_accuracy = accuracy/len(val_dataloader)
        print()
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss} | Validation Loss: {avg_val_loss} | Accuracy: {avg_accuracy}")


    return model 

def save_checkpoint(model,class_to_idx,path,
                    arch,hidden_units,output_units):
    
    model.class_to_idx = class_to_idx
    

    checkpoint = {
        "state_dict":model.state_dict(),
        "class_to_idx": model.class_to_idx,
        "arch":arch,
        "hidden_units":hidden_units,
        "output_units":output_units,

    }
    torch.save(checkpoint,path)
    print("Model checkpoint saved successfully.")


def main():
    args = arg_parse()
    data_dir = Path(args.data_dir)
    save_path = args.save_dir
    epochs = args.epochs 
    lr = args.lr 
    hidden_units = args.hidden_units 
    output_units = args.output_units
    gpu = args.gpu 
    architecture = args.arch 

    train_dir = data_dir / "train"
    val_dir = data_dir / "valid"
    test_dir = data_dir / "test"

    train_dataset = train_transform(train_dir)
    val_dataset = val_test_transform(val_dir)
    test_dataset = val_test_transform(test_dir)

    train_dataloader = data_loader(train_dataset,batch_size=64,shuffle=True)
    val_dataloader = data_loader(val_dataset,batch_size=64,shuffle=False)
    test_dataloader = data_loader(test_dataset,batch_size=64,shuffle=False)

    if gpu:
        device = device_agnostic()



    model = load_model(architecture)

    classifier = modify_classifier(model,hidden_units,output_units)

    if hasattr(model,'classifier'):
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(),lr=lr)
    else:
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(),lr=lr)

    criterion = nn.NLLLoss()

    model.to(device)

    model = train_model(train_dataloader,val_dataloader,device,model,
                criterion,optimizer,epochs)
    
    acc = test_model(test_dataloader,device,model)

    print(f"The accuracy of the model on test set: {acc}")


    save_checkpoint(model,train_dataset.class_to_idx,save_path,
                    architecture,hidden_units,output_units)
    


if __name__ == "__main__":
    main()