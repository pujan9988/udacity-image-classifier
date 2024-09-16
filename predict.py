from argparse import ArgumentParser
import torch 
from torchvision import models 
from torchvision.transforms import v2 
from PIL import Image 
import json 
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from train import load_model,device_agnostic,modify_classifier

def arg_parser():

    parser = ArgumentParser(description="Image classifier prediction")

    parser.add_argument('img_dir')
    parser.add_argument("trained_model_dir",help="Enter your trained model path")
    parser.add_argument("--topk",help="Return top K most probable classes",type=int,default=3)
    parser.add_argument("--label_names",help="uses class to real label names",default="cat_to_name.json")
    parser.add_argument("--gpu",help="use GPU for inference",default="cpu")


    return parser.parse_args()


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'),weights_only=True)
    architecture = checkpoint["arch"]
    hidden_units = checkpoint["hidden_units"]
    output_units = checkpoint["output_units"]

    model = load_model(architecture)
    

    classifier = modify_classifier(model,hidden_units,output_units)
    
    if hasattr(model,'classifier'):
        model.classifier = classifier 
    else:
        model.fc = classifier


    model.load_state_dict(checkpoint["state_dict"])

    model.class_to_idx = checkpoint["class_to_idx"]

    
    return model 

def preprocess_image(image):

    transforms = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])
    
    img = Image.open(image)
    tensor_img = transforms(img)
    return tensor_img


def predict(model,img_path,topk,device):
    img_input = preprocess_image(img_path)
    img_input = img_input.unsqueeze(dim=0) #expect batch size too so adding another dimension
    img_input = img_input.to(device)
    model.eval()
    with torch.inference_mode():
        output = model(img_input)

        probab_scores = torch.exp(output)
        top_scores,top_classes = probab_scores.topk(topk,dim=1)

    return top_scores,top_classes

def main():

    args = arg_parser()

    img_dir = Path(args.img_dir)
    trained_model = Path(args.trained_model_dir)
    gpu = args.gpu 
    topk = args.topk
    labels_name = args.label_names

    with open(labels_name, 'r') as label_file:
        cat_to_name = json.load(label_file)

    model = load_checkpoint(trained_model)

    if gpu:
        device = device_agnostic()

    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        _,top_classes = predict(model,img_dir,topk,device)

        top_classes = top_classes[0].tolist()
        idx_to_classes = {value:key for key,value in model.class_to_idx.items()}
        top_classes = [idx_to_classes[x] for x in top_classes]
        top_classes_names = [cat_to_name[i] for i in top_classes]
        print(top_classes)
        print(top_classes_names)

if __name__ == "__main__":
    main()
