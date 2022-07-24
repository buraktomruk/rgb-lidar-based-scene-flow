import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
def resnet_feature_extractor():
    seq1 = Image.open("/mmdetection3d/mmdet3d/models/voxel_encoders/nuscenes_first.png").convert('RGB')
    seq2 = Image.open("/mmdetection3d/mmdet3d/models/voxel_encoders/nuscenes_second.png").convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor1 = preprocess(seq1)
    seq1_batch = input_tensor1.unsqueeze(0) # create a mini-batch as expected by the model
    
    input_tensor2 = preprocess(seq2)
    seq2_batch = input_tensor2.unsqueeze(0) # create a mini-batch as expected by the model
    
    def getImageFeatures(img):
        
        resnet152 = models.resnet152(pretrained=True)
        feature_extractor = torch.nn.Sequential(*list(resnet152.children())[:-1])
        features = feature_extractor(img) # output now has the features corresponding to input 
        return features
    
    features1 = getImageFeatures(seq1_batch)
    features2 = getImageFeatures(seq2_batch)
    
    adjusted_features1 = adjust_dimensions(features1)
    adjusted_features2 = adjust_dimensions(features2)
    
    return (adjusted_features1, adjusted_features2)

def adjust_dimensions(features):
        
    flattened = torch.flatten(features) # we have (1,2048,1,1) so it will be 2048
    
    flattened_reshaped = flattened.unsqueeze(0).unsqueeze(2)

    flattened_reshaped = flattened_reshaped.repeat(1,1,2048)    # -> (1,2048,2048)
     
    return flattened_reshaped

#/mmdetection3d/mmdet3d/models/voxel_encoders
(features1, features2) = resnet_feature_extractor()
torch.save(features1.to("cuda:0"),'tensorA.pt')
torch.save(features2.to("cuda:0"),'tensorB.pt')
X=torch.load('tensorA.pt')
Y=torch.load('tensorB.pt')
