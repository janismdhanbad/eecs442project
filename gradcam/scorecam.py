import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
from PIL import Image

from models import *
# from models_orig import *
from misc_functions import *


"""
Created on Wed Apr 29 16:11:20 2020
@author: Haofan Wang - github.com/haofanwang
"""
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from misc_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        layer_outputs = []
        output = []

        conv_output = None
        x = x.float()

        for i, (mdef, module) in enumerate(zip(self.model.module_defs, self.model.module_list)):
            mtype = mdef["type"]
            if mtype in ["convolutional", "upsample", "maxpool"]:
                    x = module(x)
            elif mtype == "route":
                layers = [int(x) for x in mdef["layers"]]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    try:
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    except:
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
            elif mtype == "shortcut":
                    x = x + layer_outputs[int(mdef["from"][0])]
            elif mtype == "yolo":
                x = module(x, (416, 416))
                output.append(x)
            # import pdb; pdb.set_trace()
            layer_outputs.append(x)

            if i == self.target_layer:
                x.register_hook(self.save_gradient)
                #x.register_hook(lambda grad: print(grad))
                conv_output = x

        return conv_output, x[1]

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        # x = self.model.classifier(x)
        return conv_output, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.to("cuda:0").eval()

        # define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(416, 416), mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
            w = F.softmax(self.extractor.forward_pass(input_image*norm_saliency_map)[1],dim=1)[0][target_class]
            cam += w.data.cpu().numpy() * target[i, :, :].data.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam


if __name__ == '__main__':
    image_dir = "sample_files/"

    target_class = 15
    
    file_name_to_export = "cat" # class name you are predicting

    original_image, prep_img = params_for_yolo(image_dir, target_class, file_name_to_export)
    # /home/datumx/data_science_experiments/traffic_sign_recogntiton/new_proj/eecs442project/PyTorch-YOLOv3/config/yolov3-custom.cfg
    # /home/datumx/data_science_experiments/traffic_sign_recogntiton/new_proj/eecs442project/PyTorch-YOLOv3/checkpoints/yolov3_ckpt.pth
    model = Darknet("/home/datumx/data_science_experiments/traffic_sign_recogntiton/new_proj/yolov3/cfg/yolov3-spp.cfg", 416)
    model.load_state_dict(torch.load("/home/datumx/data_science_experiments/traffic_sign_recogntiton/new_proj/yolov3/weights/yolov3-spp-ultralytics.pt", map_location = "cuda:0")["model"])
    # import pdb; pdb.set_trace()
    prep_img = torch.from_numpy(prep_img).to("cuda:0")

    if prep_img.ndimension() == 3:
        prep_img = prep_img.unsqueeze(0)

    model = model.to("cuda:0").eval()
    score_cam = ScoreCam(model, target_layer=90)
    # Generate cam mask
    cam = score_cam.generate_cam(prep_img, target_class)
    import pdb; pdb.set_trace()
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Score cam completed')