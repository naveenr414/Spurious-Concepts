"""
InceptionV3 Network modified from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
New changes: add softmax layer + option for freezing lower layers except fc
"""
import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['MLP', 'Inception3', 'inception_v3', 'End2EndModel','SimpleConvNet','SimpleConvNet7','SimpleConvNetEqualParameter']

model_urls = {
    # Downloaded inception model (optional)
    'downloaded': 'pretrained/inception_v3_google-1a9a5a14.pth',
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


class CBRBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBRBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CBRNet(nn.Module):
    def __init__(self, num_classes, 
                 aux_logits=True,
                 transform_input=False, 
                 n_attributes=0, 
                 bottleneck=False, 
                 expand_dim=0, 
                 three_class=False, 
                 connect_CY=False,
                pretrained=False,
                freeze=False):
        
        super(CBRNet, self).__init__()
        self.cbr1 = CBRBlock(3, 32)
        self.cbr2 = CBRBlock(32, 64)
        self.cbr3 = CBRBlock(64, 128)
        self.cbr4 = CBRBlock(128, 256)
        self.cbr5 = CBRBlock(256, 512)
        self.all_fc = nn.ModuleList()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, n_attributes=self.n_attributes, bottleneck=bottleneck, \
                                                expand_dim=expand_dim, three_class=three_class, connect_CY=connect_CY)

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(45773312, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(45773312, 1, expand_dim))
        else:
            self.all_fc.append(FC(45773312, num_classes, expand_dim))
            
    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        x = self.cbr4(x)
        x = self.cbr5(x)
        x = x.view(x.size(0), -1)
        
        if self.training and self.aux_logits:
            out_aux = self.AuxLogits(x)
        
        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            out[0] += self.cy_fc(attr_preds)
        if self.training and self.aux_logits:
            return out, out_aux
        else:
            return out
        
        return x

class End2EndModel(torch.nn.Module):
    def __init__(self, model1, model2, use_relu=False, use_sigmoid=False, n_class_attr=2):
        super(End2EndModel, self).__init__()
        self.first_model = model1
        self.sec_model = model2
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid

    def forward_stage2(self, stage1_out):
        if self.use_relu:
            attr_outputs = [nn.ReLU()(o) for o in stage1_out]
        elif self.use_sigmoid:
            attr_outputs = [torch.nn.Sigmoid()(o) for o in stage1_out]
        else:
            attr_outputs = stage1_out
            
        stage2_inputs = attr_outputs

        stage2_inputs = torch.cat(stage2_inputs, dim=1)
        all_out = [self.sec_model(stage2_inputs)]
        
        all_out.extend(stage1_out)
        
        return all_out

    def forward(self, x,binary=False):
        if self.first_model.training:
            ret = self.first_model(x,binary=binary)
            outputs = ret[0]
            aux_outputs = ret[1]

            if len(ret) == 3:
                mask = ret[2]
                return self.forward_stage2(outputs), self.forward_stage2(aux_outputs), mask
            else:
                return self.forward_stage2(outputs), self.forward_stage2(aux_outputs)
        else:
            outputs = self.first_model(x,binary=binary)
            return self.forward_stage2(outputs)


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, expand_dim,encoder_model=False,num_middle_encoder=0):
        super(MLP, self).__init__()
        self.input_image_size = 256

        if num_classes == 112:
            self.input_image_size = 299

        self.expand_dim = expand_dim
        self.num_middle_encoder = num_middle_encoder 

        # For Encoder Models, generalize the notion of expander dim 
        # So there's multiple stacked 
        if encoder_model:
            input_dim = 3*self.input_image_size**2
            self.activation = torch.nn.ReLU()

            self.linear_layers = []

            if self.num_middle_encoder == 0:
                self.linear = nn.Linear(input_dim,num_classes)
                self.linear_layers = [self.linear]
            elif self.num_middle_encoder == 1:
                self.linear = nn.Linear(input_dim, expand_dim)
                self.activation = torch.nn.ReLU()
                self.linear2 = nn.Linear(expand_dim, num_classes) 
                self.linear_layers = [self.linear,self.activation,self.linear2]
            else:
                self.linear = nn.Linear(input_dim, expand_dim)
                self.activation = torch.nn.ReLU()
                self.linear2 = nn.Linear(expand_dim, expand_dim) 
                self.linear3 = nn.Linear(expand_dim, num_classes) 

                self.linear_layers = [self.linear] 
                for i in range(self.num_middle_encoder-1):
                    self.linear_layers += [self.activation,nn.Linear(expand_dim,expand_dim)] 
                
                self.linear_layers += [self.activation,self.linear3]
        else: 
            if self.expand_dim:
                self.linear = nn.Linear(input_dim, expand_dim)
                self.activation = torch.nn.ReLU()
                self.linear2 = nn.Linear(expand_dim, num_classes) #softmax is automatically handled by loss function
            else: 
                self.linear = nn.Linear(input_dim, num_classes)

        self.encoder_model = encoder_model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.linear = self.linear.to(device)
        if hasattr(self,'linear2'):
            self.linear2 = self.linear2.to(device)
        if hasattr(self,'linear3'):
            self.linear3 = self.linear3.to(device)
        if hasattr(self,'activation'):
            self.activation = self.activation.to(device) 
        if hasattr(self,'linear_layers'):
            for i in range(len(self.linear_layers)):
                self.linear_layers[i] = self.linear_layers[i].to(device)

    def forward(self, x, binary=False):
        if hasattr(self,'encoder_model') and self.encoder_model:
            x = x.view(x.shape[0],3*self.input_image_size**2)
            for i in self.linear_layers:
                x = i(x) 
        else:
            x = self.linear(x)
            if hasattr(self, 'expand_dim') and self.expand_dim:
                x = self.activation(x)
                x = self.linear2(x)
        if hasattr(self,'encoder_model') and self.encoder_model:
            x = [x[:,i].reshape((len(x),1)) for i in range(x.shape[1])]
            if self.training:
                return x, x
            else:
                return x
        else:
            return x


def inception_v3(pretrained, freeze, **kwargs):
    """Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception3(**kwargs)
        if os.path.exists(model_urls.get('downloaded')):
            model.load_partial_state_dict(torch.load(model_urls['downloaded']))
        else:
            model.load_partial_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        if freeze:  # only finetune fc layer
            for name, param in model.named_parameters():
                if 'fc' not in name:  # and 'Mixed_7c' not in name:
                    param.requires_grad = False
        return model

    return Inception3(**kwargs)

class SimpleConvNetN(nn.Module):
    def __init__(self, num_classes, num_layers,aux_logits=True, transform_input=False, 
                 n_attributes=0, bottleneck=False, expand_dim=0, 
                 three_class=False, connect_CY=False):
        super(SimpleConvNetN, self).__init__()
        self.conv_layers = nn.ModuleList()
        channels = 512
        for i in range(num_layers):
            in_channels = 3 if i == 0 else 2**(i+5)
            out_channels = 2**(i+6) 
            in_channels = min(in_channels,512)
            out_channels = min(out_channels,512)
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv_layers.append(conv_layer)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.input_size = 256

        dataset = "synthetic"

        if n_attributes == 112:
            dataset = "cub"
        elif n_attributes == 15:
            dataset = "coco"

        # Calculate the output size of the last conv layer before the linear layer
        first_num = min(512,256*2**(num_layers-3))
        
        if dataset == "cub" or dataset == "coco":
            second_num = 37//(2**(num_layers-3))
        elif dataset == "synthetic":
            second_num = 32//(2**(num_layers-3))
        else:
            second_num = 32//(2**(num_layers-3))
        self.conv_output_size = first_num*second_num**2

        self.all_fc = nn.ModuleList()
        
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, n_attributes=self.n_attributes, bottleneck=bottleneck, \
                                                expand_dim=expand_dim, three_class=three_class, connect_CY=connect_CY)

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

            
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(self.conv_output_size, 1, expand_dim))
        else:
            self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))

    def forward(self, x,binary=False):
        for conv_layer in self.conv_layers:
            x = self.pool(torch.relu(conv_layer(x)))        
        self.last_conv_output = x

        # Flatten the tensor before passing it through the fully connected layers
        x = x.view(-1, self.conv_output_size)
        self.output_before_fc = x

        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            if binary:
                attr_preds = torch.round(attr_preds).float()
            
            out[0] += self.cy_fc(attr_preds)
        if self.training and self.aux_logits:
            return out, out
        else:
            return out

class EqualReceptiveFieldN(nn.Module):
    def __init__(self, num_classes, num_layers,aux_logits=True, transform_input=False, 
                 n_attributes=0, bottleneck=False, expand_dim=0, 
                 three_class=False, connect_CY=False):
        super(EqualReceptiveFieldN, self).__init__()
        self.conv_layers = nn.ModuleList()
        channels = 512
        for i in range(num_layers):
            in_channels = 3 if i == 0 else 2**(i+5)
            out_channels = 2**(i+6) 
            in_channels = min(in_channels,512)
            out_channels = min(out_channels,512)

            if i == 0:
                conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=15-2*(num_layers-1), stride=1, padding=1)
            else:
                conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv_layers.append(conv_layer)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.input_size = 256

        dataset = "synthetic"

        if n_attributes == 112:
            dataset = "cub"
        elif n_attributes == 15:
            dataset = "coco"

        # Calculate the output size of the last conv layer before the linear layer
        first_num = min(2**9,2**(num_layers+5))
        
        if dataset == "cub" or dataset == "coco":
            second_num = 37//(2**(num_layers-3))
        elif dataset == "synthetic":
            if num_layers == 7:
                second_num = 32//(2**(num_layers-3))
            else:
                second_num = 31//(2**(num_layers-3))
        else:
            second_num = 32//(2**(num_layers-3))
        self.conv_output_size = first_num*second_num**2

        self.all_fc = nn.ModuleList()
        
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, n_attributes=self.n_attributes, bottleneck=bottleneck, \
                                                expand_dim=expand_dim, three_class=three_class, connect_CY=connect_CY)

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

            
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(self.conv_output_size, 1, expand_dim))
        else:
            self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))

    def forward(self, x,binary=False):
        for conv_layer in self.conv_layers:
            x = self.pool(torch.relu(conv_layer(x)))        
        self.last_conv_output = x

        # Flatten the tensor before passing it through the fully connected layers
        x = x.view(-1, self.conv_output_size)
        self.output_before_fc = x

        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            if binary:
                attr_preds = torch.round(attr_preds).float()
            
            out[0] += self.cy_fc(attr_preds)
        if self.training and self.aux_logits:
            return out, out
        else:
            return out


class SimpleConvNetEqualParameter(nn.Module):
    def __init__(self, num_classes, num_layers,aux_logits=True, transform_input=False, 
                 n_attributes=0, bottleneck=False, expand_dim=0, 
                 three_class=False, connect_CY=False):
        super(SimpleConvNetEqualParameter, self).__init__()
        self.conv_layers = nn.ModuleList()
        channel_list = []

        if num_layers == 3:
            channel_list = [3,64,64,32]
        elif num_layers == 4:
            channel_list = [3,64,64,64,48]
        elif num_layers == 5:
            channel_list = [3,64,64,64,40,40]
        elif num_layers == 6:
            channel_list = [3,64,64,64,32,32,32]
        elif num_layers == 7:
            channel_list = [3,64,64,64,32,28,24,20]

        for i in range(num_layers):
            in_channels = channel_list[i]
            out_channels = channel_list[i+1]
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv_layers.append(conv_layer)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.input_size = 256

        # Calculate the output size of the last conv layer before the linear layer
        self.conv_output_size = channel_list[-1]*(2**(8-num_layers))**2

        self.all_fc = nn.ModuleList()
        
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, n_attributes=self.n_attributes, bottleneck=bottleneck, \
                                                expand_dim=expand_dim, three_class=three_class, connect_CY=connect_CY)

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

            
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(self.conv_output_size, 1, expand_dim))
        else:
            self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))

    def forward(self, x,binary=False):
        for conv_layer in self.conv_layers:
            x = self.pool(torch.relu(conv_layer(x)))        
        self.last_conv_output = x


        # Flatten the tensor before passing it through the fully connected layers
        x = x.view(-1, self.conv_output_size)
        self.output_before_fc = x

        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            if binary:
                attr_preds = torch.round(attr_preds).float()
            
            out[0] += self.cy_fc(attr_preds)
        if self.training and self.aux_logits:
            return out, out
        else:
            return out



class SimpleConvNet(nn.Module):
    def __init__(self, num_classes, aux_logits=True, transform_input=False, 
                 n_attributes=0, bottleneck=False, expand_dim=0, 
                 three_class=False, connect_CY=False):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Essentially, for CUB, it uses a 299x299 dataset whereas with 
        # The other one, we use a 256x256
        # When we run final experiments, change everything to be 299x299

        if num_classes == 200 or num_classes == 100:
            self.conv_output_size = 256*37*37
        else:
            self.conv_output_size = 256*32*32

        self.all_fc = nn.ModuleList()
        
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, n_attributes=self.n_attributes, bottleneck=bottleneck, \
                                                expand_dim=expand_dim, three_class=three_class, connect_CY=connect_CY)

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

            
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(self.conv_output_size, 1, expand_dim))
        else:
            self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))

    def forward(self,x,binary=False):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        self.last_conv_output = x

        x = x.view(-1, self.conv_output_size)
        
        self.output_before_fc = x

        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            if binary:
                attr_preds = torch.round(attr_preds).float()
            
            out[0] += self.cy_fc(attr_preds)
        if self.training and self.aux_logits:
            return out, out
        else:
            return out

class SimpleConvNet4(nn.Module):
    def __init__(self, num_classes, aux_logits=True, transform_input=False, 
                 n_attributes=0, bottleneck=False, expand_dim=0, 
                 three_class=False, connect_CY=False):
        super(SimpleConvNet4, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size of the last conv layer before the linear layer
        self.conv_output_size = 256 * 16 * 16  # This may change depending on the input size
        self.all_fc = nn.ModuleList()
        
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, n_attributes=self.n_attributes, bottleneck=bottleneck, \
                                                expand_dim=expand_dim, three_class=three_class, connect_CY=connect_CY)

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

            
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(self.conv_output_size, 1, expand_dim))
        else:
            self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))

    def forward(self,x,binary=False):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))

        self.last_conv_output = x
        
        # Flatten the tensor before passing it through the fully connected layers
        x = x.view(-1, self.conv_output_size)
        
        self.output_before_fc = x

        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            if binary:
                attr_preds = torch.round(attr_preds).float()
            
            out[0] += self.cy_fc(attr_preds)
        if self.training and self.aux_logits:
            return out, out
        else:
            return out

class SimpleConvNet5(nn.Module):
    def __init__(self, num_classes, aux_logits=True, transform_input=False, 
                 n_attributes=0, bottleneck=False, expand_dim=0, 
                 three_class=False, connect_CY=False):
        super(SimpleConvNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size of the last conv layer before the linear layer
        self.conv_output_size = 512 * 8 * 8  # This may change depending on the input size
        
        self.all_fc = nn.ModuleList()
        
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, n_attributes=self.n_attributes, bottleneck=bottleneck, \
                                                expand_dim=expand_dim, three_class=three_class, connect_CY=connect_CY)

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

            
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(self.conv_output_size, 1, expand_dim))
        else:
            self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))

    def forward(self, x,binary=False):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        
        self.last_conv_output = x

        # Flatten the tensor before passing it through the fully connected layers
        x = x.view(-1, self.conv_output_size)
                
        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            if binary:
                attr_preds = torch.round(attr_preds).float()
            
            out[0] += self.cy_fc(attr_preds)
        if self.training and self.aux_logits:
            return out, out
        else:
            return out

class SimpleConvNet6(nn.Module):
    def __init__(self, num_classes, aux_logits=True, transform_input=False, 
                 n_attributes=0, bottleneck=False, expand_dim=0, 
                 three_class=False, connect_CY=False):
        super(SimpleConvNet6, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size of the last conv layer before the linear layer
        self.conv_output_size = 512 * 4 * 4  # This may change depending on the input size
        
        self.all_fc = nn.ModuleList()
        
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, n_attributes=self.n_attributes, bottleneck=bottleneck, \
                                                expand_dim=expand_dim, three_class=three_class, connect_CY=connect_CY)

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

            
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(self.conv_output_size, 1, expand_dim))
        else:
            self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))

    def forward(self, x,binary=False):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.pool(torch.relu(self.conv6(x)))   

        self.last_conv_output = x

        # Flatten the tensor before passing it through the fully connected layers
        x = x.view(-1, self.conv_output_size)
                
        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            if binary:
                attr_preds = torch.round(attr_preds).float()
            
            out[0] += self.cy_fc(attr_preds)
        if self.training and self.aux_logits:
            return out, out
        else:
            return out


class SimpleConvNet7(nn.Module):
    def __init__(self, num_classes, aux_logits=True, transform_input=False, 
                 n_attributes=0, bottleneck=False, expand_dim=0, 
                 three_class=False, connect_CY=False):
        super(SimpleConvNet7, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size of the last conv layer before the linear layer
        self.conv_output_size = 512 * 2 * 2  # This may change depending on the input size
        
        self.all_fc = nn.ModuleList()
        
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, n_attributes=self.n_attributes, bottleneck=bottleneck, \
                                                expand_dim=expand_dim, three_class=three_class, connect_CY=connect_CY)

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

            
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(self.conv_output_size, 1, expand_dim))
        else:
            self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))

    def forward(self, x,binary=False):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.pool(torch.relu(self.conv6(x)))
        x = self.pool(torch.relu(self.conv7(x)))
        
        self.last_conv_output = x

        # Flatten the tensor before passing it through the fully connected layers
        x = x.view(-1, self.conv_output_size)
        self.output_before_fc = x

        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            if binary:
                attr_preds = torch.round(attr_preds).float()
            
            out[0] += self.cy_fc(attr_preds)
        if self.training and self.aux_logits:
            return out, out
        else:
            return out

class SimpleConvNet7(nn.Module):
    def __init__(self, num_classes, aux_logits=True, transform_input=False, 
                 n_attributes=0, bottleneck=False, expand_dim=0, 
                 three_class=False, connect_CY=False):
        super(SimpleConvNet7, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size of the last conv layer before the linear layer
        self.conv_output_size = 512 * 2 * 2  # This may change depending on the input size
        
        self.all_fc = nn.ModuleList()
        
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, n_attributes=self.n_attributes, bottleneck=bottleneck, \
                                                expand_dim=expand_dim, three_class=three_class, connect_CY=connect_CY)

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

            
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(self.conv_output_size, 1, expand_dim))
        else:
            self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))

    def forward(self, x,binary=False):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.pool(torch.relu(self.conv6(x)))
        x = self.pool(torch.relu(self.conv7(x)))
        
        self.last_conv_output = x

        # Flatten the tensor before passing it through the fully connected layers
        x = x.view(-1, self.conv_output_size)
        self.output_before_fc = x

        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            if binary:
                attr_preds = torch.round(attr_preds).float()
            
            out[0] += self.cy_fc(attr_preds)
        if self.training and self.aux_logits:
            return out, out
        else:
            return out

class SimpleConvNet7_SoftPlus(nn.Module):
    def __init__(self, num_classes, aux_logits=True, transform_input=False, 
                 n_attributes=0, bottleneck=False, expand_dim=0, 
                 three_class=False, connect_CY=False):
        super(SimpleConvNet7_SoftPlus, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size of the last conv layer before the linear layer
        self.conv_output_size = 512 * 2 * 2  # This may change depending on the input size
        
        self.all_fc = nn.ModuleList()
        self.softplus = torch.nn.Softplus()
        
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, n_attributes=self.n_attributes, bottleneck=bottleneck, \
                                                expand_dim=expand_dim, three_class=three_class, connect_CY=connect_CY)

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

            
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(self.conv_output_size, 1, expand_dim))
        else:
            self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))

    def forward(self, x,binary=False):
        x = self.pool(self.softplus(self.conv1(x)))
        x = self.pool(self.softplus(self.conv2(x)))
        x = self.pool(self.softplus(self.conv3(x)))
        x = self.pool(self.softplus(self.conv4(x)))
        x = self.pool(self.softplus(self.conv5(x)))
        x = self.pool(self.softplus(self.conv6(x)))
        x = self.pool(self.softplus(self.conv7(x)))
        
        self.last_conv_output = x

        # Flatten the tensor before passing it through the fully connected layers
        x = x.view(-1, self.conv_output_size)
        self.output_before_fc = x

        out = []
        for fc in self.all_fc: 
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            if binary:
                attr_preds = torch.round(attr_preds).float()
            
            out[0] += self.cy_fc(attr_preds)
        if self.training and self.aux_logits:
            return out, out
        else:
            return out


class SimpleConvNet9(nn.Module):
    def __init__(self, num_classes, aux_logits=True, transform_input=False, 
                 n_attributes=0, bottleneck=False, expand_dim=0, 
                 three_class=False, connect_CY=False):
        super(SimpleConvNet7, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)


        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the output size of the last conv layer before the linear layer
        self.conv_output_size = 512 * 2 * 2  # This may change depending on the input size
        
        self.all_fc = nn.ModuleList()
        
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, n_attributes=self.n_attributes, bottleneck=bottleneck, \
                                                expand_dim=expand_dim, three_class=three_class, connect_CY=connect_CY)

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

            
        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(self.conv_output_size, 1, expand_dim))
        else:
            self.all_fc.append(FC(self.conv_output_size, num_classes, expand_dim))

    def forward(self, x,binary=False):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.pool(torch.relu(self.conv6(x)))
        x = self.pool(torch.relu(self.conv7(x)))
        x = self.pool(torch.relu(self.conv8(x)))
        x = self.pool(torch.relu(self.conv9(x)))
        
        self.last_conv_output = x

        # Flatten the tensor before passing it through the fully connected layers
        x = x.view(-1, self.conv_output_size)
        self.output_before_fc = x

        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            if binary:
                attr_preds = torch.round(attr_preds).float()
            
            out[0] += self.cy_fc(attr_preds)
        if self.training and self.aux_logits:
            return out, out
        else:
            return out
        
class Inception3(nn.Module):

    def __init__(self, num_classes, aux_logits=True, transform_input=False, n_attributes=0, bottleneck=False, expand_dim=0, three_class=False, connect_CY=False):
        """
        Args:
        num_classes: number of main task classes
        aux_logits: whether to also output auxiliary logits
        transform input: whether to invert the transformation by ImageNet (should be set to True later on)
        n_attributes: number of attributes to predict
        bottleneck: whether to make X -> A model
        expand_dim: if not 0, add an additional fc layer with expand_dim neurons
        three_class: whether to count not visible as a separate class for predicting attribute
        """
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes, n_attributes=self.n_attributes, bottleneck=bottleneck, \
                                                expand_dim=expand_dim, three_class=three_class, connect_CY=connect_CY)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        self.all_fc = nn.ModuleList() #separate fc layer for each prediction task. If main task is involved, it's always the first fc in the list
        
        self.last_conv_output = None

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(2048, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(2048, 1, expand_dim))
        else:
            self.all_fc.append(FC(2048, num_classes, expand_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x,binary=False):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:
            out_aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        
        self.last_conv_output = x
        
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 2048
        out = []
        
        self.output_before_fc = x
        
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            if binary:
                attr_preds = torch.round(attr_preds).float()
            
            out[0] += self.cy_fc(attr_preds)
        if self.training and self.aux_logits:
            return out, out_aux
        else:
            return out

    def load_partial_state_dict(self, state_dict):
        """
        If dimensions of the current model doesn't match the pretrained one (esp for fc layer), load whichever weights that match
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state or 'fc' in name:
                continue
            if isinstance(param, Parameter):
                param = param.data
            own_state[name].copy_(param)


class FC(nn.Module):

    def __init__(self, input_dim, output_dim, expand_dim, stddev=None):
        """
        Extend standard Torch Linear layer to include the option of expanding into 2 Linear layers
        """
        super(FC, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim > 0:
            self.relu = nn.ReLU()
            self.fc_new = nn.Linear(input_dim, expand_dim)
            self.fc = nn.Linear(expand_dim, output_dim)
        else:
            self.fc = nn.Linear(input_dim, output_dim)
        if stddev:
            self.fc.stddev = stddev
            if expand_dim > 0:
                self.fc_new.stddev = stddev

    def forward(self, x):
        if self.expand_dim > 0:
            x = self.fc_new(x)
            x = self.relu(x)
        x = self.fc(x)
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, n_attributes=0, bottleneck=False, expand_dim=0, three_class=False, connect_CY=False):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.expand_dim = expand_dim

        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

        self.all_fc = nn.ModuleList()

        if n_attributes > 0:
            if not bottleneck: #cotraining
                self.all_fc.append(FC(768, num_classes, expand_dim, stddev=0.001))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(768, 1, expand_dim, stddev=0.001))
        else:
            self.all_fc.append(FC(768, num_classes, expand_dim, stddev=0.001))

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=6, stride=2)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 768
        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            out[0] += self.cy_fc(attr_preds)
        return out


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
