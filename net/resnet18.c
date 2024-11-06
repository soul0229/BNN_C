#include "common.h"
#include <stdio.h>
#include "conv.h"
#include "utils.h"

char net_lable[107][64] = {
    "conv1.weight", "bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var", 
    "layer1.0.conv1.weight", "layer1.0.bn1.weight", "layer1.0.bn1.bias", "layer1.0.bn1.running_mean", 
    "layer1.0.bn1.running_var", "layer1.0.conv2.weight", "layer1.0.bn2.weight", "layer1.0.bn2.bias", "layer1.0.bn2.running_mean", 
    "layer1.0.bn2.running_var",
    "layer1.1.conv1.weight", "layer1.1.bn1.weight", "layer1.1.bn1.bias", "layer1.1.bn1.running_mean", 
    "layer1.1.bn1.running_var", "layer1.1.conv2.weight", "layer1.1.bn2.weight", "layer1.1.bn2.bias", "layer1.1.bn2.running_mean", 
    "layer1.1.bn2.running_var",
    "layer2.0.conv1.weight", "layer2.0.bn1.weight", "layer2.0.bn1.bias", "layer2.0.bn1.running_mean", 
    "layer2.0.bn1.running_var", "layer2.0.conv2.weight", "layer2.0.bn2.weight", "layer2.0.bn2.bias", "layer2.0.bn2.running_mean", 
    "layer2.0.bn2.running_var",
    "layer2.0.shortcut.0.weight", "layer2.0.shortcut.1.weight", "layer2.0.shortcut.1.bias", "layer2.0.shortcut.1.running_mean",
    "layer2.0.shortcut.1.running_var", 
    "layer2.1.conv1.weight", "layer2.1.bn1.weight", "layer2.1.bn1.bias", "layer2.1.bn1.running_mean", 
    "layer2.1.bn1.running_var", "layer2.1.conv2.weight", "layer2.1.bn2.weight", "layer2.1.bn2.bias", "layer2.1.bn2.running_mean", 
    "layer2.1.bn2.running_var",
    "layer3.0.conv1.weight", "layer3.0.bn1.weight", "layer3.0.bn1.bias", "layer3.0.bn1.running_mean", 
    "layer3.0.bn1.running_var", "layer3.0.conv2.weight", "layer3.0.bn2.weight", "layer3.0.bn2.bias", "layer3.0.bn2.running_mean", 
    "layer3.0.bn2.running_var",
    "layer3.0.shortcut.0.weight", "layer3.0.shortcut.1.weight", "layer3.0.shortcut.1.bias", "layer3.0.shortcut.1.running_mean",
    "layer3.0.shortcut.1.running_var", 
    "layer3.1.conv1.weight", "layer3.1.bn1.weight", "layer3.1.bn1.bias", "layer3.1.bn1.running_mean", 
    "layer3.1.bn1.running_var", "layer3.1.conv2.weight", "layer3.1.bn2.weight", "layer3.1.bn2.bias", "layer3.1.bn2.running_mean", 
    "layer3.1.bn2.running_var",
    "layer4.0.conv1.weight", "layer4.0.bn1.weight", "layer4.0.bn1.bias", "layer4.0.bn1.running_mean", 
    "layer4.0.bn1.running_var", "layer4.0.conv2.weight", "layer4.0.bn2.weight", "layer4.0.bn2.bias", "layer4.0.bn2.running_mean", 
    "layer4.0.bn2.running_var",
    "layer4.0.shortcut.0.weight", "layer4.0.shortcut.1.weight", "layer4.0.shortcut.1.bias", "layer4.0.shortcut.1.running_mean",
    "layer4.0.shortcut.1.running_var",
    "layer4.1.conv1.weight", "layer4.1.bn1.weight", "layer4.1.bn1.bias", "layer4.1.bn1.running_mean", 
    "layer4.1.bn1.running_var", "layer4.1.conv2.weight", "layer4.1.bn2.weight", "layer4.1.bn2.bias", "layer4.1.bn2.running_mean", 
    "layer4.1.bn2.running_var",
    "bn2.weight", "bn2.bias", "bn2.bias", "bn2.running_mean", "bn2.running_var", 
    "linear.weight", "linear.bias",
};

Activate_TypeDef *net_inference(Net_List* Net, Activate_TypeDef *active)
{
    Activate_TypeDef *output = active;
    while(Net != NULL){
        switch(*((Data_Type*)(Net->data))){
            case KERNEL:
            CONV_KernelTypeDef *kernel = Net->data;
            if(kernel->binary == BINARY)
                output = BinarizeConv2d(kernel, output, 1, 1, 1);
            else
                output = Conv2d(kernel, output, 1, 1, 1);
            break;
            case BATCHNORM:
            BatchNorm *bn = Net->data;
            output = batchnorm_pcs(output, bn);
            break;
            case LINEAR:
            break;
            case LAYER:
            Net_List* net = ((Layer*)(Net->data))->data;
            output = net_inference(net, output);
            break;
            case NET_LIST:
            default:break;
        }
        Net = Net->next;
    }
    return output;
}
