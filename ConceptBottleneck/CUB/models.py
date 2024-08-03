from CUB.template_model import MLP, inception_v3, End2EndModel, SimpleConvNetN, SimpleConvNetEqualParameter, EqualReceptiveFieldN


# Independent & Sequential Model
def ModelXtoC(pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim, three_class):    
    return inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                        n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                        three_class=three_class)

# Independent Model
def ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part
    if n_class_attr == 3:
        model = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)
    return model

# Sequential Model
def ModelXtoChat_ChatToY(n_class_attr, n_attributes, num_classes, expand_dim):
    # X -> C part is separate, this is only the C -> Y part (same as Independent model)
    return ModelOracleCtoY(n_class_attr, n_attributes, num_classes, expand_dim)

# Joint Model
def ModelXtoCtoY(n_class_attr, pretrained, freeze, num_classes, use_aux, n_attributes, expand_dim,
                 use_relu, use_sigmoid,use_unknown,encoder_model,expand_dim_encoder=0,num_middle_encoder=0):
    
    if use_unknown:
        n_attributes += 1
        
    if encoder_model == 'inceptionv3':
        model1 = inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                              n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                              three_class=(n_class_attr == 3))
    elif 'equal_parameter' in encoder_model:
        num_layers = int(encoder_model[-1])
        model1 = SimpleConvNetEqualParameter(num_classes=num_classes,num_layers=num_layers, aux_logits=use_aux,
                            n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                            three_class=(n_class_attr == 3))
    elif 'small' in encoder_model:
        num_layers = int(encoder_model[-1])
        model1 = SimpleConvNetN(num_classes=num_classes,num_layers=num_layers, aux_logits=use_aux,
                            n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                            three_class=(n_class_attr == 3))
    elif 'receptive_field' in encoder_model:
        num_layers = int(encoder_model[-1])
        model1 = SimpleConvNetN(num_classes=num_classes,num_layers=num_layers, aux_logits=use_aux,
                            n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                            three_class=(n_class_attr == 3))
        model1 = EqualReceptiveFieldN(num_classes=num_classes,num_layers=num_layers, aux_logits=use_aux,
                            n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim,
                            three_class=(n_class_attr == 3))
    elif encoder_model == 'mlp':
        model1 = MLP(299**2*3,n_attributes,expand_dim_encoder,encoder_model=True,num_middle_encoder=num_middle_encoder)
    else:
        raise Exception("{} not found".format(encoder_model))
            
    if n_class_attr == 3:
        model2 = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model2 = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim)

    return End2EndModel(model1, model2, use_relu, use_sigmoid, n_class_attr)

# Standard Model
def ModelXtoY(pretrained, freeze, num_classes, use_aux):
    return inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux)

# Multitask Model
def ModelXtoCY(pretrained, freeze, num_classes, use_aux, n_attributes, three_class, connect_CY):
    return inception_v3(pretrained=pretrained, freeze=freeze, num_classes=num_classes, aux_logits=use_aux,
                        n_attributes=n_attributes, bottleneck=False, three_class=three_class,
                        connect_CY=connect_CY)
