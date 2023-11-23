"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""
import pdb
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import numpy as np
from analysis import Logger, AverageMeter, accuracy, binary_accuracy
from sklearn.metrics import roc_auc_score
import pickle

from CUB import probe, tti, gen_cub_synthetic, hyperopt
from CUB.dataset import load_data, find_class_imbalance
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from CUB.models import ModelXtoCY, ModelXtoChat_ChatToY, ModelXtoY, ModelXtoC, ModelOracleCtoY, ModelXtoCtoY
from CUB.sam import SAM

def run_epoch_simple(model, optimizer, loader, loss_meter, acc_meter, criterion, args, is_training):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels = data
        if isinstance(inputs, list):
            #inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t().float()
        inputs = torch.flatten(inputs, start_dim=1).float()
        inputs_var = torch.autograd.Variable(inputs).cuda() if torch.cuda.is_available() else torch.autograd.Variable(inputs)
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels).cuda() if torch.cuda.is_available() else torch.autograd.Variable(labels)
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var
        
        outputs = model(inputs_var)
        loss = criterion(outputs, labels_var)
        acc = accuracy(outputs, labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))

        if is_training:
            optimizer.zero_grad() #zero the parameter gradients
            loss.backward()
            optimizer.step() #optimizer step to update parameters
    return loss_meter, acc_meter

def run_epoch(model, optimizer, loader, loss_meter, acc_meter, concept_acc_meter,criterion, attr_criterion, args, epoch, top_pairs=[],is_training=True):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """

    if is_training:
        model.train()
    else:
        model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
        
    for _, data in enumerate(loader):
        if attr_criterion is None:
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
        else:
            inputs, labels, attr_labels = data
            if args.n_attributes > 1:
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).t()#.float() #N x 312
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
            attr_labels_var = torch.autograd.Variable(attr_labels).float()
            attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var
            
        inputs_var = torch.autograd.Variable(inputs).to(device)
        labels_var = torch.autograd.Variable(labels).to(device)

        loss_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        
        if is_training and args.use_aux:
            binary = args.train_addition == "binary"

            if args.encoder_model == 'mlp_mask':
                outputs, aux_outputs, mask = model(inputs_var,binary=binary)
            else:
                outputs, aux_outputs = model(inputs_var,binary=binary)
            losses = []
            out_start = 0
            
            if not args.bottleneck: #loss main is for the main task label (always the first output)
                loss_main = 1.0 * criterion(outputs[0], labels_var) + 0.4 * criterion(aux_outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * (1.0 * attr_criterion[i](outputs[i+out_start].squeeze().type(loss_type), attr_labels_var[:, i]) \
                                                            + 0.4 * attr_criterion[i](aux_outputs[i+out_start].squeeze().type(loss_type), attr_labels_var[:, i])))
            if args.encoder_model == 'mlp_mask':
                zeroes = torch.zeros_like(mask[0])
                
                if torch.cuda.is_available():
                    zeroes = zeroes.cuda()

                for m in mask:
                    m = m / torch.max(m)

                    # Calculate L1 norms along the grouped dimensions
                    l1_norm = torch.norm(m, 1)
                    
                    # Sum L1 norms across groups and take the mean
                    l1_norm = torch.mean(l1_norm)
                    losses.append(args.mask_loss_weight * l1_norm)

        else: #testing or no aux logits
            outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck:
                loss_main = criterion(outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end                
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * attr_criterion[i](outputs[i+out_start].squeeze().type(loss_type), attr_labels_var[:, i]))

        if args.bottleneck: #attribute accuracy
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
            if args.use_unknown:
                sigmoid_outputs = sigmoid_outputs[:,:-1]
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
            concept_acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            concept_predictions = torch.nn.Sigmoid()(torch.stack(outputs[1:])[:,:,0].T)
            concept_accuracy = binary_accuracy(concept_predictions,attr_labels)
            concept_acc_meter.update(concept_accuracy.cpu().numpy(),inputs.size(0))

            acc = accuracy(outputs[0], labels, topk=(1,)) #only care about class prediction accuracy
            acc_meter.update(acc[0], inputs.size(0))

        if attr_criterion is not None:
            if args.train_variation == "loss":
                SCALE_FACTOR = 1.5
                for (i,j) in top_pairs:
                    matching_data_0 = (attr_labels[:,i] == 1) & (attr_labels[:,j] == 0)
                    matching_data_1 = (attr_labels[:,i] == 0) & (attr_labels[:,j] == 1)
                    m_0 = torch.sum(matching_data_0)/len(matching_data_0)
                    m_1 = torch.sum(matching_data_1)/len(matching_data_1)

                    losses[i] *= (1-m_0 + m_0*SCALE_FACTOR)
                    losses[j] *= (1-m_1 + m_1*SCALE_FACTOR)

            if args.train_variation == "half" and epoch < args.epochs/2:
                total_loss = sum(losses[1:])
                if args.normalize_loss:
                    total_loss /= (args.attr_loss_weight*args.n_attributes)
            elif args.bottleneck:
                total_loss = sum(losses)/ args.n_attributes
            else: #cotraining, loss by class prediction and loss by attribute prediction have the same weight
                total_loss = losses[0] + sum(losses[1:])
                if args.normalize_loss:
                    total_loss = total_loss / (1 + args.attr_loss_weight * args.n_attributes)
        else: #finetune
            total_loss = sum(losses)
        loss_meter.update(total_loss.item(), inputs.size(0))

        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return loss_meter, acc_meter, concept_acc_meter 


def train(model, args): 
    imbalance = None
    if args.use_attr and not args.no_img and args.weighted_loss:
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
        """if args.weighted_loss == 'multiple':
            imbalance = find_class_imbalance(train_data_path, True)
        else:
            imbalance = find_class_imbalance(train_data_path, False)"""
        
        imbalance = [1 for i in range(args.n_attributes)]

    top_pairs = []

    if args.train_variation == 'loss':
        train_data = pickle.load(open(train_data_path,"rb"))
        FRACTION_PAIRS = 0.25
        num_pairs = round(len(train_data[0]['attribute_label'])**2*FRACTION_PAIRS)
        co_occurences = np.zeros((len(train_data[0]['attribute_label']),
                                    len(train_data[0]['attribute_label'])))

        for i in train_data: 
            co_occurences += np.array([i['attribute_label']]).dot(np.array([i['attribute_label']]).T)
        
        for i in range(co_occurences.shape[0]):
            for j in range(i,co_occurences.shape[0]):
                co_occurences[i][j] = 0

        def top_k_indices(arr, k):
            """Find the top K highest pairs in array"""
            indices = np.argpartition(arr.flatten(), -k)[-k:]
            indices = np.unravel_index(indices, arr.shape)
            return list(zip(indices[0], indices[1]))

        top_pairs = top_k_indices(co_occurences,num_pairs)

    if not os.path.exists(args.log_dir): # job restarted by cluster
        os.makedirs(args.log_dir)
    #     for f in os.listdir(args.log_dir):
    #         os.remove(os.path.join(args.log_dir, f))
    # else:
    #     os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'))
    logger.write(str(args) + '\n')
    logger.write(str(imbalance) + '\n')
    logger.flush()

    if torch.cuda.is_available():
        model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    if args.use_attr and not args.no_img:
        attr_criterion = [] #separate criterion (loss function) for each attribute
        if args.weighted_loss:
            assert(imbalance is not None)
            for ratio in imbalance:
                weight = torch.FloatTensor([ratio])
                if torch.cuda.is_available():
                    weight = weight.cuda()
                attr_criterion.append(torch.nn.BCEWithLogitsLoss(weight=weight))
        else:
            for i in range(args.n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
    else:
        attr_criterion = None

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'sam':
        optimizer = SAM(model.parameters(), torch.optim.SGD, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9) 
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.00001, min_lr=0.00001, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    stop_epoch = int(math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE)) * args.scheduler_step
    print("Stop epoch: ", stop_epoch)

    train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    logger.write('train data path: %s\n' % train_data_path)

    resize = args.encoder_model=='inceptionv3' or 'dsprites' in args.data_dir or 'CUB' in args.data_dir

    if args.ckpt: #retraining
        train_loader = load_data([train_data_path, val_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling,experiment_name=args.experiment_name,resize=resize)
        val_loader = None
    else:        
        train_loader = load_data([train_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling, experiment_name=args.experiment_name,resize=args.encoder_model=='inceptionv3')
        val_loader = load_data([val_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr, experiment_name=args.experiment_name,resize=resize)
    
    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    if args.train_variation == "half":
        for param in model.sec_model.parameters():
            param.requires_grad = False

    for epoch in range(0, args.epochs):
        print("On epoch {}".format(epoch))
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        train_concept_acc_meter = AverageMeter()

        if args.train_variation == "half" and epoch == args.epochs//2:
            scale_lr = 5 
            for param in model.sec_model.parameters():
                param.requires_grad = True

            new_lr = scheduler.get_lr()/scale_lr 
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        
        if args.no_img:
            train_loss_meter, train_acc_meter = run_epoch_simple(model, optimizer, train_loader, train_loss_meter, train_acc_meter,criterion, args, is_training=True)
        else:
            train_loss_meter, train_acc_meter, train_concept_acc_meter = run_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter,  train_concept_acc_meter, criterion, attr_criterion, args, epoch, top_pairs=top_pairs,is_training=True)

            
        if not args.ckpt: # evaluate on val set
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()
            val_concept_acc_meter = AverageMeter()
        
            with torch.no_grad():
                if args.no_img:
                    val_loss_meter, val_acc_meter = run_epoch_simple(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, args, is_training=False)
                else:
                    val_loss_meter, val_acc_meter, val_concept_acc_meter = run_epoch(model, optimizer, val_loader, val_loss_meter, val_acc_meter, val_concept_acc_meter, criterion, attr_criterion, args, epoch, is_training=False)

        else: #retraining
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter
            val_concept_acc_meter = train_concept_acc_meter

        if best_val_acc < val_acc_meter.avg: 
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)

            torch.save(model, os.path.join(args.log_dir, 'best_model_%d.pth' % args.seed))
            
            #if best_val_acc >= 100: #in the case of retraining, stop when the model reaches 100% accuracy on both train + val sets
            #    break

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg
                
        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\tTrain concept accuracy: %.4f\t'
                'Val loss: %.4f\tVal acc: %.4f\tVal concept acc: %.4f\t'
                'Best val epoch: %d\n'
                % (epoch, train_loss_avg, train_acc_meter.avg, train_concept_acc_meter.avg, val_loss_avg, val_acc_meter.avg, val_concept_acc_meter.avg, best_val_epoch)) 
        logger.flush()
        
        if epoch <= stop_epoch:
            scheduler.step(epoch) #scheduler step to update lr at the end of epoch     
        #inspect lr
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_lr())

        # if epoch % args.save_step == 0:
        #     torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break

    print("Saving the model again to {}!".format(args.log_dir))
    
    torch.save(model, os.path.join(args.log_dir, 'best_model_%d.pth' % args.seed))
            
def train_X_to_C(args):
    if args.use_unknown:
        model = ModelXtoC(pretrained=args.pretrained, freeze=args.freeze, num_classes=args.num_classes, use_aux=args.use_aux,
                          n_attributes=args.n_attributes+1, expand_dim=args.expand_dim, three_class=args.three_class)
    else:
        model = ModelXtoC(pretrained=args.pretrained, freeze=args.freeze, num_classes=args.num_classes, use_aux=args.use_aux,
                          n_attributes=args.n_attributes, expand_dim=args.expand_dim, three_class=args.three_class)
    train(model, args)

def train_oracle_C_to_y_and_test_on_Chat(args):
    model = ModelOracleCtoY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                            num_classes=args.num_classes, expand_dim=args.expand_dim)
    train(model, args)

def train_Chat_to_y_and_test_on_Chat(args):
    model = ModelXtoChat_ChatToY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                                 num_classes=args.num_classes, expand_dim=args.expand_dim)
    train(model, args)

def train_X_to_C_to_y(args):
    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                         num_classes=args.num_classes, use_aux=args.use_aux, n_attributes=args.n_attributes,
                         expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid,
                        use_unknown=args.use_unknown,encoder_model=args.encoder_model,expand_dim_encoder=args.expand_dim_encoder, 
                        num_middle_encoder=args.num_middle_encoder)
    # Load the model 

    if args.load_model != 'none':
        model = torch.load(open("../results/models/{}.pt".format(args.load_model),"rb"))

    # model.load_state_dict(weights)

    train(model, args)

def train_X_to_y(args):
    model = ModelXtoY(pretrained=args.pretrained, freeze=args.freeze, num_classes=args.num_classes, use_aux=args.use_aux)
    train(model, args)

def train_X_to_Cy(args):
    model = ModelXtoCY(pretrained=args.pretrained, freeze=args.freeze, num_classes=args.num_classes, use_aux=args.use_aux,
                       n_attributes=args.n_attributes, three_class=args.three_class, connect_CY=args.connect_CY)
    train(model, args)

def train_probe(args):
    probe.run(args)

def test_time_intervention(args):
    tti.run(args)

def robustness(args):
    gen_cub_synthetic.run(args)

def hyperparameter_optimization(args):
    hyperopt.run(args)


def parse_arguments(experiment):
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='CUB Training')
    parser.add_argument('dataset', type=str, help='Name of the dataset.')
    parser.add_argument('exp', type=str,
                        choices=['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                                 'Standard', 'Multitask', 'Joint', 'Probe',
                                 'TTI', 'Robustness', 'HyperparameterSearch'],
                        help='Name of experiment to run.')
    parser.add_argument('--seed', required=True, type=int, help='Numpy and torch seed.')

    if experiment == 'Probe':
        return (probe.parse_arguments(parser),)

    elif experiment == 'TTI':
        return (tti.parse_arguments(parser),)

    elif experiment == 'Robustness':
        return (gen_cub_synthetic.parse_arguments(parser),)

    elif experiment == 'HyperparameterSearch':
        return (hyperopt.parse_arguments(parser),)

    else:
        parser.add_argument('-log_dir', default=None, help='where the trained model is saved')
        parser.add_argument('-batch_size', '-b', type=int, help='mini-batch size')
        parser.add_argument('-epochs', '-e', type=int, help='epochs for training process')
        parser.add_argument('-save_step', default=1000, type=int, help='number of epochs to save model')
        parser.add_argument('-lr', type=float, help="learning rate")
        parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
        parser.add_argument('-pretrained', '-p', action='store_true',
                            help='whether to load pretrained model & just fine-tune')
        parser.add_argument('-freeze', action='store_true', help='whether to freeze the bottom part of inception network')
        parser.add_argument('-use_aux', action='store_true', help='whether to use aux logits')
        parser.add_argument('-use_unknown', action='store_true',
                            help='whether to include an extra node during training (only with sequential, joint)')
        parser.add_argument('-use_attr', action='store_true',
                            help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)')
        parser.add_argument('-attr_loss_weight', default=1.0, type=float, help='weight for loss by predicting attributes')
        parser.add_argument('-no_img', action='store_true',
                            help='if included, only use attributes (and not raw imgs) for class prediction')
        parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
        parser.add_argument('-weighted_loss', default='', # note: may need to reduce lr
                            help='Whether to use weighted loss for single attribute or multiple ones')
        parser.add_argument('-uncertain_labels', action='store_true',
                            help='whether to use (normalized) attribute certainties as labels')
        parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES,
                            help='whether to apply bottlenecks to only a few attributes')
        parser.add_argument('-expand_dim', type=int, default=0,
                            help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only')
        parser.add_argument('-n_class_attr', type=int, default=2,
                            help='whether attr prediction is a binary or triary classification')
        parser.add_argument('-data_dir', default='official_datasets', help='directory to the training data')
        parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
        parser.add_argument('-resampling', help='Whether to use resampling', action='store_true')
        parser.add_argument('-train_addition', default='',type=str,
                            help='Catch-all argument to account for different training configs. Examples include "mse", "concept_loss", and "binary", and "alternate_loss"')
        parser.add_argument('-end2end', action='store_true',
                            help='Whether to train X -> A -> Y end to end. Train cmd is the same as cotraining + this arg')
        parser.add_argument('-optimizer', default='SGD', help='Type of optimizer to use, options incl SGD, RMSProp, Adam')
        parser.add_argument('-ckpt', default='', help='For retraining on both train + val set')
        parser.add_argument('-scheduler_step', type=int, default=1000,
                            help='Number of steps before decaying current learning rate by half')
        parser.add_argument('-normalize_loss', action='store_true',
                            help='Whether to normalize loss by taking attr_loss_weight into account')
        parser.add_argument('-use_relu', action='store_true',
                            help='Whether to include relu activation before using attributes to predict Y. '
                                 'For end2end & bottleneck model')
        parser.add_argument('-use_sigmoid', action='store_true',
                            help='Whether to include sigmoid activation before using attributes to predict Y. '
                                 'For end2end & bottleneck model')
        parser.add_argument('-connect_CY', action='store_true',
                            help='Whether to use concepts as auxiliary features (in multitasking) to predict Y')
        parser.add_argument('-num_classes', type=int,default=N_CLASSES,
                            help='How many classes there are for classification')
        parser.add_argument('-encoder_model',type=str,default='inceptionv3',
                           help='Which encoder model to use, inceptionv3 or small3')
        parser.add_argument('-expand_dim_encoder',type=int,default=0,
                           help='When using an MLP, what should the expand dim of the encoder be')
        parser.add_argument('-num_middle_encoder',type=int,default=0,
                           help='When using an MLP, how many dimensions does the middle layer have')
        parser.add_argument('-mask_loss_weight',type=float,default=1.0,
            help='Mask Loss Weight for Encoder Models with Mask')
        parser.add_argument('-load_model',type=str,default='none',
            help='Name of the pre-trained model to load; if applicable')
        parser.add_argument('-train_variation',type=str,default='none',
            help='Run the "half" training variation or the "loss" modification')
        args = parser.parse_args()
        args.three_class = (args.n_class_attr == 3)
        return (args,)
