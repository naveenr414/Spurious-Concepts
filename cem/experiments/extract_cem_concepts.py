import pytorch_lightning as pl
from cem.models.cem import ConceptEmbeddingModel
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.models import resnet50, vgg16, resnet34
from torch.utils.data import TensorDataset, DataLoader
from cem.data.CUB200.cub_loader import load_data, find_class_imbalance
import numpy as np
import argparse
import random
import cem.train.training as cem_train
import pytorch_lightning as pl
import cem.train.training as training


def c_extractor_arch(output_dim):
    """A feedforward architecture used before concept extraction 
    
    Returns: 
        A Torch Architecture which consists of a series of layers
    """
    
    return torch.nn.Sequential(*[
        torch.nn.Linear(2, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, output_dim),
    ])

def subsample_transform(sample):
    if isinstance(sample, list):
        sample = np.array(sample)
    return sample[selected_concepts]

def generate_data_loaders_synthetic(experiment_name):
    """Generate the train and validation dataloaders for the birds dataset
    
    Parameters: None
    
    Returns: Two Things
        train_dl: A PyTorch dataloader with data, output, and concepts
        valid_dl: A PyTorch dataloader with data, output, and concepts
    """

    cub_location = '../../../datasets/{}'.format(experiment_name)
    train_data_path = cub_location+'/preprocessed/train.pkl'
    valid_data_path = cub_location+'/preprocessed/val.pkl'
    test_data_path = cub_location+'/preprocessed/test.pkl'
    
    train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=cub_location+'/images',
        resampling=False,
        root_dir=cub_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    valid_dl = load_data(
        pkl_paths=[valid_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=cub_location+'/images',
        resampling=False,
        root_dir=cub_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    test_dl = load_data(
        pkl_paths=[test_data_path],
        use_attr=True,
        no_img=False,
        batch_size=64,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=cub_location+'/images',
        resampling=False,
        root_dir=cub_location,
        num_workers=num_workers,
        path_transform=lambda path: "../../../datasets/"+path
    )
    
    return train_dl, valid_dl, test_dl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CEM Concept Vectors')
    parser.add_argument('--experiment_name', type=str,
                        help='Name of the experiment we plan to run. Valid names include mnist, cub, and xor' )
    parser.add_argument('--num_gpus',type=int,
                        help='Number of GPUs to use when training',
                        default=0)
    parser.add_argument('--num_epochs',type=int,default=1,help='How many epochs to train for')
    parser.add_argument('--validation_epochs',type=int,default=1,help='How often should we run the validation script')
    parser.add_argument('--seed',type=int,default=42,help='Random seed for training')
    parser.add_argument('--num_workers',type=int,default=8,help='Number of workers')
    parser.add_argument('--sample_train',type=float,default=1.0,help='Fraction of the train dataset to sample')
    parser.add_argument('--sample_valid',type=float,default=1.0,help='Fraction of the valid dataset to sample')    
    parser.add_argument('--sample_test',type=float,default=1.0,help='Fraction of the test dataset to sample')    
    parser.add_argument('--concept_pair_loss_weight',type=float,default=0,help='Weight for the concept pair loss in the loss')
    parser.add_argument('--lr',type=float,default=0.01,help='Learning Rate for training')
    parser.add_argument('--weight_decay',type=float,default=4e-05,help="Weight decay regularization")
    parser.add_argument('--concept_loss_weight',type=float,default=5.0,help="How much emphasis to place on concept accuracy")

    args = parser.parse_args()
    
    experiment_name = args.experiment_name
    num_gpus = args.num_gpus
    num_epochs = args.num_epochs
    validation_epochs = args.validation_epochs
    seed = args.seed
    num_workers = args.num_workers
    lr = args.lr 
    weight_decay = args.weight_decay
    concept_loss_weight = args.concept_loss_weight

    pl.seed_everything(args.seed, workers=True)

    
    trainer = pl.Trainer(
            gpus=num_gpus,
    )
    
    suffix = ""
    
    existing_weights = ''
    if suffix == '_model_robustness':
        existing_weights = 'resnet_model_robustness.pt'
    elif suffix == '_model_responsiveness':
        existing_weights = 'resnet_model_responsiveness.pt'
    
    train_dl, valid_dl, test_dl = generate_data_loaders_synthetic(experiment_name)
    n_tasks = 2

    if experiment_name == "synthetic_1":
        n_concepts = 2
    elif experiment_name == "synthetic_2":
        n_concepts = 4
    elif experiment_name == "synthetic_4":
        n_concepts=8
    elif experiment_name == "synthetic_8":
        n_concepts=16
    elif experiment_name == "CUB":
        n_concepts=112
        n_tasks = 200
    elif experiment_name == "coco":
        n_concepts=10
        n_tasks = 2
    else:
        raise Exception("{} not found".format(experiment_name))
    
    imbalance = None 
    extractor_arch = "resnet34"

    config = dict(
        cv=5,
        max_epochs=num_epochs,
        patience=15,
        batch_size=128,
        num_workers=num_workers,
        emb_size=16,
        extra_dims=0,
        concept_loss_weight=concept_loss_weight,
        normalize_loss=False,
        learning_rate=lr,
        weight_decay=weight_decay,
        weight_loss=True,
        pretrain_model=True,
        c_extractor_arch=extractor_arch,
        optimizer="sgd",
        bool=False,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        early_stopping_delta=0.0,
        sampling_percent=1,
        momentum=0.9,
        validation_epochs=validation_epochs,
        shared_prob_gen=False,
        sigmoidal_prob=False,
        sigmoidal_embedding=False,
        training_intervention_prob=0.25,
        embeding_activation=None,
        concat_prob=False,
        seed=seed,
        concept_pair_loss_weight = args.concept_pair_loss_weight,
        existing_weights=""
    )
    config["architecture"] = "ConceptEmbeddingModel"
    config["extra_name"] = f"New"
    config["shared_prob_gen"] = True
    config["sigmoidal_prob"] = True
    config["sigmoidal_embedding"] = False
    config['training_intervention_prob'] = 0.25 
    config['concat_prob'] = False
    config['emb_size'] = config['emb_size']
    config["embeding_activation"] = "leakyrelu"  
    config["check_val_every_n_epoch"] = validation_epochs

    training.train_model(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        config=config,
        train_dl=train_dl,
        val_dl=valid_dl,
        test_dl=test_dl,
        split=0,
        result_dir="results/{}".format(experiment_name),
        rerun=False,
        project_name='concept_hierarchies_{}'.format(experiment_name),
        seed=42,
        activation_freq=0,
        single_frequency_epochs=0,
        imbalance=imbalance,
    )
    
#     cem_model = cem_train.construct_model(
#         n_concepts=n_concepts,
#         n_tasks=n_tasks,
#         config=og_config,
#         imbalance=None,
#         intervention_idxs=None,
#         adversarial_intervention=None,
#         active_intervention_values=None,
#         inactive_intervention_values=None,
#         c2y_model=False,
#     )
        
# #     cem_model = ConceptEmbeddingModel(
# #       n_concepts=n_concepts, # Number of training-time concepts
# #       n_tasks=n_tasks, # Number of output labels
# #       emb_size=16,
# #       concept_loss_weight=0.1,
# #       concept_pair_loss_weight = args.concept_pair_loss_weight,
# #       learning_rate=0.01,
# #       optimizer="adam",
# #       c_extractor_arch=extractor_arch, # Replace this appropriately
# #       training_intervention_prob=0.25, # RandInt probability
# #       experiment_name=experiment_name+suffix, 
# #       seed=seed, 
# #       existing_weights = existing_weights,
# #     )
    
#     trainer = pl.Trainer(
#         gpus=num_gpus,
#         max_epochs=num_epochs,
#         check_val_every_n_epoch=validation_epochs,
#     )

#     trainer.fit(cem_model, train_dl, valid_dl)
#     cem_model.write_concepts()
    
#     torch.save(cem_model.state_dict(), "cem_model.pt")

    
