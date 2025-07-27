#!/usr/bin/env python3
import argparse
import os
import subprocess
import secrets
import glob 
import json 
import shutil

dataset_folder = "datasets"

def run_command(command):
    """Run some command, print it's output, and any errors it produces
    
    Arguments:
        command: String of command to be run
        
    Returns: Return Code: Integer Return Code
    
    Side Effects: Runs and prints a bash command
    """
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True  # This allows decoding the output as text (str) instead of bytes (bytes)
    )

    # Read and print the output in real-time
    for line in process.stdout:
        print(line, end='')

    # Read and print the error in real-time
    for line in process.stderr:
        print(line, end='')

    # Wait for the process to finish and get the return code
    return_code = process.wait()

    return return_code

def delete_same_dict(save_data):
    """Delete any data + model that's running the exact same experiment, 
        so we can replace
        
    Arguments:
        save_data: Dictionary, with information such as seed on the experiment
        
    Returns: Nothing
    
    Side Effects: Deletes any model + data with the same seed, etc. parameters"""

    all_dicts = glob.glob("models/model_data/*.json")
    files_to_delete = []

    for file_name in all_dicts:
        json_file = json.load(open(file_name))

        if json_file == save_data:
            files_to_delete.append(file_name.split("/")[-1].replace(".json",""))

    print("Files to delete are {}".format(files_to_delete))
    
    for file_name in files_to_delete:
        try:
            os.remove("models/model_data/{}.json".format(file_name))
            shutil.rmtree("models/{}/{}".format(args.dataset,file_name))
        except:
            print("File {} doesn't exist".format(file_name))

def get_log_folder(args):
    """Determine the name of the folder where to store the model
    
    Arguents:
        args: Processed argparse argumetns
        
    Returns: String containing the name of the folder
    """

    rand_name = secrets.token_hex(4)
    save_data = vars(args)

    if save_data["concept_restriction"] == None:
        del save_data["concept_restriction"]

    if args.debugging:
        rand_name = 'debugging'
    elif args.load_model != 'none':
        rand_name = args.load_model.split("/")[-1].replace(".pt","")

    if args.load_model != 'none':
        load_folder = "/".join(args.load_model.split("/")[:-1])
        log_folder = "models/{}/{}".format(load_folder,rand_name)
    
    else:
        log_folder = "models/{}/{}".format(args.dataset,rand_name)
        if not args.debugging: 
            delete_same_dict(save_data)

        json.dump(save_data,open("models/model_data/{}.json".format(rand_name),"w"))

    return log_folder
        
def main(args):
    """Run our main arguments for the appropriate CBM model
    
    Arguments:
        args: Processed argparse arguments
        
    Returns: Nothing
    
    Side Effects: Trains the CBM model, stores it in results"""
    
    dataset = args.dataset
    model_type = args.model_type
    num_attributes = args.num_attributes
    num_classes = args.num_classes
    seed = args.seed
    epochs = args.epochs
    learning_rate=args.lr
    encoder_model = args.encoder_model
    weight_decay = args.weight_decay
    attr_loss_weight = args.attr_loss_weight
    optimizer = args.optimizer
    expand_dim_encoder = args.expand_dim_encoder
    num_middle_encoder = args.num_middle_encoder
    mask_loss_weight = args.mask_loss_weight
    load_model = args.load_model
    train_variation = args.train_variation
    scale_lr = args.scale_lr
    scale_factor = args.scale_factor
    adversarial_epsilon = args.adversarial_epsilon
    adversarial_weight = args.adversarial_weight 
    scheduler_step = args.scheduler_step
    pretrained = args.pretrained 
    one_batch = args.one_batch
    scheduler = args.scheduler
    concept_restriction = args.concept_restriction
    use_residual = args.use_residual

    os.makedirs(f"results/{dataset}", exist_ok=True)

    log_folder = get_log_folder(args)

    if model_type == "independent":
        cmd1 = (
            f"python3 locality/cbm_variants/ConceptBottleneck/experiments.py cub Independent_CtoY --seed {seed} -log_dir {log_folder}/bottleneck "
            f"-e {epochs} -use_attr -optimizer {optimizer} -data_dir {dataset_folder}/{dataset}/preprocessed "
            f"-n_attributes {num_attributes} -no_img -b 64 -weight_decay 0.00005 -lr 0.01 -scheduler_step 100 "
            f"-num_classes {num_classes} -encoder_model {encoder_model} "
            f"-expand_dim_encoder {expand_dim_encoder} -num_middle_encoder {num_middle_encoder} -mask_loss_weight {mask_loss_weight}"
        )
        cmd2 = (
            f"python3 locality/cbm_variants/ConceptBottleneck/experiments.py cub Concept_XtoC --seed {seed} -ckpt 1 -log_dir {log_folder}/concept "
            f"-e {epochs} -optimizer {optimizer} -pretrained -use_aux -use_attr -weighted_loss multiple "
            f"-data_dir {dataset_folder}/{dataset}/preprocessed -n_attributes {num_attributes} -normalize_loss "
            f"-b 64 -weight_decay 0.00004 -lr 0.01 -encoder_model {encoder_model} -num_classes {num_classes} "
            f"-scheduler_step 100 -bottleneck"
        )
        
        run_command(cmd2)
        run_command(cmd1)

    elif model_type == "independent_encoder":
        cmd1 = (
            f"python3 locality/cbm_variants/ConceptBottleneck/experiments.py cub Independent_CtoY --seed {seed} -log_dir {log_folder}/bottleneck "
            f"-e {epochs} -use_attr -optimizer {optimizer} -data_dir {dataset_folder}/{dataset}/preprocessed "
            f"-n_attributes {num_attributes} -no_img -b 64 -weight_decay 0.00005 -lr 0.01 -scheduler_step 100 "
            f"-num_classes {num_classes} -encoder_model {encoder_model} "
            f"-expand_dim_encoder {expand_dim_encoder} -num_middle_encoder {num_middle_encoder} -mask_loss_weight {mask_loss_weight}"
        )
        run_command(cmd1)


    elif model_type == "sequential":
        cmd1 = (
            f"python3 locality/cbm_variants/ConceptBottleneck/experiments.py cub Concept_XtoC --seed {seed} -ckpt 1 -log_dir {log_folder}/concept "
            f"-e {epochs} -optimizer {optimizer} -pretrained -use_aux -use_attr -weighted_loss multiple "
            f"-encoder_type {encoder_model} -data_dir {dataset_folder}/{dataset}/preprocessed "
            f"-n_attributes {num_attributes} -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 "
            f"-scheduler_step 100 -encoder_model {encoder_model} -num_classes {num_classes} -bottleneck"
        )
        cmd2 = (
            f"python3 CUB/generate_new_data.py ExtractConcepts "
            f"--model_path results/{dataset}/sequential/concept/best_model_42.pth "
            f"--data_dir {dataset_folder}/{dataset}/preprocessed -num_classes {num_classes} "
            f"--out_dir results/{dataset}/sequential/output/"
        )
        cmd3 = (
            f"python3 locality/cbm_variants/ConceptBottleneck/experiments.py cub Sequential_CtoY --seed {seed} -log_dir {log_folder}/bottleneck "
            f"-e {epochs} -use_attr -optimizer {optimizer} -data_dir results/{dataset}/sequential/output "
            f"-n_attributes {num_attributes} -num_classes {num_classes} -no_img -b 64 -weight_decay 0.00005 "
            f"-encoder_model {encoder_model} -lr 0.001 -scheduler_step 100"
        )
        subprocess.check_output([cmd1], shell=True,stderr=subprocess.STDOUT)
        subprocess.check_output([cmd2], shell=True,stderr=subprocess.STDOUT)
        subprocess.check_output([cmd3], shell=True,stderr=subprocess.STDOUT)

    elif model_type == "joint":
        if pretrained: 
            pretrained = " -pretrained"
        else:
            pretrained = ""
        
        if one_batch:
            one_batch = " -one_batch"
        else:
            one_batch = ""

        if use_residual:
            residual = " -use_residual"
        else:
            residual = ""

        if concept_restriction:
            concept_restriction = "-concept_restriction {}".format(" ".join([str(i) for i in concept_restriction]))
        else:
            concept_restriction = ""

        cmd = (
            f"python3 locality/cbm_variants/ConceptBottleneck/experiments.py cub Joint --seed {seed} -ckpt 0 -log_dir {log_folder}/{model_type} "
            f"-e {epochs} -optimizer {optimizer} {one_batch} {pretrained} -use_aux -weighted_loss multiple -use_attr "
            f"-data_dir {dataset_folder}/{dataset}/preprocessed -n_attributes {num_attributes} "
            f"-attr_loss_weight {attr_loss_weight} -normalize_loss -b 32 -weight_decay {weight_decay} -num_classes {num_classes} "
            f"-lr {learning_rate} -encoder_model {encoder_model} -scheduler_step {scheduler_step} -end2end -use_sigmoid "
            f"-expand_dim_encoder {expand_dim_encoder} -num_middle_encoder {num_middle_encoder} -mask_loss_weight {mask_loss_weight} "
            f"-load_model {load_model} -train_variation {train_variation} -scale_lr {scale_lr} -scheduler {scheduler} -scale_factor {scale_factor} "
            f"-adversarial_epsilon {adversarial_epsilon} -adversarial_weight {adversarial_weight} {concept_restriction} {residual}"
        )
        
        run_command(cmd)
            
    else:
        print(f"Model type {model_type} not found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description.")
    parser.add_argument("-dataset", type=str, help="Dataset name",default='synthetic_2')
    parser.add_argument("-model_type", type=str, help="Model type (independent, sequential, or joint)", default='joint')
    parser.add_argument("-num_attributes", type=int, help="Number of attributes",default=4)
    parser.add_argument("-num_classes", type=int, help="Number of classes",default=2)
    parser.add_argument("-seed", type=int, help="Random seed",default=42)
    parser.add_argument("-epochs", type=int, help="Number of epochs",default=25)
    parser.add_argument("-lr",type=float,help="Learning rate",default=0.05)
    parser.add_argument("--encoder_model", type=str, help="Encoder type", default="inceptionv3")
    parser.add_argument("--weight_decay", type=float, help="Weight decay", default=0.0004)
    parser.add_argument('--attr_loss_weight',type=float,default=1.0,help='Amount of weight to put on the concept loss')
    parser.add_argument('--optimizer',type=str,default='sgd',help='Which optimizer to use')
    parser.add_argument('--expand_dim_encoder',type=int,help="Expand Dim for the encoder MLP",default=0)
    parser.add_argument('--num_middle_encoder',type=int,help="Middle Dimension for the encoder MLP",default=0)
    parser.add_argument('--mask_loss_weight',type=float,default=1.0,help="Strength of mask weight")
    parser.add_argument('--load_model',type=str,default='none',help="Load in a pretrained model")
    parser.add_argument('--train_variation',type=str,default='none',help='Run the "half" training variation or the "loss" modification')
    parser.add_argument('--debugging', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--scale_lr',default=5,type=int)
    parser.add_argument('--scale_factor',default=1.5,type=float)
    parser.add_argument('--adversarial_epsilon',default=0.01,type=float)
    parser.add_argument('--adversarial_weight',default=0.25,type=float)
    parser.add_argument('--scheduler_step',default=30,type=int,help="How often to decrease the LR by a factor of 10")
    parser.add_argument('--one_batch',action='store_true',help="Should we only train on one batch?")
    parser.add_argument('--use_residual',action='store_true',help="Should we add a residual layer?")
    parser.add_argument('--scheduler',type=str,default='none',help="'none' or 'cyclic'")
    parser.add_argument('--concept_restriction',nargs='+',type=int,help="List of concept combinations to use when training")

    args = parser.parse_args()
    main(args)