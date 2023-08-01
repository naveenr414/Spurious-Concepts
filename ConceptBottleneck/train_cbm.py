#!/usr/bin/env python3
import argparse
import os
import subprocess

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


def get_log_folder(args):
    """Determine the name of the folder where to store the model
    
    Arguents:
        args: Processed argparse argumetns
        
    Returns: String containing the name of the folder
    """
    
    if args.weight_decay == 0.0004 and args.encoder_model == 'inceptionv3':
        log_folder = f"results/{args.dataset}/{args.model_type}"
    elif args.encoder_model == 'inceptionv3':
        log_folder = f"results/{args.dataset}/{args.model_type}_wd_{args.weight_decay}"
    elif args.weight_decay == 0.0004:
        log_folder = f"results/{args.dataset}/{args.model_type}_model_{args.encoder_model}"
    else:
        log_folder = f"results/{args.dataset}/{args.model_type}_model_{args.encoder_model}_wd_{args.weight_decay}"
    
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

    os.makedirs(f"results/{dataset}", exist_ok=True)

    log_folder = get_log_folder(args)

    if model_type == "independent":
        cmd1 = (
            f"python3 experiments.py cub Independent_CtoY --seed {seed} -log_dir {log_folder}/bottleneck "
            f"-e {epochs} -use_attr -optimizer sgd -data_dir ../../cem/cem/{dataset}/preprocessed "
            f"-n_attributes {num_attributes} -no_img -b 64 -weight_decay 0.00005 -lr 0.01 -scheduler_step 100 "
            f"-num_classes {num_classes} -encoder_model {encoder_model}"
        )
        cmd2 = (
            f"python3 experiments.py cub Concept_XtoC --seed {seed} -ckpt 1 -log_dir {log_folder}/concept "
            f"-e {epochs} -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple "
            f"-data_dir ../../cem/cem/{dataset}/preprocessed -n_attributes {num_attributes} -normalize_loss "
            f"-b 64 -weight_decay 0.00004 -lr 0.01 -encoder_model {encoder_model} -num_classes {num_classes} "
            f"-scheduler_step 100 -bottleneck"
        )
        run_command(cmd1)
        run_command(cmd2)


    elif model_type == "sequential":
        cmd1 = (
            f"python3 experiments.py cub Concept_XtoC --seed {seed} -ckpt 1 -log_dir {log_folder}/concept "
            f"-e {epochs} -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple "
            f"-encoder_type {encoder_model} -data_dir ../../cem/cem/{dataset}/preprocessed "
            f"-n_attributes {num_attributes} -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 "
            f"-scheduler_step 100 -encoder_model {encoder_model} -num_classes {num_classes} -bottleneck"
        )
        cmd2 = (
            f"python3 CUB/generate_new_data.py ExtractConcepts "
            f"--model_path results/{dataset}/sequential/concept/best_model_42.pth "
            f"--data_dir ../../cem/cem/{dataset}/preprocessed -num_classes {num_classes} "
            f"--out_dir results/{dataset}/sequential/output/"
        )
        cmd3 = (
            f"python3 experiments.py cub Sequential_CtoY --seed {seed} -log_dir {log_folder}/bottleneck "
            f"-e {epochs} -use_attr -optimizer sgd -data_dir results/{dataset}/sequential/output "
            f"-n_attributes {num_attributes} -num_classes {num_classes} -no_img -b 64 -weight_decay 0.00005 "
            f"-encoder_model {encoder_model} -lr 0.001 -scheduler_step 100"
        )
        subprocess.check_output([cmd1], shell=True,stderr=subprocess.STDOUT)
        subprocess.check_output([cmd2], shell=True,stderr=subprocess.STDOUT)
        subprocess.check_output([cmd3], shell=True,stderr=subprocess.STDOUT)

    elif model_type == "joint":
        cmd = (
            f"python3 experiments.py cub Joint --seed {seed} -ckpt 1 -log_dir {log_folder}/{model_type} "
            f"-e {epochs} -optimizer sgd -pretrained -use_aux -weighted_loss multiple -use_attr "
            f"-data_dir ../../cem/cem/{dataset}/preprocessed -n_attributes {num_attributes} "
            f"-attr_loss_weight {attr_loss_weight} -normalize_loss -b 64 -weight_decay {weight_decay} -num_classes {num_classes} "
            f"-lr {learning_rate} -encoder_model {encoder_model} -scheduler_step 30 -end2end -use_sigmoid"
        )
        
        run_command(cmd)
                
#         try:
#             subprocess.check_output([cmd], shell=True,stderr=subprocess.STDOUT)
#         except subprocess.CalledProcessError as e:
#             print(str(e.output).replace("\\n","\n"))
#             raise RuntimeError("command '{}' return with error (code {})".format(e.cmd, e.returncode))

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
    args = parser.parse_args()
    main(args)