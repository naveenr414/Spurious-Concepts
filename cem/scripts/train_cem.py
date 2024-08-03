#!/usr/bin/env python3
import argparse
import os
import subprocess
import secrets
import glob 
import json 
import shutil

dataset_folder = "../../../datasets"

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

    all_dicts = glob.glob("../models/model_data/*.json")
    files_to_delete = []

    for file_name in all_dicts:
        json_file = json.load(open(file_name))

        if json_file == save_data:
            files_to_delete.append(file_name.split("/")[-1].replace(".json",""))

    print("Files to delete are {}".format(files_to_delete))
    
    for file_name in files_to_delete:
        try:
            # TODO: Uncomment this
            print("Deleting {}".format(file_name))
            # os.remove("../models/model_data/{}.json".format(file_name))
            # shutil.rmtree("../models/{}/{}".format(args.dataset,file_name))
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

    log_folder = "../models/{}/{}".format(args.dataset,rand_name)
    delete_same_dict(save_data)

    json.dump(save_data,open("../models/model_data/{}.json".format(rand_name),"w"))

    return log_folder
        
def main(args):
    """Run our main arguments for the appropriate CBM model
    
    Arguments:
        args: Processed argparse arguments
        
    Returns: Nothing
    
    Side Effects: Trains the CBM model, stores it in results"""
    
    dataset = args.dataset
    seed = args.seed
    epochs = args.epochs
    learning_rate=args.lr
    concept_loss_weight=args.concept_loss_weight

    os.makedirs(f"../results/{dataset}", exist_ok=True)

    log_folder = get_log_folder(args)

    cmd1 = (
        f"LD_LIBRARY_PATH=../../anaconda3/lib python experiments/extract_cem_concepts.py "
        f"--experiment_name {dataset} --num_gpus 1 --num_epochs {epochs} --validation_epochs 25 "
        f"--seed {seed} --concept_pair_loss_weight 0 --concept_loss_weight {concept_loss_weight} --lr {learning_rate} "
    )

    print(cmd1)

        
    run_command(cmd1)
    # run_command("mv results/ ../models/synthetic_object/{}/{}/joint/best_model_{}.pth".format())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description.")
    parser.add_argument("-dataset", type=str, help="Dataset name",default='synthetic_2')
    parser.add_argument("-seed", type=int, help="Random seed",default=42)
    parser.add_argument("-epochs", type=int, help="Number of epochs",default=25)
    parser.add_argument("-lr",type=float,help="Learning rate",default=1e-3)
    parser.add_argument("-concept_loss_weight",type=float,help="Weight on concept loss",default=0.5)
    parser.add_argument("-model_type",type=str,help="CEM models",default="cem")

    args = parser.parse_args()
    main(args)