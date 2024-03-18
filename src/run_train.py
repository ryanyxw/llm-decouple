import argparse
import os
import shutil
from utils import confirm_with_user, get_hash
from modeling.configs import update_configs
from random import randint


def validate_inputs(args):
    if not os.path.exists(args.target):
        raise ValueError(f"data source {args.target} does not exist")
    if os.path.exists(args.destination):
        message = f"model output dir {args.destination} already exists. Delete? "
        if confirm_with_user(message):
            shutil.rmtree(args.destination)
        else:
            raise ValueError(f"model output path {args.destination} already exists")

    #check the configs exist
    if not os.path.exists(args.path_to_model_yaml):
        raise ValueError(f"model yaml {args.path_to_model_yaml} does not exist")
    if not os.path.exists(args.path_to_setup_yaml):
        raise ValueError(f"setup yaml {args.path_to_setup_yaml} does not exist")


    #check that per gpu batch size works
    if args.train_batch_size % args.global_num_gpus != 0:
        raise ValueError(f"train_batch_size {args.train_batch_size} is not divisible by global_num_gpus {args.global_num_gpus}")
    if args.train_batch_size % (args.train_micro_batch_size_per_gpu * args.global_num_gpus) != 0:
        raise ValueError(f"train_batch_size {args.train_batch_size} is not divisible by train_micro_batch_size_per_gpu * global_num_gpus {args.train_micro_batch_size_per_gpu} * {args.global_num_gpus}")

def update_experiment_configs(args, temp_model_file, temp_setup_file):

    # for batch size configs
    train_args = dict()

    train_args["global_num_gpus"] = args.global_num_gpus
    train_args["train_batch_size"] = args.train_batch_size
    train_args["train_micro_batch_size_per_gpu"] = args.train_micro_batch_size_per_gpu
    train_args["gradient_accumulation_steps"] = args.train_batch_size // args.global_num_gpus // args.train_micro_batch_size_per_gpu
    train_args["seq_length"] = args.seq_length

    train_args["train_iters"] = args.train_iters
    train_args["lr_decay_iters"] = args.train_iters
    train_args["checkpoint_factor"] = args.train_iters - 1
    train_args["eval_iters"] = args.eval_iters
    train_args["eval_interval"] = args.eval_interval
    train_args["log_interval"] = args.log_interval
    train_args["steps_per_print"] = train_args["log_interval"]

    update_configs(temp_model_file, **train_args)


    # for the local_setup.yml
    configs_args = dict()

    configs_args["train_data_paths"] = [os.path.join(args.target, "tokenized_text_document")]
    configs_args["label_data_paths"] = [os.path.join(args.target, "tokenized_label_document")]

    configs_args["save"] = args.destination
    configs_args["load"] = args.destination

    configs_args["wandb_project"] = "decouple"
    configs_args["wandb_group"] = args.wandb_group
    # configs_args["include"] = args.include

    configs_args["master_port"] = 29500 + randint(1, 10000)

    update_configs(temp_setup_file, **configs_args)

def main(args):
    setattr(args, "target", os.path.join(args.data_in_dir, args.input_name))
    setattr(args, "destination", os.path.join(args.model_out_dir, args.input_name))
    validate_inputs(args)

    # create temporary file for configs
    temp_config_dir = os.path.join(args.CONFIG_DIR, "temp", args.input_name + "_" + get_hash(args))
    os.makedirs(temp_config_dir, exist_ok=True)
    temp_model_file = os.path.join(temp_config_dir, "model.yml")
    temp_setup_file = os.path.join(temp_config_dir, "local_setup.yml")

    #copy the original files to the temp folder
    shutil.copy(args.path_to_model_yaml, temp_model_file)
    shutil.copy(args.path_to_setup_yaml, temp_setup_file)

    #update the configs for this experimental run
    update_experiment_configs(args, temp_model_file, temp_setup_file)

    #call the training script
    print("executing command...")
    cmd = f"python {args.NEOX_DIR}/deepy.py {args.NEOX_DIR}/train.py -d {temp_config_dir} model local_setup"


    # run the command and check for errors
    status = os.system(cmd)
    if (status != 0):
        return

    print("yay!")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_in_dir",
        type=str,
        required=True,
        help="(input) the directory that stores the input files"
    )
    parser.add_argument(
        "--input_name",
        type=str,
        required=True,
        help="(input) the name of the file in target_dir"
    )

    parser.add_argument(
        "--model_out_dir",
        type=str,
        required=True,
        help="(output) path to the destination folder"
    )



    parser.add_argument(
        '--path_to_model_yaml',
        required=True,
        help="the path to the model yaml"
    )
    parser.add_argument(
        '--path_to_setup_yaml',
        required=True,
        help="the path to the local_setup yaml"
    )

    parser.add_argument(
        '--global_num_gpus',
        required=True,
        type=int,
        help="global_num_gpus"
    )

    parser.add_argument(
        '--train_batch_size',
        required=True,
        type=int,
        help="train_batch_size"
    )

    parser.add_argument(
        '--train_micro_batch_size_per_gpu',
        required=True,
        type=int,
        help="train_micro_batch_size_per_gpu"
    )

    parser.add_argument(
        '--train_iters',
        required=True,
        type=int,
        help="train_iters -> look at how long dataset is"
    )
    parser.add_argument(
        '--eval_iters',
        required=True,
        type=int,
        help="number of times to run the eval dataset for"
    )

    parser.add_argument(
        '--eval_interval',
        required=True,
        type=int,
        help="the number of train steps in between each eval run"
    )

    parser.add_argument(
        '--log_interval',
        required=True,
        type=int,
        help="the number of train steps in between each log"
    )

    parser.add_argument(
        '--seq_length',
        required=True,
        type=int,
        help="how many sequences per document that the model will use to train"
    )

    # parser.add_argument(
    #     '--include',
    #     required=True,
    #     # default="localhost:0",
    #     help="GPU numbers"
    # )
    parser.add_argument(
        '--wandb_group',
        required=True,
        help="Name of the wandb group folder"
    )

    parser.add_argument(
        "--NEOX_DIR",
        type=str,
        required=True,
        help="(input) path to the Neox directory"
    )

    parser.add_argument(
        "--DATA_DIR",
        type=str,
        required=True,
        help="(input) path to the data directory"
    )
    parser.add_argument(
        "--CONFIG_DIR",
        type=str,
        required=True,
        help="(input) path to the config folder"
    )


    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    main(args)