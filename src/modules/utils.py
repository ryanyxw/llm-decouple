import os
import shutil
import wandb



def confirm_with_user(message):
    print(message)
    print('Are you sure? (y/n)')
    response = input()
    return response.lower() == 'y'

def prepare_folder(file_path, isFile=True):
    """Prepare a folder for a file"""
    import os
    if (isFile):
        folder = os.path.dirname(file_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
    else:
        if not os.path.exists(file_path):
            os.makedirs(file_path)

def get_md5(string):
    """Get the md5 hash of a string"""
    import hashlib
    return hashlib.md5(string.encode()).hexdigest()

def get_hash(args):
    """Get a hash of the arguments"""
    return get_md5(str(args))

def load_config(config_fn):
    """loads a yml config file into a object with attributes"""
    from omegaconf import OmegaConf

    import os

    if not os.path.exists(config_fn):
        raise FileNotFoundError(f"Config file {config_fn} does not exist")
    return OmegaConf.load(config_fn)

def save_config(config, config_fn):
    """Saves a config object to a yml file"""
    from omegaconf import OmegaConf

    OmegaConf.save(config, config_fn)

def validate_inputs(configs):
    """recursively validate the inputs and outputs. Assumes that all inputs start with 'input' and all outputs start with 'output'"""
    import os

    def recurse_keys(subconfig):
        if ("do" in subconfig and not subconfig["do"]):
            return
        for key, value in subconfig.items():
            # if the value is an omegaconf object
            if hasattr(value, "items"):
                recurse_keys(value)
            elif key[:5] == "input":
                if not os.path.exists(value):
                    raise FileNotFoundError(f"{key} {value} does not exist")
            elif key[:6] == "output":
                if os.path.exists(value):
                    if confirm_with_user(f"Output {value} already exists. Do you want to overwrite it?"):
                        shutil.rmtree(value)
                isFile = len(value.split("/")[-1].split(".")) > 1
                prepare_folder(value, isFile=isFile)

    recurse_keys(configs)


def prepare_wandb():
    os.environ["WANDB_PROJECT"] = "decouple"