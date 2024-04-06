def confirm_with_user(message):
    print(message)
    print('Are you sure? (y/n)')
    response = input()
    return response.lower() == 'y'

def prepare_folder(file_path):
    """Prepare a folder for a file"""
    import os
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_md5(string):
    """Get the md5 hash of a string"""
    import hashlib
    return hashlib.md5(string.encode()).hexdigest()

def get_hash(args):
    """Get a hash of the arguments"""
    return get_md5(str(args))


class Configs:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_config(args):
    """loads a yml config file into a object with attributes"""
    import yaml

    configs = dict()

    with open(args, 'r') as f:
        conf = yaml.safe_load(f)

    for key, value in conf.items():
        configs[key] = value

    return Configs(**configs)