import argparse
import sys
import os
import yaml

#this edits a yaml file with certain parameters
def edit_yaml(path_to_yaml, **kwargs):
    print(path_to_yaml)
    with open(path_to_yaml) as f:
        list_doc = yaml.safe_load(f)
    for k, v in kwargs.items():
        list_doc[k] = v
    with open(path_to_yaml, 'w') as f:
        yaml.dump(list_doc, f, default_flow_style=False)


def update_configs(path, **kwargs):
    update_configs_model = dict()

    #loop through all arguments in kwargs and update the model yaml
    for k, v in kwargs.items():
        update_configs_model[k] = v


    edit_yaml(path, **update_configs_model)
