import os
import sys
import yaml
import jinja2
import copy
from jinja2 import Environment, FileSystemLoader

# import typing
from typing import List, Dict, Tuple

def load_inst_info(file: str) -> Tuple[str, Dict, List]:
    """load inst_info.yml, parse it to insts_dict and insts_list.
    - args:
        - file: inst_info.yml path
    - returns:
        - target_name: target name
        - insts_dict: insts_dict, key is inst name, value is inst info
        - insts_list: list of inst name, determine enum order in InstInfoDecl.hpp

    ```
    target_name: RISCV
    insts_dict: {
        'Add': {
            'name': 'Add',
            'format': ['add', ' ', 0, ', ', 1, ', ', 2],
            'operands': {
                0: {'name': 'dst', 'type': 'INTREG', 'flag': 'Def'},
                1: {'name': 'src1', 'type': 'INTVAL', 'flag': 'Use'},
                2: {'name': 'src2', 'type': 'INTVAL', 'flag': 'Use'}
            }
        },
        'Sub': { ... },
        'Mul': { ... },
        ...
    }
    insts_list: ['Add', 'Sub', 'Mul', ...]
    ```
    """
    target_name = ""
    inst_info_dict = dict()
    inst_name_list = list()

    with open(file, "r") as stream:
        try:
            isa_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    target_name = isa_data["Target"]["name"]
    isa_instinfo = isa_data["InstInfo"]
    templates = isa_instinfo["Templates"]
    instances = isa_instinfo["Instances"]

    # parse templates
    for template_name, template_info in templates.items():
        template_instances = template_info.get("instances", [])
        for name in template_instances:
            info = {
                "name": name,
                "format": copy.deepcopy(template_info["format"]),
                "operands": copy.deepcopy(template_info["operands"]),
            }
            info["format"][0] = name
            inst_info_dict[name] = info

    # parse instances
    for name, info in instances.items():
        info["name"] = name
        if "template" in info:
            template = templates[info["template"]]
            info["format"] = copy.deepcopy(template["format"])
            info["operands"] = copy.deepcopy(template["operands"])

            info["format"][0] = info["mnem"] #! mnem
        elif "format" in info:
            pass
        else:
            print("Error: no format or template for instance {}".format(name))

        inst_info_dict[name] = info

    if "InstList" in isa_instinfo:
        inst_name_list = isa_instinfo["InstList"]
    else:
        inst_name_list = list(inst_info_dict.keys())
    return target_name, inst_info_dict, inst_name_list


def get_mirgen_insts(gen_data_yml: str) -> Tuple[Dict, List]:
    """Load MIR Generic Insts Info from gen_data.yml.
    Format of inst name is `InstXXX`.
    - args:
        - gen_data_yml: gen_data.yml path
    - returns:
        - mirgen_insts_dict: inst_name -> inst_info
        - mirgen_insts_list: inst_name list
    """
    _, gen_insts_dict, gen_insts_list = load_inst_info(gen_data_yml)
    mirgen_insts_list = ["Inst" + name for name in gen_insts_list]
    mirgen_insts_dict = dict()
    for gen_inst_name, gen_inst_info in gen_insts_dict.items():
        mirgen_inst_name = "Inst" + gen_inst_name
        mirgen_inst_info = copy.deepcopy(gen_inst_info)
        mirgen_inst_info["name"] = mirgen_inst_name
        mirgen_insts_dict[mirgen_inst_name] = mirgen_inst_info
    return mirgen_insts_dict, mirgen_insts_list
