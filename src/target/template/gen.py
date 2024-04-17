import os
import sys
import yaml
import jinja2
import copy
from jinja2 import Environment, FileSystemLoader

# import typing
from typing import List, Dict, Tuple


def gen_file_jinja2(template_file, output_dir, params):
    env = jinja2.Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    template = env.get_template(template_file)
    output_file = os.path.join(
        output_dir, os.path.basename(template_file).replace(".jinja2", "")
    )
    with open(output_file, "w") as f:
        res = template.render(params)
        f.write(res)
    os.system("clang-format -i {}".format(output_file))


"""
{
    "ADD": {
        "format": ["add", 0, ", ", 1, ", ", 2],
        "operands": {
            0: { "name": "rd", "type": "GPR", "flag": "Def"},
            1: { "name": "rs1", "type": "GPR", "flag": "Use"},
            2: { "name": "rs2", "type": "GPR", "flag": "Use"}
        },
        "comment": []
    }
}

"""
# def new_inst_from_template(template: List[str]):


def load_inst_info(file):
    with open(file, "r") as stream:
        try:
            isa_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    target_name = isa_data["Target"]["name"]
    isa_instinfo = isa_data["InstInfo"]
    templates = isa_instinfo["Templates"]
    instances = isa_instinfo["Instances"]

    insts = list()

    for template_name, template_info in templates.items():
        template_instances = template_info.get("instances", [])
        for name in template_instances:
            info = dict()
            info["name"] = name
            info["format"] = copy.deepcopy(template_info["format"])
            info["format"][0] = name
            info["operands"] = copy.deepcopy(template_info["operands"])
            insts.append(info)
    for name, info in instances.items():
        info["name"] = name
        if "template" in info:
            template = templates[info["template"]]
            format = copy.deepcopy(template["format"])
            format[0] = info["mnem"]
            info["format"] = format
            info["operands"] = copy.deepcopy(template["operands"])
        elif "format" in info:
            pass
        else:
            print("Error: no format or template for instance {}".format(name))

        insts.append(info)
    return target_name, insts


def load_isel_info(file):
    with open(file, "r") as stream:
        try:
            isa_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    target_name = isa_data["Target"]["name"]
    isel_info = isa_data["InstSelInfo"]
    return isel_info

if __name__ == "__main__":
    isa_data_yml = sys.argv[1]
    output_dir = sys.argv[2]
    target_name, insts = load_inst_info(isa_data_yml)
    params = {"target_name": target_name, "insts": insts}
    gen_file_jinja2("InstInfoDecl.hpp.jinja2", output_dir, params)
    gen_file_jinja2("InstInfoImpl.hpp.jinja2", output_dir, params)

    # isel_info = load_isel_info(isa_data_yml)
    # params = {"target_name": target_name, "isel_info": isel_info}
    # gen_file_jinja2("ISelInfoDecl.hpp.jinja2", output_dir, params)
    # gen_file_jinja2("ISelInfoImpl.hpp.jinja2", output_dir, params)
