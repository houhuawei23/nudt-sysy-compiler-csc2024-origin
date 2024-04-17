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
            # print(data)
            # pprint.pprint(data)

        except yaml.YAMLError as exc:
            print(exc)
    target_name = isa_data["Target"]["name"]
    templates = isa_data["Templates"]
    instances = isa_data["Instances"]
    # insts = dict()
    for name, info in instances.items():
        info["name"] = name
        template = templates[info["template"]]
        format = copy.deepcopy(template["format"])
        format[0] = info["mnem"]
        info["format"] = format
        info["operands"] = copy.deepcopy(template["operands"])
        # print(info)
    return target_name, instances
    # print(yaml.dump(isa_data, sort_keys=False, default_flow_style=False))


if __name__ == "__main__":
    isa_data_yml = sys.argv[1]
    output_dir = sys.argv[2]
    target_name, instances = load_inst_info(isa_data_yml)
    params = {"target_name": target_name, "insts": instances}
    gen_file_jinja2("InstInfoDecl.hpp.jinja2", output_dir, params)
    gen_file_jinja2("InstInfoImpl.hpp.jinja2", output_dir, params)
    # with open(isa_data_yml, 'r') as f:
    #     isa_data = yaml.load(f, Loader=yaml.FullLoader)
    # print(isa_data)
    # env = jinja2.Environment(loader=FileSystemLoader("."))
    # template = env.get_template("test.jinja2")
    # print(template.render(target="RISCV"))
    # template = jinja2.Template(open("test.jinja2").read())
