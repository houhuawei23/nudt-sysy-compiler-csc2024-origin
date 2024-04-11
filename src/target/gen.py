import os
import sys
import yaml
import jinja2
from jinja2 import Environment, FileSystemLoader


def gen_from_template(template_file, output_dir, params):
    env = jinja2.Environment(loader=FileSystemLoader("."))
    template = env.get_template(template_file)
    output_file = os.path.join(
        output_dir, os.path.basename(template_file).replace(".jinja2", "")
    )
    with open(output_file, "w") as f:
        f.write(template.render(params))

# def load_inst_info(file):

with open("riscv.yml", "r") as stream:
    try:
        data = yaml.safe_load(stream)
        # print(data)
        # pprint.pprint(data)
        print(yaml.dump(data, sort_keys=False, default_flow_style=False))

    except yaml.YAMLError as exc:
        print(exc)

# if __name__ == "__main__":
#     isa_data_yml = sys.argv[1]
#     # output_dir = sys.argv[2]

#     with open(isa_data_yml, 'r') as f:
#         isa_data = yaml.load(f, Loader=yaml.FullLoader)
#     print(isa_data)
#     # env = jinja2.Environment(loader=FileSystemLoader("."))
    # template = env.get_template("test.jinja2")
    # print(template.render(target="RISCV"))
    # template = jinja2.Template(open("test.jinja2").read())
