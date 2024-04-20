import os
import sys
import yaml
import jinja2
import copy
from jinja2 import Environment, FileSystemLoader

# import typing
from typing import List, Dict, Tuple

global_idx = 0


def get_id():
    global global_idx
    global_idx += 1
    return global_idx


# in matchAndSelect function
operand_idx = 0


def get_operand_id():
    global operand_idx
    operand_idx += 1
    return operand_idx


def reset_operand_idx():
    global operand_idx
    operand_idx = 0


inst_idx = 0


def get_inst_id():
    global inst_idx
    inst_idx += 1
    return inst_idx


def reset_inst_idx():
    global inst_idx
    inst_idx = 0


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
    insts_list = isa_instinfo["InstList"]
    insts = list()
    insts_dict = dict()
    for template_name, template_info in templates.items():
        template_instances = template_info.get("instances", [])
        for name in template_instances:
            info = dict()
            info["name"] = name
            info["format"] = copy.deepcopy(template_info["format"])
            info["format"][0] = name
            info["operands"] = copy.deepcopy(template_info["operands"])
            # insts.append(info)
            insts_dict[name] = info
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

        # insts.append(info)
        insts_dict[name] = info

    return target_name, insts_dict, insts_list


"""
isel_info: {
    'InstAdd' : [
        {
            pattern: ...,
            replace: ...
        },
        {
            pattern: ...,
            replace: ...
        }
    ],
    'InstSub' : [
        {...},
        {...}
    ]

}

example
"""


def load_isel_info_gen(file):
    """
    load generic.yml, from 'generic mir inst' to 'generic target inst'
    """
    with open(file, "r") as stream:
        try:
            isa_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    target_name = isa_data["Target"]["name"]
    isel_info = isa_data["InstSelInfo"]

    templates = isel_info["Templates"]
    isel_item_list = list()
    # isel_item: pattern, replace
    for template_name, template_info in templates.items():
        for generic_inst_name, target_inst_name in template_info["instances"].items():
            isel_item = dict()

            isel_item["pattern"] = copy.deepcopy(template_info["pattern"])
            isel_item["replace"] = copy.deepcopy(template_info["replace"])
            # print(generic_inst_name, target_inst_name)
            isel_item["pattern"]["name"] = generic_inst_name
            isel_item["replace"]["name"] = target_inst_name
            isel_item_list.append(isel_item)

    for isel_item in isel_info["Instances"]:
        isel_item_list.append(isel_item)
    return target_name, isel_info, isel_item_list


def load_isel_info(file):
    with open(file, "r") as stream:
        try:
            isa_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    target_name = isa_data["Target"]["name"]
    isel_info = isa_data["InstSelInfo"]
    # match_insts = isa_data["InstInfo"]["InstList"]

    templates = isel_info["Templates"]

    match_list = list()
    """
    pattern:
    replace:
    """
    for template_name, template_info in templates.items():
        for generic_inst_name, target_inst_name in template_info["instances"].items():
            pass

    return target_name, isel_info


"""
example:
target_name: RISCV
insts_dict: {
    'Add': {
        'name': 'Add', 
        'format': ['Add', 0, ', ', 1, ', ', 2], 
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
"""


def parse_isel_match(
    pattern: dict,
    idx: int,
    match_item_list: list,
    operand_map: dict,
    genmir_insts_dict: dict,
):
    """
    operand_map: $x -> global_id
    """

    capture_list = list()
    lookup_list = list()
    local_map = dict()  # just for this pattern, opearnd_name -> global_id
    # match_item = dict()

    pinst_name = pattern["name"]
    pinst_info = genmir_insts_dict[pinst_name]

    for op_idx, op_info in pinst_info["operands"].items():
        gidx = get_operand_id()
        local_map[op_info["name"]] = gidx
        capture_list.append(gidx)
    # match_item["type"] = "match_inst"
    match_item = {
        "type": "match_inst",
        "inst_name": pinst_name,
        "capture_list": capture_list,
        "lookup_list": lookup_list,
    }
    match_item_list.append(match_item)
    for k, v in pattern.items():
        if k == "name":
            continue
        if k in local_map:  # operand name
            if isinstance(v, str):
                assert v.startswith("$")
                operand_map[v] = local_map[k]  # $x -> gidx

    # 递归地处理


def replace_operad(code, operand_map):
    for k, v in operand_map.items():
        code = code.replace(k, str(v))
    return code


def parse_isel_select(
    replace: dict | str,
    select_item_list: list,
    operand_map: dict,
    target_insts_dict: dict,
    used_as_operand: bool = False,
):
    if isinstance(replace, str):
        gidx = get_operand_id()  # $x -> gidx
        select_item_list.append(
            {
                "type": "custom",
                "code": replace_operad(replace, operand_map),
                "idx": gidx,
            }
        )
        return gidx
    local_map = dict()
    rinst_name = replace["name"]
    for k, v in replace.items():
        if k == "name":
            continue
        if isinstance(v, str):
            # operand
            assert v.startswith("$")
            local_map[k] = parse_isel_select(
                v, select_item_list, operand_map, target_insts_dict, True
            )

    # select inst
    operands = list()
    for op_idx, op_info in target_insts_dict[rinst_name]["operands"].items():
        operands.append(local_map[op_info["name"]])

    idx = get_inst_id()  # select inst id
    select_item = {
        "type": "select_inst",
        "inst_name": replace["name"],
        "operands": operands,
        "idx": idx,
        "used_as_operand": used_as_operand,
    }
    select_item_list.append(select_item)
    return idx


def has_reg_def(inst_info):
    for opid, info in inst_info["operands"].items():
        if info["flag"] == "Def":
            return True
    return False


def parse_isel_item(isel_item: dict, mirgen_insts_dict: dict, target_insts_dict: dict):
    """
    isel_item {pattern: dict, replace: dict}
        pattern:
        replace:
    要为每个 isel_item 解析出 match_list, select_list
    """
    # 维护操作数信息
    operand_map = dict()
    match_item_list = list()
    select_item_list = list()
    root_id = get_inst_id()
    parse_isel_match(
        isel_item["pattern"], root_id, match_item_list, operand_map, mirgen_insts_dict
    )
    repalce_id = parse_isel_select(
        isel_item["replace"], select_item_list, operand_map, target_insts_dict, False
    )
    match_inst_name = match_item_list[0]["inst_name"]
    replace_inst_name = select_item_list[-1]["inst_name"]
    replace_operand = has_reg_def(mirgen_insts_dict[match_inst_name]) and has_reg_def(
        target_insts_dict[replace_inst_name]
    )
    # isel_item = {
    #     "match_id": root_id,
    #     "match_inst_name": match_item_list[0]["inst_name"],
    #     "match_list": match_item_list,
    #     "replace_id": repalce_id,
    #     "select_list": select_item_list,
    #     "replace_operand": replace_operad,
    # }
    isel_item["idx"] = get_id()
    isel_item["match_id"] = root_id
    isel_item["match_inst_name"] = match_item_list[0]["inst_name"]
    isel_item["match_list"] = match_item_list
    isel_item["replace_id"] = repalce_id
    isel_item["select_list"] = select_item_list
    isel_item["replace_operand"] = replace_operand
    reset_operand_idx()
    reset_inst_idx()
    # return isel_item


# mirgen_insts_dict, mirgen_insts_list
def get_mirgen_insts(gen_insts_dict, gen_insts_list):
    mirgen_insts_list = ["Inst" + name for name in gen_insts_list]
    mirgen_insts_dict = dict()
    for gen_inst_name, gen_inst_info in gen_insts_dict.items():
        mirgen_inst_name = "Inst" + gen_inst_name
        mirgen_inst_info = copy.deepcopy(gen_inst_info)
        mirgen_inst_info["name"] = mirgen_inst_name
        mirgen_insts_dict[mirgen_inst_name] = mirgen_inst_info
    return mirgen_insts_dict, mirgen_insts_list


if __name__ == "__main__":
    gen_data_yml = sys.argv[1]
    isa_data_yml = sys.argv[2]
    output_dir = sys.argv[3]

    target_name, insts_dict, insts_list = load_inst_info(isa_data_yml)
    params = {
        "target_name": target_name,
        "insts_dict": insts_dict,
        "insts_list": insts_list,
    }
    gen_file_jinja2("InstInfoDecl.hpp.jinja2", output_dir, params)
    gen_file_jinja2("InstInfoImpl.hpp.jinja2", output_dir, params)

    # gen_data_yml = "generic.yml"
    _, gen_insts_dict, gen_insts_list = load_inst_info(gen_data_yml)
    # mirgen_insts_list = ["Inst" + name for name in gen_insts_list]
    # print(mirgen_insts_list)
    mirgen_insts_dict, mirgen_insts_list = get_mirgen_insts(
        gen_insts_dict, gen_insts_list
    )

    # params = {
    #     "match_insts_list": mirgen_insts_list,
    #     "match_insts_dict": mirgen_insts_dict,
    # }
    # print(mirgen_insts_list)
    # print(mirgen_insts_dict)
    # print(gen_insts_list)
    # print(gen_insts_dict)

    target_name, isel_info, isel_item_list = load_isel_info_gen(isa_data_yml)

    isel_dict = dict()

    for isel_item in isel_item_list:
        parse_isel_item(isel_item, mirgen_insts_dict, insts_dict)
        if isel_item["match_inst_name"] not in isel_dict:
            isel_dict[isel_item["match_inst_name"]] = []
        isel_dict[isel_item["match_inst_name"]].append(isel_item)

        # isel_item_list.append(isel_item)
    params = {
        "target_name": target_name,
        "isel_dict": isel_dict,
        "match_insts_list": mirgen_insts_list,
        "match_insts_dict": mirgen_insts_dict,
    }
    # print(isel_dict)
    # for
    gen_file_jinja2("ISelInfoDecl.hpp.jinja2", output_dir, params)
    gen_file_jinja2("ISelInfoImpl.hpp.jinja2", output_dir, params)
    # params = {"target_name": target_name, "isel_info": isel_info}
    # gen_file_jinja2("ISelInfoImpl.hpp.jinja2", output_dir, params)


"""
解析 pattern, replace, 得到 match_list, select_list
input:
isel_info, mirgen_insts_dict
output:
isel_dict: dict(inst_name: str -> isel_item_list: list(isel_item))
    isel_item:
        # match
        match_id: int
        match_list: list(match_item)
            match_item:
                type: match_inst
                root: id
                inst_name: InstXXX
                capture_list: [0, 1]
                lookup_list: [0] # lookup def

                type: predicate
                code:
                new_ops:
        # select 
        select_list: list(select_item)
            select_item:


pattern:
    match_id

    match_list: []
        type: match_inst
        root: id
        inst_name: InstXXX
        capture_list: [0, 1]
        lookup_list: [0] # lookup def

        type: predicate
        code:
        new_ops:

    select_list: []
        type: select_inst
        inst_name: InstXXX
        inst_ref_name: InstXXX
        operands:
        idx:
        used_as_operand: true/false

        type: custom
        code: 
        idx:

"""
