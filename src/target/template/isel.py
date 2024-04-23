import os
import sys
import yaml
import jinja2
import copy
from jinja2 import Environment, FileSystemLoader

# import typing
from typing import List, Dict, Tuple


class IselIndex:
    def __init__(self):
        self.isel_item_idx = 0
        self.inst_idx = 0
        self.operand_idx = 0

    def get_isel_item_idx(self):
        self.isel_item_idx += 1
        return self.isel_item_idx

    def get_inst_idx(self):
        self.inst_idx += 1
        return self.inst_idx

    def get_operand_idx(self):
        self.operand_idx += 1
        return self.operand_idx

    def reset_isel_item_idx(self):
        self.isel_item_idx = 0

    def reset_inst_idx(self):
        self.inst_idx = 0

    def reset_operand_idx(self):
        self.operand_idx = 0


iselindex = IselIndex()


def load_isel_info(file: str) -> Tuple[str, List]:
    """Load generic.yml, from 'generic mir inst' to 'generic target inst'
        对模板进行实例化
    args:
        file: str, path to generic.yml
    return:
        isel_item_list: List, list of isel items

    """

    with open(file, "r") as stream:
        try:
            isa_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    isel_info = isa_data["InstSelInfo"]

    templates = isel_info["Templates"]
    isel_item_list = list()
    # isel_item: pattern, replace
    for template_name, template_info in templates.items():
        for generic_inst_name, target_inst_name in template_info["instances"].items():
            isel_item = dict()

            isel_item["pattern"] = copy.deepcopy(template_info["pattern"])
            isel_item["replace"] = copy.deepcopy(template_info["replace"])

            isel_item["pattern"]["name"] = generic_inst_name
            isel_item["replace"]["name"] = target_inst_name
            isel_item_list.append(isel_item)

    for isel_item in isel_info["Instances"]:
        isel_item_list.append(isel_item)
    return isel_item_list


def handle_new_ops(code: str, map, new_ops):
    while True:
        pos = code.find("[$")
        if pos == -1:
            return code
        end = code.find("]", pos)
        name = code[pos + 1 : end]
        new_id = iselindex.get_operand_idx()
        code = code.replace(code[pos : end + 1], "op" + str(new_id))
        map[name] = new_id
        new_ops.append(new_id)


def parse_isel_match(
    pattern: dict,
    pattern_id: int,
    match_item_list: list,  # append match_item to
    operand_map: dict,  # $xx -> id in select_item
    genmir_insts_dict: dict,  # MIR Generic Inst Info Dict: InstXXX -> info
):
    """Append one match_item to match_item_list
    operand_map: $x -> global_id
    In one inst, opearnd_name -> $xxx (global_name),
    local_map: operand_name -> $xxx -> id in select_item
    """

    capture_list = list()  # 捕获列表, 用于匹配 pattern inst 后捕获其操作数
    lookup_list = list()  # 递归匹配前，需要 lookup 的操作数列表
    local_map = dict()  # just for this pattern, opearnd_name -> global_id
    # match_item = dict()

    pinst_name = pattern["name"]
    pinst_info = genmir_insts_dict[pinst_name]

    for op_idx, op_info in pinst_info["operands"].items():
        gidx = iselindex.get_operand_idx()
        local_map[op_info["name"]] = gidx
        capture_list.append(gidx)

    match_item = {
        "type": "match_inst",
        "pattern_id": pattern_id,
        "inst_name": pinst_name,
        "capture_list": capture_list,
        "lookup_list": lookup_list,
    }

    match_item_list.append(match_item)

    for k, v in pattern.items():
        if k == "name":
            continue
        elif k == "predicate":
            new_ops = []
            v = handle_new_ops(v, operand_map, new_ops)
            match_item_list.append(
                {
                    "type": "predicate",
                    "code": replace_operad(v, operand_map),
                    "new_ops": new_ops,
                }
            )
        elif k in local_map:  # operand name
            if isinstance(v, str):
                assert v.startswith("$")
                operand_map[v] = local_map[k]  # $x -> gidx
            elif isinstance(v, dict):
                # v is a sub-pattern
                # append to lookup_list, same to the before!
                lookup_list.append(local_map[k])
                # 递归地处理
                parse_isel_match(
                    v,
                    iselindex.get_inst_idx(),
                    match_item_list,
                    operand_map,
                    genmir_insts_dict,
                )
            else:
                raise ValueError("Invalid operand type")


def replace_operad(code, operand_map):
    for k, v in operand_map.items():
        code = code.replace(k, "op" + str(v))
    return code


def parse_isel_select(
    replace: dict | str,
    select_item_list: list,
    operand_map: dict,
    target_insts_dict: dict,
    used_as_operand: bool = False,
):
    local_map = dict()
    
    if "$" in replace["name"]:
        rinst_name = replace_operad(replace["name"], operand_map)
    else:
        rinst_name = replace["name"]

    inst_ref = rinst_name
    for k, v in replace.items():
        if k == "name": # 
            continue
        elif k == "ref":
            inst_ref = v
        elif isinstance(v, str):
            # operand
            local_map[k] = iselindex.get_operand_idx()  # $x -> gidx
            select_item_list.append(
                {
                    "type": "custom",
                    "code": replace_operad(v, operand_map),
                    "idx": local_map[k],
                }
            )
        elif isinstance(v, dict):
            # v is a sub-pattern
            local_map[k] = parse_isel_select(
                v,
                select_item_list,
                operand_map,
                target_insts_dict,
                used_as_operand=True,
            )

    # select inst
    operands = list()
    inst_info = target_insts_dict[inst_ref]
    for op_idx, op_info in inst_info["operands"].items():
        operands.append(local_map[op_info["name"]])

    inst_idx = iselindex.get_inst_idx()  # select inst id
    operand_idx = iselindex.get_operand_idx()  # select operand id
    select_item_list.append(
        {
            "type": "select_inst",
            "inst_name": rinst_name,
            "inst_ref": inst_ref,
            "operands": operands,
            "idx": inst_idx,
            "operand_idx": operand_idx,
            "used_as_operand": used_as_operand,
        }
    )
    return operand_idx


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
    pattern_id = iselindex.get_inst_idx()  # 根节点 id
    parse_isel_match(
        isel_item["pattern"],
        pattern_id,
        match_item_list,
        operand_map,
        mirgen_insts_dict,
    )
    parse_isel_select(
        isel_item["replace"], select_item_list, operand_map, target_insts_dict, False
    )
    repalce_id = select_item_list[-1]["idx"]
    match_inst_name = match_item_list[0]["inst_name"]

    replace_inst_name = select_item_list[-1]["inst_ref"]
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
    isel_item["idx"] = iselindex.get_isel_item_idx()
    isel_item["match_id"] = pattern_id
    isel_item["match_inst_name"] = match_item_list[0]["inst_name"]
    isel_item["match_list"] = match_item_list
    isel_item["replace_id"] = repalce_id
    isel_item["select_list"] = select_item_list
    isel_item["replace_operand"] = replace_operand
    iselindex.reset_inst_idx()
    iselindex.reset_operand_idx()

    # return isel_item
