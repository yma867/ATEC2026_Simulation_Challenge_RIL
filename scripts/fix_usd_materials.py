#!/usr/bin/env python3
"""修复 USD 文件中的材质绑定路径问题"""

import os
from pxr import Usd, UsdGeom, Sdf

def fix_material_bindings_in_layer(layer):
    """修复图层中的材质绑定路径"""
    modified = False
    
    # 遍历所有 PrimSpec
    for prim_spec in layer.GetPrimAtPath("/").nameChildren:
        # 递归处理子 Prim
        modified |= _fix_prim_material_bindings(prim_spec, layer)
    
    return modified

def _fix_prim_material_bindings(prim_spec, layer):
    """递归修复 Prim 的材质绑定"""
    modified = False
    
    # 检查是否有材质绑定关系
    if "material:binding" in prim_spec.relationships:
        rel_spec = prim_spec.relationships["material:binding"]
        targets = list(rel_spec.targetPathList.explicitItems)
        
        for target_path in targets:
            # 检查是否是超出范围的绝对路径
            if str(target_path).startswith("/piper/"):
                print(f"发现无效材质绑定: {prim_spec.path} -> {target_path}")
                # 移除这个目标
                rel_spec.targetPathList.Remove(target_path)
                modified = True
    
    # 递归处理子 Prim
    for child_spec in prim_spec.nameChildren:
        modified |= _fix_prim_material_bindings(child_spec, layer)
    
    return modified

def main():
    usd_path = "/home/ril/myq/ATEC2026_Simulation_Challenge_RIL/atec_robot_model/robot/b2/b2_piper.usda"
    
    if not os.path.exists(usd_path):
        print(f"文件不存在: {usd_path}")
        return
    
    # 打开 USD 文件
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"无法打开文件: {usd_path}")
        return
    
    print(f"正在处理: {usd_path}")
    
    # 获取主图层
    root_layer = stage.GetRootLayer()
    
    # 修复材质绑定
    if fix_material_bindings_in_layer(root_layer):
        # 保存修改
        stage.Save()
        print("修复完成，已保存修改")
    else:
        print("未发现需要修复的材质绑定")

if __name__ == "__main__":
    main()