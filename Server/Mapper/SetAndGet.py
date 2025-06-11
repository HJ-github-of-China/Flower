import os

import yaml

from Server.Mapper.MySQLConfig import OpsForMySQL

# 获取当前脚本的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建绝对路径
config_path = os.path.join(script_dir, 'config.yml')

exporter = OpsForMySQL()


# 存储并获取配置的脚本
def getconfig():
    #TODO  能不能动态配置一下

    # 导出配置ID为1的记录
    config_id = 1  # 替换为您要导出的配置ID
    output_file = "Mapper/config.yml"  # 输出文件名

    if exporter.export_to_yml(config_id, output_file):
        print("导出成功！")
    else:
        print("导出失败")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


config = getconfig()
FED_CONFIG = config['federated']

