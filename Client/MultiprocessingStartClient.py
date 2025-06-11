"""
1.多进程：通过 multiprocessing.Process 创建多个独立进程，每个进程运行一个客户端实例（避免客户端之间互相干扰）。
2.参数传递：每个客户端进程会传入 client_id（0、1、2），最终通过 client_fn 函数加载对应分区的数据（partition_id 与 client_id 一一对应）。
3。与服务器通信：所有客户端会连接到 SERVER_ADDRESS 指定的服务器（需与服务器代码中的地址一致），参与联邦学习的参数同步。
"""
import multiprocessing

from Utils import startClientUtil

if __name__ == "__main__":
    # 配置参数
    NUM_CLIENTS = 3
    SERVER_ADDRESS = "127.0.0.1:8088"
    # 启动多个客户端进程
    # TODO 创建相关的启动函数
    processes = []
    for client_id in range(NUM_CLIENTS):
        process = multiprocessing.Process(target=startClientUtil.start_client_process, args=(client_id, SERVER_ADDRESS))
        processes.append(process)
        process.start()

    # 等待所有客户端完成
    for process in processes:
        process.join()