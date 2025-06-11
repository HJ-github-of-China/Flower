"""
1.暴露接口，用户可以通过前端调用
2.获取训练的参数，由用户配置，call Mapper × 应该是server负责
2.call startServer ---> call startClient
"""
import multiprocessing

from flask import Flask, jsonify
from flask_cors import CORS

from Server import startServer

app = Flask(__name__)
# TODO 什么是跨域访问
CORS(app)


@app.route("/run", methods=["GET"])
def run():
    # 返回结果
    # TODO 前端联调注意勒
    result = {
        "status": "服务端启动成功"
        }
    try:
        # 调用 startServer
        p = multiprocessing.Process(target=startServer.start_server)
        p.start()
    except Exception as e:
        result["status"] = "服务端启动失败"
        result["error"] = str(e)
    return jsonify(result)


# 说明只能在当前文件下运行
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)