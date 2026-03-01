from cerebellum import Cerebellum, CerebellumConfig
from pathlib import Path

# 配置技能文件夹
config = CerebellumConfig(
    skills_dir=Path("./skills"),  # 自定义技能目录
    debug=True
)

with open("doc.txt", "rb") as f:
    data = f.read()

with Cerebellum() as agent:
    result = agent.run(
        task="总结这份文档,并生成一个图表",
        files=[(data, "doc.txt")]  # 可选参数
    )
    
    # 访问返回的文件
    for file in result["files"]:
        print(f"文件名: {file.name}")
        print(f"类型: {file.type}")
        # 保存到本地
        with open(f"output/{file.name}", "wb") as f:
            f.write(file.get_bytes())