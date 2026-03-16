"""
Cerebellum 智能代理示例

演示如何使用 Cerebellum 处理文档任务
"""

from cerebellum import Cerebellum, CerebellumConfig, SandboxConfig
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

config = CerebellumConfig(
    skills_dir=Path("./cerebellum/skills"),
    database_path=Path("./my_cache.db"),
    debug=True,
)

sandbox_config = SandboxConfig(
    timeout_seconds=300,
    max_retries=3,
    auto_cleanup=True,
    workdir="/home/daytona/workspace",
    pre_install=[
        {"name": "fonts-wqy-zenhei", "type": "apt"},
        {"name": "fonts-wqy-microhei", "type": "apt"},
        {"name": "fc-cache", "type": "apt"},
    ]
)
config.sandbox = sandbox_config

with open("doc.txt", "rb") as f:
    data = f.read()

with Cerebellum(config=config) as agent:
    result = agent.run(
        task="总结这份文档的数据和内容,根据当前数据并生成一个带图表的社会统计报告,中文",
        files=[(data, "doc.txt")]
    )
    
    print(f"\n执行结果: {'成功' if result['success'] else '失败'}")
    print(f"消息: {result['message']}")
    print(f"生成文件数: {len(result['files'])}")
    
    for file in result["files"]:
        print(f"\n文件名: {file['name']}")
        print(f"类型: {file['type']}")
        
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        content = file['content']
        if isinstance(content, bytes):
            with open(output_dir / file['name'], "wb") as f:
                f.write(content)
        else:
            with open(output_dir / file['name'], "w", encoding="utf-8") as f:
                f.write(content)
        
        print(f"已保存到: output/{file['name']}")
