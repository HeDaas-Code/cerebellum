"""
Cerebellum 智能代理示例

演示如何使用 Cerebellum 处理文档任务
"""

from cerebellum import Cerebellum, CerebellumConfig
from pathlib import Path

config = CerebellumConfig(
    skills_dir=Path("./cerebellum/skills"),
    database_path=Path("./my_cache.db"),
    debug=True
)

with open("doc.txt", "rb") as f:
    data = f.read()

with Cerebellum(config=config) as agent:
    result = agent.run(
        task="总结这份文档,并生成一个图表",
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
