---
name: file-delivery
description: 文件交付技能。当智能体在沙盒中生成文件后，必须将文件从沙盒下载到本地 ./output 文件夹。这是核心职责！
model: glm-5
---

# 文件交付技能

## 核心职责
当你完成任何涉及生成文件的任務後，你必須將文件從沙盒交付到用戶的本地機器。

## 文件交付流程

### 步驟 1：確認沙盒文件位置
- 沙盒工作目錄：`/home/daytona`
- 例如：`/home/daytona/fibonacci.png`、`/home/daytona/result.pdf`

### 步驟 2：讀取沙盒中的文件
使用 `read_file` 工具讀取沙盒文件內容：
- 工具：`read_file`
- 參數：沙盒中的完整路徑，如 `/home/daytona/fibonacci.png`

### 步驟 3：創建本地輸出目錄
在本地創建 `./output` 文件夾（如果不存在）：
```bash
mkdir -p output
```

### 步驟 4：保存文件到本地
使用 Python 將讀取的文件內容保存到本地：
```python
# 讀取沙盒文件
content = <read_file結果>

# 保存到本地 ./output 文件夾
with open("output/文件名.png", "wb") as f:
    f.write(content)
```

### 步驟 5：告知用戶
告訴用戶文件已保存到本地 `./output/` 文件夾。

## 重要規則

1. **不要只給路徑**：必須實際將文件內容保存到本地
2. **output 文件夾**：本地輸出目錄是 `./output`，不是其他位置
3. **完整流程**：生成文件 → 讀取文件 → 保存到 output → 告知用戶

## 錯誤示例（不要這樣做）
- ❌ "文件已生成在 /home/daytona/fibonacci.png，請自行下載"
- ❌ "文件路徑是 ./fibonacci.png"

## 正確示例（應該這樣做）
- ✅ "文件已保存到 ./output/fibonacci.png"
- ✅ "我已將 PDF 文件保存到本地 ./output/report.pdf"
