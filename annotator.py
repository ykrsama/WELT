import os
import re
import glob
import requests
import difflib
import argparse
from pathlib import Path
from typing import List, Tuple

class CodeAnnotator:
    def __init__(self, repo_path: str, doxyfile: str = "Doxyfile", api_endpoint: str = "http://localhost:7860/api/chat/completions"):
        self.repo_path = Path(repo_path)
        self.doxyfile_path = self.repo_path / doxyfile
        self.api_endpoint = api_endpoint
        self.file_backups = {}

    def parse_doxyfile(self) -> Tuple[List[str], List[str]]:
        """解析Doxyfile配置"""
        config = {}
        with open(self.doxyfile_path, 'r') as f:
            for line in f:
                line = re.sub(r'#.*$', '', line).strip()  # 移除注释
                if line and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        
        input_dirs = [self.repo_path / d for d in config.get('INPUT', '').split()]
        file_patterns = config.get('FILE_PATTERNS', '*.c *.cpp *.h *.hpp').split()
        return input_dirs, file_patterns

    def get_code_files(self) -> List[Path]:
        """获取需要注释的代码文件"""
        input_dirs, file_patterns = self.parse_doxyfile()
        files = []
        for directory in input_dirs:
            for pattern in file_patterns:
                files.extend(Path(directory).rglob(pattern))
        return sorted(set(files))

    def generate_annotation(self, code_content: str) -> str:
        """调用OpenWebUI API生成注释"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY"  # 根据实际配置修改
        }

        #message_content=
        
        payload = {
            "model": "code-annotator",
            "messages": [{
                "role": "system",
                "content": """为了给以上代码添加详细注释，请先决定是否用Knowledge Search检索源码，然后基于函数实现代码，在以上代码中的类、函数的声明处添加注释。指引：

- 如果缺少函数实现的代码，请先根据函数名，用Knowledge Search检索源码。可以多次搜索直到找到全部函数的实现代码。

- 使用以下markdown代码块进行搜索和替换，用于添加注释:

```
<<<<<<< ORIGINAL
original content
=======
updated content
>>>>>>> UPDATED
```

- 使用Doxygen风格的英文注释。

- 如果以上代码没有要添加注释的地方，请回复"Nothing to do."
"""
            }, {
                "role": "user",
                "content": code_content
            }],
            "temperature": 0.2,
            "max_tokens": 2000
        }

        try:
            response = requests.post(self.api_endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return ""

    def apply_diff(self, original: List[str], diff: str) -> List[str]:
        """应用diff修改"""
        patched = []
        lines = iter(original)
        
        for line in diff.split('\n')[4:]:  # 跳过diff头
            if line.startswith('+'):
                patched.append(line[1:])
            elif line.startswith('-'):
                next(lines)  # 跳过原行
            else:
                patched.append(next(lines)[1:] if line.startswith(' ') else next(lines))
        return patched

    def process_file(self, file_path: Path):
        """处理单个文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            original = f.readlines()
        
        # 生成注释diff
        diff = self.generate_annotation(''.join(original))
        if not diff.startswith('@@'):
            print(f"无效的diff格式：{file_path}")
            return

        # 备份原始文件
        backup_path = file_path.with_suffix(file_path.suffix + '.bak')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.writelines(original)
        self.file_backups[str(file_path)] = str(backup_path)

        # 应用修改
        patched = self.apply_diff(original, diff)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(patched)

    def run(self):
        """主运行逻辑"""
        files = self.get_code_files()
        print(f"找到 {len(files)} 个需要注释的文件")
        
        for idx, file_path in enumerate(files, 1):
            print(f"处理文件 ({idx}/{len(files)})：{file_path}")
            try:
                self.process_file(file_path)
            except Exception as e:
                print(f"处理失败：{file_path} - {str(e)}")
        
        print(f"完成！备份文件保存在：")
        for original, backup in self.file_backups.items():
            print(f"- {original} -> {backup}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="代码自动注释工具")
    parser.add_argument("repo_path", help="代码仓库路径")
    parser.add_argument("--doxyfile", default="Doxyfile", help="Doxyfile路径")
    args = parser.parse_args()

    annotator = CodeAnnotator(args.repo_path, args.doxyfile)
    annotator.run()
