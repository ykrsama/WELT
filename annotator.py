"""
Ignore me, just a WIP
"""

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

        prompt_zh = """如果以上代码中的函数声明缺少对应的实现源码，请调用Knowledge Search检索源码。然后基于函数实现的源码，在以上代码中的类、函数的声明处，添加详细注释。如果以上代>码没有要添加注释的地方，请回复"Nothing to do."

检索源码指引：

- 根据函数名，调用Knowledge Search检索源码。

- 可以重复搜索直到找到全部函数的实现代码。

添加注释指引：

- 使用以下markdown代码块进行搜索和替换，用于添加注释:

```
<<<<<<< ORIGINAL
original content
=======
updated content
>>>>>>> UPDATED
```

- 使用Doxygen风格的注释。

- 注意更新时源码必须保持一致，不能修改源码，只能添加注释。
"""

        prompt_en = """If the function declarations in the above code lack corresponding implementation source code, please call Knowledge Search to retrieve the source code. Then, based on the implementation of the functions, add detailed comments at the declarations of classes and functions in the above code. If there is no need to add comments in the above code, please reply with "Nothing to do."

Source Code Retrieval Guidelines:
- Query with the function name for Knowledge Search.
- You can perform repeated searches until you find all the implementation code for all functions.

Comment Addition Guidelines:
- Use the following markdown code block for search and replacement to add comments:
```
<<<<<<< ORIGINAL
original content
=======
updated content
>>>>>>> UPDATED
```
- Use Doxygen-style comments.
- Ensure that the source code remains consistent during updates; do not modify the source code, only add comments.
"""
        message_content = f"```\n{code_content}\n```\n\n{prompt_en}"
        
        payload = {
            "model": "code-annotator",
            "messages": [{
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
