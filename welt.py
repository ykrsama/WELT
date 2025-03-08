"""
title: Welt
author: Xuliang
description: OpenWebUI pipe function for Welt: Workflow-adaptivE LLM with Target-oriented system
version: 0.0.8
licence: MIT
"""

import logging
import io, sys
import json
import httpx
import re
import requests
import time
from typing import AsyncGenerator, Callable, Awaitable, Optional, List, Tuple
from pydantic import BaseModel, Field
import asyncio
from jinja2 import Template
from datetime import datetime
from open_webui.utils.misc import (
    add_or_update_user_message,
)
from open_webui.models.messages import (
    Messages,
    MessageModel,
    MessageResponse,
    MessageForm,
)
from open_webui.retrieval.vector.connector import VECTOR_DB_CLIENT
from langchain_core.documents import Document
from open_webui.retrieval.utils import (
    generate_embeddings,
)
from open_webui.models.knowledge import (
    Knowledges,
    KnowledgeForm,
    KnowledgeResponse,
    KnowledgeUserResponse,
    KnowledgeUserModel,
)
from hepai import HRModel

log = logging.getLogger(__name__)
log.setLevel("DEBUG")


class ResultObject:
    def __init__(self, id, distance, document, metadata):
        self.id = id
        self.distance = distance
        self.document = document
        self.metadata = metadata


class Pipe:
    class Valves(BaseModel):
        DEEPSEEK_API_BASE_URL: str = Field(
            default="https://api.deepseek.com/v1",
            description="DeepSeek API的基础请求地址",
        )
        DEEPSEEK_API_KEY: str = Field(
            default="", description="用于身份验证的DeepSeek API密钥，可从控制台获取"
        )
        DEEPSEEK_API_MODEL: str = Field(
            default="deepseek-reasoner",
            description="API请求的模型名称，默认为 deepseek-reasoner ",
        )
        USE_CODE_INTERPRETER: bool = Field(default=True)
        USE_WEB_SEARCH: bool = Field(default=True)
        USE_KNOWLEDGE_SEARCH: bool = Field(default=True)
        EMBEDDING_BATCH_SIZE: int = Field(
            default=2000,
            description="Batch size for knowledge search",
        )
        GOOGLE_PSE_API_KEY: str = Field(default="")
        GOOGLE_PSE_ENGINE_ID: str = Field(default="")
        MAX_LOOP: int = Field(default=20, description="Prevent dead loop")

    def __init__(self):
        # Configs
        self.valves = self.Valves()
        self.data_prefix = "data:"
        self.max_loop = self.valves.MAX_LOOP  # Save moneya
        self.client = httpx.AsyncClient(
                http2=True,
                proxy="http://127.0.0.1:7890",
                timeout=None
        )
        self.CODE_INTERPRETER_PROMPT: str = """Code Interpreter

You have access to a user's {{OP_SYSTEM}} code workspace, use `<code_interpreter>` XML tag to write codes to do analysis, calculations, or problem-solving. Here's how it works:

<code_interpreter type="exec" lang="bash" filename="">
code here (DO NOT consider xml escaping, e.g. use `<`, DO NOT use `&lt;`)
</code_interpreter>

#### Tool Attributes

- `type`: Specifies the action to perform.
   - `exec`: Execute the code immediately.
      - Supported languages: `python`, `bash`, `boss`
   - `write`: Write and save the code to a file.
      - Supports any programming language.

- `filename`: The file path where the code will be written.  
   - Must be **relative to the user's workspace base directory**, do not use paths relative to subdirectory.

#### Usage Instructions

- The Python code you write can incorporate a wide array of libraries, handle data manipulation or visualization, perform API calls for web-related tasks, or tackle virtually any computational challenge. Use this flexibility to **think outside the box, craft elegant solutions, and harness Python's full potential**.
- Output XML node simply like `<code_interpreter ...>...</code_interpreter>`, DO NOT put XML node inside the markdown code block (```xml). 
- Coding style instruction:
  - **Always aim to give meaningful outputs** (e.g., results, tables, summaries, or visuals) to better interpret and verify the findings. Avoid relying on implicit outputs; prioritize explicit and clear print statements so the results are effectively communicated to the user.
   - Run in batch mode. Save figures to file.
   - Prefer object-oriented programming
   - Prefer arguments with default value than hard coded
   - For potentially time-consuming code, e.g., loading file with unknown size, use argument to control the running scale, and defaulty run on small scale test.

#### Examples Begin

User: plot something
Assistant: ...
**Calling Code Intepreter:**
<code_interpreter type="exec" lang="python" filename="plot.py">
# plotting code here
</code_interpreter>

---

User: Create and test a simple cmake project named HelloWorld
Assistant: ...
**Calling Code Intepreter:**
<code_interpreter type="save" lang="cmake" filename="HelloWorld/CMakeList.txt">
...
</code_interpreter>
<code_interpreter type="save" lang="c++" filename="HelloWorld/src/main.cpp">
...
</code_interpreter>
<code_interpreter type="exec" lang="bash" filename="HelloWorld/build_and_test.sh">
# assume run in parent directory of filename
mkdir -p build
cd build
cmake ..
make
./MyExecutable
</code_interpreter>

#### Examples End

"""
        self.WEB_SEARCH_PROMPT: str = """Web Search

- You have access to internet, use `<web_search>` XML tag to search the web for new information and references. Example:

**Calling Web Search Tool:**

<web_search engine="google">first query here</web_search>
<web_search engine="google">second query here</web_search>

#### Tool Attributes

- `engine`: available options:
  - `google`: Search on google.
  - `arxiv`: Always use english keywords for arxiv.

####  Usage Instructions

- Enclose only one query in one pair of `<web_search engine="...">` `</web_search>` XML tags. You can use multiple lines of `<web_search>` XML tags for each query, to do parallel search.
- Err on the side of suggesting search queries if there is **any chance** they might provide useful or updated information.
- Always prioritize providing actionable and broad query that maximize informational coverage.
- In each web_search XML tag, be concise and focused on composing high-quality search query, **avoiding unnecessary elaboration, commentary, or assumptions**.
- No need to bother API keys because user can handle by themselves in this tool.
- **The date today is: {{CURRENT_DATE}}**. So you can search for web to get information up do date {{CURRENT_DATE}}.

"""
        self.KNOWLEDGE_SEARCH_PROMPT: str = """Knowledge Search

- You have access to user's database, use `<knowledge_search>` XML tag to search user's internal and personal documents. Example:

**Calling Knowledge Search Tool:**

<knowledge_search collection="DarkSHINE_Simulation_Software">one query</knowledge_search>

#### Tool Attributes
  - `collection`: Available options:
     - `DarkSHINE_Simulation_Software`: Source code of simulation program based on Geant4 and ROOT, characterized by detector of DarkSHINE experiment. **Must use English to query this collection**.
     - `OpenWebUI_Backend`: Source code of backend of OpenWebUI, an extensible, self-hosted AI interface.

#### Usage Insructions

- In each `<knowledge_search>` XML tag, be concise and focused on composing high-quality search query, **avoiding unnecessary elaboration, commentary, or assumptions**.
- Enclose only one query in one pair of `<knowledge_search collection="...">` `</knowledge_search>` XML tags.
"""
        self.GUIDE_PROMPT: str = """## DarkSHINE Physics Analysis Guide:

### Introduction

DarkSHINE Experiment is a fixed-target experiment to search for dark photons (A') produced in 8 GeV electron-on-target (EOT) collisions. The experiment is designed to detect the invisible decay of dark photons, which escape the detector with missing energy and missing momentum. The DarkSHINE detector consists of Tagging Tracker, Target, Recoil Tracker, Electromagnetic Calorimeter (ECAL), Hadronic Calorimeter (HCAL).

### Simulation and Reconstruction

1. Configure the beam parameters and detector geometries for the simulation setup
2. Signal simulation and reconstruction
   1. Decide the free parameters to scan according to the signal model
   2. Simulate signal events
      1. Prepare config file
      2. Run simulation program
         - DSimu: DarkSHINE MC event generator
         - boss.exe: BESIII MC event generator
   3. Reconstruct the signal events.
      1. Prepare config file
      2. Run reconstruction program
         - DAna: DarkSHINE reconstruction program
         - boss.exe: BESIII reconstruction program
3. Background simulation and reconstruction
   1. Configure the physics process for background events
   2. Simulate background events
   3. Reconstruct background events

#### Examples Begin

User: For DarkSHINE (electron-on-target), generate and reconstruct 100 dark photon invisible decay signal events per signal mass
Assistant: Scan dark photon mass mAp, and output simulation files to eot/signal/invisible/mAp_*/dp_simu/0.root
<code_interpreter type="exec" lang="bash" filename="signal_invisible_decay_eot.sh">
#!/bin/bash

# Set the base directory for the simulation
base_dir=$PWD/eot/signal/invisible
mkdir -p $base_dir

# Set the original config file directory
dsimu_script_dir="/opt/darkshine-simulation/source/DP_simu/scripts"
default_yaml="$dsimu_script_dir/default.yaml"
magnet_file="$dsimu_script_dir/magnet_0.75_20240521.root"
signal_lut="$dsimu_script_dir/dp_E4_W_LUT_20240327.root"

# Prepare reconstruction config file
DAna -x > $base_dir/config.txt

# Scan over dark photon mass mAp
for mAp in 10 20 50 100 200 500 1000; do
    echo "Processing mAp = ${mAp} MeV"
    # Create and goto directory for each mA
    cur_dir=$base_dir/mAp_${mAp}
    mkdir -p $cur_dir/dp_simu
    mkdir -p $cur_dir/dp_ana
    cd $cur_dir

    # Prepare config file for simulation
    echo "-- Preparing simulation config"
    sed -e "s:signal_production\::signal_production\: true  \#:" \
        -e "s:signal_mass\::signal_mass\: \[${mAp},\"MeV\"\]  \#:" \
        -e "s:signal_lookup_table\::signal_lookup_table\: \"${signal_lut}\"  \#:" \
        -e "s:  mag_field_input\::  mag_field_input\: \"${magnet_file}\"  \#:" \
        $default_yaml > default.yaml

    # Run Simulation
    echo "-- Running simulation"
    DSimu -y default.yaml -b 100 -f dp_simu/0.root > simu.out 2> simu.err

    # Prepare reconstruction config file
    echo "-- Preparing reconstruction config"
    sed -e "s:InputFile:InputFile = dp_simu/0.root  \#:" \
        -e "s:InputGeoFile:InputGeoFile = dp_simu/0.root  \#:" \
        -e "s:OutputFile:OutputFile = dp_ana/0.root  \#:" \
        $base_dir/config.txt > $cur_dir/config.txt

    # Reconstruction
    echo "-- Running reconstruction"
    DAna -c config.txt > ana.out 2> ana.err
done

echo "All done!"

</code_interpreter>

---

User: For DarkSHINE, simulate and reconstruct inclusive background events
Assistant: <code_interpreter type="exec" lang="bash" filename="background_inclusive_eot.sh">
#!/bin/bash

# Set the base directory for the simulation
base_dir=$PWD/eot/background/inclusive
mkdir -p $base_dir/dp_simu
mkdir -p $base_dir/dp_ana
cd $base_dir

# Set the original config file directory
dsimu_script_dir="/opt/darkshine-simulation/source/DP_simu/scripts"
default_yaml="$dsimu_script_dir/default.yaml"
magnet_file="$dsimu_script_dir/magnet_0.75_20240521.root"
DAna -x > config_temp.txt

# Prepare config file for simulation
echo "-- Preparing simulation config"
sed "s:  mag_field_input\::  mag_field_input\: \"${magnet_file}\"  \#:" $default_yaml > default.yaml

# Run Simulation
echo "-- Running simulation"
DSimu -y default.yaml -b 100 -f dp_simu/0.root > simu.out 2> simu.err

# Prepare reconstruction config file
echo "-- Preparing reconstruction config"
sed -e "s:InputFile:InputFile = dp_simu/0.root  \#:" \
    -e "s:InputGeoFile:InputGeoFile = dp_simu/0.root  \#:" \
    -e "s:OutputFile:OutputFile = dp_ana/0.root  \#:" \
    config_temp.txt > config.txt

# Reconstruction
echo "-- Running reconstruction"
DAna -c config.txt > ana.out 2> ana.err

echo "All done!"

</code_interpreter>

#### Examples End

### Validation

Plot histograms to compare the signal and background kinematic distributions

#### Variables

Tree Name: `dp`

| Branch Name | Type | Description |
| --- | --- | --- |
| TagTrk2_pp | Double_t[] | Reconstructed Tagging Tracker momentum [MeV]. TagTrk2_pp[0] - Leading momentum track |
| TagTrk2_track_No | Int_t | Number of reconstructed Tagging Tracker tracks |
| TagTrk2_track_chi2 | Double_t[] | Chi2 of reconstructed Tagging Tracker tracks. TagTrk2_track_chi2[0] - Leading momentum track  |
| RecTrk2_pp | Double_t[] | Reconstructed Recoil Tracker momentum [MeV]. RecTrk2_pp[0] - Leading momentum track |
| RecTrk2_track_No | Int_t | Number of reconstructed Recoil Tracker Tracks |
| RecTrk2_track_chi2 | Double_t[] | Chi2 of reconstructed Recoil Tracker tracks. RecTrk2_track_chi2[0] - Leading momentum track |
| ECAL_E_total | vector<double> | Total energy deposited in the ECAL [MeV]. ECAL_E_total[0] - Truth total energy. ECAL_E_total[1] - Smeard total energy with configuration 1. |
| ECAL_E_max | vector<double> | Maximum energy deposited of the ECAL Cell [MeV]. ECAL_E_max[0] - Truth maximum energy. ECAL_E_max[1] - Smeard maximum energy with configuration 1. |
| HCAL_E_total | vector<double> | Total energy deposited in the HCAL [MeV]. HCAL_E_total[0] -  |
| HCAL_E_Max_Cell | vector<double> | Maximum energy deposited of the HCAL Cell |

#### Examples Begin

User: Compare the `ECAL_E_total[0]` of signal and background events
Assistant: <code_interpreter type="exec" lang="python" filename="compare_histograms.py">
import ROOT
import argparse
...

def compare_histograms(branch: str, pre_selection: str, log_scale: bool, signal_dir: str, background_dir: str):
    # overlap histograms and with automatic range

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare histograms of signal and background events.')
    parser.add_argument('branch', nargs='?', default='ECAL_E_total[0]', help='Branch name to compare (default: %(default)s)')
    parser.add_argument('--pre-selection', default='', help='Pre-selection to apply (default: %(default)s)')
    parser.add_argument('--log-scale', action='store_true', help='Use log scale for y-axis')
    parser.add_argument('--signal-dir', default='eot/signal/invisible/mAp_100/dp_ana', help='Directory containing signal ROOT files (default: %(default)s)')
    parser.add_argument('--background-dir', default='eot/background/inclusive/dp_ana', help='Directory containing background ROOT files (default: %(default)s)')
    args = parser.parse_args()

    compare_histograms(args.branch, args.pre_selection, args.log_scale, args.signal_dir, args.background_dir)

</code_interpreter>
User: Alos plot `HCAL_E_total[0]`
Assistant: Re-using compareHistograms.C
<code_interpreter type="exec" lang="bash" filename="plot_HCAL_E_total_0.sh">
python compareHistograms.py HCAL_E_total[0]
</code_interpreter>

#### Examples End

### Cut-based Analysis

1. Define signal region according to physics knowledge
2. Decide an initial loose cut values for signal region
3. Optimize cuts to maximize significance
4. Draw and print cutflow
5. Recursively optimize cut until the significance is maximized
   - Vary signal region definition and cut values
   - Optimize cuts to maximize significance
   - Draw and print cutflow

#### Guidelines

- If exists multiple signal regions, signal regions should be orthogonal to each other

#### Examples Begin

User: Optimize cut of `ECAL_E_total[0]` with pre-selection `TagTrk2_track_No == 1 && RecTrk2_track_No == 1`
Assistant: <code_interpreter type="exec" lang="python" filename="optimize_cut.py">
import ROOT
import argparse
...

def optimize_cut(cut: str, pre_selection: str, signal_dir: str, background_dir: str):
    # load files
    # apply pre-selection when getting histogram of cut variable for signal and background
    # draw cumulative histograms of the cut varaible, save to figure with distinctable filename
    # calculate `S/sqrt(S+B)` for each cut value
    # draw S/sqrt(S+B) vs cut value, save to figure with distinctble filename
    # print the cut value, cut efficiency and significance for the optimized cut

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize cut value.')
    parser.add_argument('cut', nargs='?', default='ECAL_E_total[0]', help='Cut variable to optimize (default: %(default)s)')
    parser.add_argument('--pre-selection', default='TagTrk2_track_No == 1 && RecTrk2_track_No == 1', help='Pre-selection to apply (default: %(default)s)')
    parser.add_argument('--signal-dir', default='eot/signal/invisible/mAp_100/dp_ana', help='Directory containing signal ROOT files (default: %(default)s)')
    parser.add_argument('--background-dir', default='eot/background/inclusive/dp_ana', help='Directory containing background ROOT files (default: %(default)s)')
    args = parser.parse_args()

    optimize_cut(args.cut, args.pre_selection, args.signal_dir, args.background_dir)

</code_interpreter>

#### Examples End

## Task:

- You are a independent, patient, careful and accurate assistant, utilizing tools to help user. You analysis the chat history, decide and determine wether to use tool, or simply response to user. You can call tools by using xml node. Available Tools: Code Interpreter, Web Search, or Knowledge Search.

## Guidelines:

- Provide an overall plan (task list) in markdown to describe how to solve the problem. When planning, don't write code, don't call tools, don't use XML tag.
- Check the chat history to see if there are anything left that are waiting to be executed by tool. Call tool to solve it.
- Check if all the tools is running succesfully, if not, solve it by refine and retry the tool.
- If there are anything unclear, unexpected, or require validation, make it clear by iteratively use tool, until everything is clear with it's own reference (from tool). **DO NOT make ANY assumptions, DO NOT make-up any reply, DO NOT turn to user for information**.
- Always aim to deliver meaningful insights, iterating if necessary.
- All responses should be communicated in the chat's primary language, ensuring seamless understanding. If the chat is multilingual, default to English for clarity.
- DO NOT put any tool inside or outside markdown code block. That means you can output:
<tool ...>...</tool>
But never output something like:
```xml
<tool ...>...</tool>
```

"""
        self.VISION_MODEL_PROMPT: str = """Please briefly explain and analyze this figure. If it's a histogram, also tell if the histogram binning and plotting range is suitable for the dataset."""

        self.TOOL = {}
        self.prompt_templates = {}
        self.replace_tags = {"web_search": "Searching", "knowledge_search": "Searching"}
        # Global vars
        self.emitter = None
        self.total_response = ""
        self.temp_content = ""  # Temporary string to hold accumulated content
        self.current_tag_name = None
        self.immediate_stop = False
        self._init_knowledge()
        self.code_worker = None
        self.op_system = "Linux"  # code worker system

    def _init_knowledge(self):
        """
        初始化知识库数据，将其存储在 self.knowledges 字典中。
        """
        log.debug("Initializing knowledge bases")
        self.knowledges = {}  # 初始化知识库字典
        try:
            knowledge_bases = (
                Knowledges.get_knowledge_bases()
            )  # 获取所有知识库 # FIXME: 暂时只适用于admin.对于user需要获取uuid...

            # 遍历知识库列表
            for knowledge in knowledge_bases:
                knowledge_name = knowledge.name  # 获取知识库名称
                if knowledge_name:  # 确保知识库名称存在
                    log.debug(f"Adding knowledge base: {knowledge_name}")
                    self.knowledges[knowledge_name] = (
                        knowledge  # 将知识库信息存储到字典中
                    )
                else:
                    log.warning("Found a knowledge base without a name, skipping it.")

            log.info(
                f"Initialized {len(self.knowledges)} knowledge bases: {list(self.knowledges.keys())}"
            )
        except Exception as e:
            log.debug(f"Error initializing knowledge: {e}")

    def pipes(self):
        self.max_loop = self.valves.MAX_LOOP  # Save moneya
        if self.valves.USE_CODE_INTERPRETER:
            self.TOOL["code_interpreter"] = self._code_interpreter
            self.prompt_templates["code_interpreter"] = self.CODE_INTERPRETER_PROMPT
            self.init_code_worker()
        else:
            if "code_interpreter" in self.TOOL.keys():
                self.TOOL.pop("code_interpreter")
            if "code_interpreter" in self.prompt_templates.keys():
                self.prompt_templates.pop("code_interpreter")
        if self.valves.USE_WEB_SEARCH:
            self.TOOL["web_search"] = self._web_search
            self.prompt_templates["web_search"] = self.WEB_SEARCH_PROMPT
        else:
            if "web_search" in self.TOOL.keys():
                self.TOOL.pop("web_search")
            if "web_search" in self.prompt_templates.keys():
                self.prompt_templates.pop("web_search")
        if self.valves.USE_KNOWLEDGE_SEARCH:
            self.TOOL["knowledge_search"] = self._knowledge_search
            self.prompt_templates["knowledge_search"] = self.KNOWLEDGE_SEARCH_PROMPT
        else:
            if "knowledge_search" in self.TOOL.keys():
                self.TOOL.pop("knowledge_search")
            if "knowledge_search" in self.prompt_templates.keys():
                self.prompt_templates.pop("knowledge_search")

        return [
            {
                "id": self.valves.DEEPSEEK_API_MODEL,
                "name": self.valves.DEEPSEEK_API_MODEL,
            }
        ]

    async def pipe(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:
        """主处理管道（已移除缓冲）"""
        self.emitter = __event_emitter__

        # 验证配置
        if not self.valves.DEEPSEEK_API_KEY:
            yield json.dumps({"error": "未配置API密钥"}, ensure_ascii=False)
            return

        # 准备请求参数
        headers = {
            "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            # 模型ID提取
            model_id = body["model"].split(".", 1)[-1]
            payload = {**body, "model": model_id}

            messages = payload["messages"]

            # 检查最后一条user消息是否包含图片
            log.debug("Checking last user message for images")
            if messages[-1]["role"] == "user":
                content = messages[-1]["content"]
                if isinstance(content, List):
                    text_content = ""
                    # 查找文字内容
                    for c in content:
                        if c.get("type", "") == "text":
                            text_content = c.get("text", "")
                            log.debug(f"Found text in last user message: {text_content}")
                            break

                    # 查找图片内容
                    for c in content:
                        if c.get("type", "") == "image_url":
                            log.debug("Found image in last user message")
                            image_url = c.get("image_url", {}).get("url", "")
                            if image_url:
                                if image_url.startswith("data:image"):
                                    log.debug("Image URL is a data URL")
                                else:
                                    log.debug(f"Image URL: {image_url}")
                                # Query vision language model
                                vision_summary = await self._query_vision_model(
                                    self.VISION_MODEL_PROMPT, [image_url]
                                )
                                # insert to message content
                                text_content += vision_summary
                    # 替换消息
                    messages[-1]["content"] = text_content
                else:
                    image_urls = self._extract_image_urls(content)
                    if image_urls:
                        log.debug(f"Found image in last user message: {image_urls}")
                        # Call Vision Language Model
                        vision_summary = await self._query_vision_model(
                            self.VISION_MODEL_PROMPT, image_urls
                        )
                        content += vision_summary
                        messages[-1]["content"] = content

            # 确保user message是text-only
            log.debug("Checking all user messages content format")
            for msg in messages:
                if msg["role"] == "user":
                    content = msg["content"]
                    if isinstance(content, List):
                        log.debug("Found a list of content in user message")
                        text_content = ""
                        # 查找文字内容
                        for c in content:
                            if c.get("type", "") == "text":
                                text_content = c.get("content", "")
                                log.debug(f"Found text in user message: {text_content}")
                                break

                        # 替换消息
                        log.debug("Replacing user message content")
                        msg["content"] = text_content

            # User proxy转移到User 角色
            i = 0
            while i < len(messages):
                msg = messages[i]
                if msg["role"] == "assistant":
                    # 用正则匹配所有<details type="user_proxy">内容
                    user_proxy_nodes = re.findall(
                        r'<details type="user_proxy">(.*?)</details>',
                        msg["content"],
                        flags=re.DOTALL,
                    )

                    if user_proxy_nodes:
                        user_contents = []
                        for user_proxy_node in user_proxy_nodes:
                            user_proxy_text = str(user_proxy_node)
                            summary_node = re.search(
                                r"<summary>(.*?)</summary>",
                                user_proxy_text,
                                flags=re.DOTALL,
                            )
                            if summary_node:
                                summary_text = summary_node.group(1).strip()
                            else:
                                summary_text = ""
                            user_proxy_text = re.sub(
                                r"<summary>.*?</summary>",
                                "",
                                user_proxy_text,
                                flags=re.DOTALL,
                            ).strip()
                            user_contents.append(f"{summary_text}\n\n{user_proxy_text}")
                        merged_user_contents = "\n\n".join(user_contents)

                        # (1) 删除消息中的<user_proxy>标签（保留其他内容）
                        clean_content = re.sub(
                            r'<details type="user_proxy">.*?</details>',
                            "",
                            msg["content"],
                            flags=re.DOTALL,
                        ).strip()

                        msg["content"] = clean_content

                        new_user_msg = {"role": "user", "content": merged_user_contents}
                        messages.insert(i + 1, new_user_msg)  # 在当前消息后插入
                        i += 1

                i += 1

            # 处理消息以防止连续的相同角色
            i = 0
            while i < len(messages) - 1:
                if messages[i]["role"] == messages[i + 1]["role"]:
                    # 合并相同角色的消息
                    combined_content = (
                        messages[i]["content"] + "\n" + messages[i + 1]["content"]
                    )
                    messages[i]["content"] = combined_content
                    messages.pop(i + 1)
                i += 1

            self._set_system_prompt(messages)

            # yield json.dumps(payload, ensure_ascii=False)
            log.debug("Old message:")
            log.debug(messages[1:])

            # 发起API请求
            do_pull = True
            count = 0
            while do_pull and count < self.max_loop:
                thinking_state = {"thinking": -1}  # 使用字典来存储thinking状态
                async with self.client.stream(
                    "POST",
                    f"{self.valves.DEEPSEEK_API_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=300,
                ) as response:
                    # 错误处理
                    if response.status_code != 200:
                        error = await response.aread()
                        yield self._format_error(response.status_code, error)
                        return

                    # 流式处理响应
                    async for line in response.aiter_lines():
                        if not line.startswith(self.data_prefix):
                            continue

                        # 截取 JSON 字符串
                        json_str = line[len(self.data_prefix) :]

                        # 去除首尾空格后检查是否为结束标记
                        if json_str.strip() == "[DONE]":
                            return

                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            # 格式化错误信息，这里传入错误类型和详细原因（包括出错内容和异常信息）
                            error_detail = f"解析失败 - 内容：{json_str}，原因：{e}"
                            yield self._format_error("JSONDecodeError", error_detail)
                            return

                        choice = data.get("choices", [{}])[0]

                        # 结束条件判断
                        if choice.get("finish_reason") or self.immediate_stop:
                            if not self.immediate_stop:
                                res, tag_name = self._filter_response_tag()
                                yield res
                                # Clean up
                                if self.temp_content:
                                    await asyncio.sleep(0.1)
                                    yield self.temp_content
                                    self.temp_content = ""
                            self.immediate_stop = False
                            self.current_tag_name = None
                            self.total_response = self.total_response.lstrip("\n")
                            tools = self._find_tool_usage(self.total_response)
                            ## 防止奇数反引号
                            # lines = self.total_response.split('\n')
                            # backtick_count = sum(1 for line in lines if line.startswith('```'))
                            # if backtick_count % 2 != 0:
                            #    self.total_response += "\n```\n\n"
                            #    await asyncio.sleep(0.1)
                            #    yield "\n"
                            #    await asyncio.sleep(0.1)
                            #    yield "```"
                            #    await asyncio.sleep(0.1)
                            #    yield"\n\n"
                            # Move total_response to messages
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": self.total_response,
                                }
                            )
                            self.total_response = ""
                            # =================================================
                            # Call tools
                            # =================================================
                            if tools is not None:
                                do_pull = True
                                user_proxy_reply = ""
                                for tool in tools:
                                    summary, content = await self.TOOL[tool["name"]](
                                        tool["attributes"], tool["content"]
                                    )
                                    await asyncio.sleep(0.5)

                                    # Check if content contains figures
                                    image_urls = self._extract_image_urls(content)

                                    if image_urls:
                                        # Call Vision Language Model
                                        figure_summary = await self._query_vision_model(self.VISION_MODEL_PROMPT, image_urls)
                                        content += figure_summary

                                    user_proxy_reply += f"{summary}\n\n{content}\n\n"
                                    yield f'\n<details type="user_proxy">\n<summary>{summary}</summary>\n{content}\n</details>\n'

                                messages.append(
                                    {
                                        "role": "user",
                                        "content": user_proxy_reply,
                                    }
                                )
                            else:
                                do_pull = False
                            break

                        # 状态机处理
                        state_output = await self._update_thinking_state(
                            choice.get("delta", {}), thinking_state
                        )
                        if state_output:
                            yield state_output  # 直接发送状态标记
                            if state_output == "<think>":
                                await asyncio.sleep(0.1)
                                yield "\n"

                        # 内容处理并立即发送
                        content = self._process_content(choice["delta"])
                        if content:
                            if content.startswith("<think>"):
                                match = re.match(r"^<think>", content)
                                if match:
                                    content = re.sub(r"^<think>", "", content)
                                    yield "<think>"
                                    await asyncio.sleep(0.1)
                                    yield "\n"

                            elif content.startswith("</think>"):
                                match = re.match(r"^</think>", content)
                                if match:
                                    content = re.sub(r"^</think>", "", content)
                                    yield "</think>"
                                    await asyncio.sleep(0.1)
                                    yield "\n"
                            if thinking_state["thinking"] != 0:
                                res, tag_name = self._filter_response_tag(content)
                                # if tag_name == "knowledge_search":
                                #    self.immediate_stop = True
                                # if tag_name in ["web_search","knowledge_search"]:
                                if "</code_interpreter>" in self.total_response:
                                    self.immediate_stop = True
                                if tag_name:
                                    self.current_tag_name = tag_name
                                if tag_name is None and self.current_tag_name:
                                    if res:
                                        self.immediate_stop = True
                                        self.current_tag_name = None
                                        self.temp_content = ""
                                        res = ""
                                        # clip total response:
                                        self.total_response = self.total_response[
                                            : -len(content)
                                        ]
                                if res:
                                    yield res

                            else:
                                yield content
                log.debug(messages[1:])
                count += 1
        except Exception as e:
            yield self._format_exception(e)

    async def _update_thinking_state(self, delta: dict, thinking_state: dict) -> str:
        """更新思考状态机（简化版）"""
        state_output = ""

        # 状态转换：未开始 -> 思考中
        if thinking_state["thinking"] == -1 and delta.get("reasoning_content"):
            thinking_state["thinking"] = 0
            state_output = "<think>"

        # 状态转换：思考中 -> 已回答
        elif (
            thinking_state["thinking"] == 0
            and not delta.get("reasoning_content")
            and delta.get("content")
        ):
            thinking_state["thinking"] = 1
            state_output = "\n</think>\n\n"

        return state_output

    def _process_content(self, delta: dict) -> str:
        """直接返回处理后的内容"""
        if delta.get("reasoning_content", ""):
            return delta.get("reasoning_content", "")
        elif delta.get("content", ""):
            delta = delta.get("content", "")
            self.total_response += delta
            return delta

    def _filter_response_tag(self, content: str = "") -> str:
        tag_name = None
        self.temp_content += content
        res = ""
        if "<" in self.temp_content:
            # Conver tool calling tags into content (Except code_interpreter, let openwebui to handle)
            if len(self.temp_content) > 20:
                if (
                    "<web_search" in self.temp_content
                    or "<knowledge_search" in self.temp_content
                ):
                    pattern = re.compile(
                        r"^<(web_search|knowledge_search)\s*([^>]*)>(.*?)</\1>",
                        re.DOTALL | re.MULTILINE,
                    )
                    # Find all matches in the self.temp_content
                    match = pattern.search(self.temp_content)
                    if match:
                        tag_name = match.group(1)
                        attributes_str = match.group(2)
                        tag_content = match.group(3).strip()
                        summary = (
                            self.replace_tags[tag_name]
                            + " "
                            + tag_content
                            + " in "
                            + attributes_str
                        )
                        res = (
                            self.temp_content[: match.start()]
                            + f'\n<details type="{tag_name}">\n<summary>{summary}</summary>\n{tag_content}\n</details>'
                        )
                        self.temp_content = self.temp_content[match.end() :]
                else:
                    res = self.temp_content
                    self.temp_content = ""
        else:
            res = self.temp_content
            self.temp_content = ""
        return res, tag_name

    def _format_error(self, status_code: int, error: bytes) -> str:
        # 如果 error 已经是字符串，则无需 decode
        if isinstance(error, str):
            error_str = error
        else:
            error_str = error.decode(errors="ignore")

        try:
            err_msg = json.loads(error_str).get("message", error_str)[:200]
        except Exception as e:
            err_msg = error_str[:200]
        return json.dumps(
            {"error": f"HTTP {status_code}: {err_msg}"}, ensure_ascii=False
        )

    def _format_exception(self, e: Exception) -> str:
        """异常格式化保持不变"""
        err_type = type(e).__name__
        return json.dumps({"error": f"{err_type}: {str(e)}"}, ensure_ascii=False)

    async def _web_search(self, attributes: dict, content: str) -> Tuple[str, str]:
        # Extract the search query from the content
        search_query = content.strip()

        if not search_query:
            return "No search query provided", ""

        engine = attributes.get("engine", "")

        # Handle Google Custom Search
        if (
            engine == "google"
            and self.valves.GOOGLE_PSE_API_KEY
            and self.valves.GOOGLE_PSE_ENGINE_ID
        ):
            # Construct the Google search URL
            google_search_url = f"https://www.googleapis.com/customsearch/v1?q={search_query}&key={self.valves.GOOGLE_PSE_API_KEY}&cx={self.valves.GOOGLE_PSE_ENGINE_ID}&num=5"

            try:
                response = await self.client.get(google_search_url)
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])
                    search_results = []
                    urls = []
                    for item in items:
                        title = item.get("title", "No title")
                        link = item.get("link", "No link")
                        urls.append(link)
                        snippet = item.get("snippet", "No snippet")
                        search_results.append(f"**{title}**\n{snippet}\n{link}\n")

                    if search_results:
                        result = "\n\n".join(search_results)
                        return f"Searched {len(urls)} sites", result
                    else:
                        return "No results found on Google", search_query
                else:
                    return (
                        f"Google search failed with status code {response.status_code}",
                        search_query,
                    )
            except Exception as e:
                return "Error during Google search", f"{str(e)}\nQuery: {search_query}"

        # Handle ArXiv search
        if engine == "arxiv":
            arxiv_search_url = f"http://export.arxiv.org/api/query?search_query=all:{search_query}&start=0&max_results=5"

            try:
                response = await self.client.get(arxiv_search_url)
                if response.status_code == 200:
                    data = response.text
                    # Extract entries using regex
                    pattern = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
                    matches = pattern.findall(data)

                    arxiv_results = []
                    urls = []
                    for match in matches:
                        title_match = re.search(r"<title>(.*?)</title>", match)
                        link_match = re.search(r"<id>(.*?)</id>", match)
                        summary_match = re.search(
                            r"<summary>(.*?)</summary>", match, re.DOTALL
                        )

                        if title_match and link_match and summary_match:
                            title = title_match.group(1)
                            link = link_match.group(1)
                            urls.append(link)
                            summary = summary_match.group(1).strip()
                            arxiv_results.append(f"**{title}**\n{summary}\n{link}\n")
                        else:
                            log.error("Error parsing ArXiv entry.")

                    if arxiv_results:
                        result = "\n\n".join(arxiv_results)
                        return f"Searched {len(urls)} papers", result
                    else:
                        return "No results found on ArXiv", search_query
                else:
                    return (
                        f"ArXiv search failed with status code {response.status_code}",
                        search_query,
                    )
            except Exception as e:
                return "Error during ArXiv search", f"{str(e)}\nQuery: {search_query}"

        return (
            "Invalid search source or query",
            f"Search engine: {engine}\nQuery:{search_query}",
        )


    async def _generate_openai_batch_embeddings(
        self,
        model: str,
        texts: List[str],
        url: str = "https://api.openai.com/v1",
        key: str = "",
    ) -> Optional[List[List[float]]]:
        try:
            # Construct the API request
            response = await self.client.post(
                f"{url}/embeddings",
                json={"input": texts, "model": model},
                headers={"Authorization": f"Bearer {key}"},
            )

            # Check for valid response
            response.raise_for_status()  # Will raise an HTTPError for bad responses

            # Parse and return embeddings if available
            data = response.json()
            if "data" in data:
                return [elem["embedding"] for elem in data["data"]]
            else:
                raise ValueError("Response from OpenAI API did not contain 'data'.")
        except httpx.HTTPStatusError as e:
            log.error(
                f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            log.error(f"An error occurred while generating embeddings: {str(e)}")
        return None

    async def _query_collection(
        self, knowledge_name: str, query_keywords: str, top_k: int = 3
    ) -> list:
        """
        Query the vector database by knowledge name and keywords, and return metadata and contexts.

        Args:
            knowledge_name (str): The name of the knowledge to query.
            query_keywords (str): The query keywords to search for.
            top_k (int): The number of top results to retrieve.

        Returns:
            list: A list of dictionaries containing metadata and context documents.
        """
        retries = 3
        embeddings = None
        query_keywords = query_keywords.strip()
        for i in range(retries):
            log.debug(f"Generating Embeddings")
            try:
                embeddings = await self._generate_openai_batch_embeddings(
                    model="Pro/BAAI/bge-m3",
                    texts=[query_keywords],
                    url=self.valves.DEEPSEEK_API_BASE_URL,
                    key=self.valves.DEEPSEEK_API_KEY,
                )
                if isinstance(embeddings, list):
                    embeddings = embeddings[0]
                    break
            except Exception as e:
                log.error(f"Generating Embeddings attempt {i + 1} failed: {e}")
                await asyncio.sleep(2)
        if not embeddings:
            raise ValueError(
                f"Faild generating embeddings, could be a network fluctuation."
            )
        log.debug(f"Embeddings length: {len(embeddings)}")
        log.debug("Searching VECTOR_DB_CLIENT")
        knowledge = self.knowledges.get(knowledge_name, [])
        if not knowledge:
            raise ValueError(
                f"No knowledge name {knowledge_name} found in knowledge base. Availables knowledges: {vars(self.knowledges.keys())}"
            )

        file_ids = knowledge.data["file_ids"]
        all_results = []

        for file_id in file_ids:
            file_name = "file-" + file_id
            result = VECTOR_DB_CLIENT.search(
                collection_name=file_name,
                vectors=[embeddings],
                limit=top_k,
            )

            if not result or not hasattr(result, "ids") or not result.ids:
                continue

            if not all(
                hasattr(result, attr)
                for attr in ["ids", "distances", "documents", "metadatas"]
            ):
                continue

            if (
                not result.ids
                or not result.distances
                or not result.documents
                or not result.metadatas
            ):
                continue

            for i in range(len(result.documents)):
                result_object = ResultObject(
                    id=result.ids[i],
                    distance=result.distances[i],
                    document=result.documents[i],
                    metadata=result.metadatas[i],
                )
                all_results.append(result_object)

        # Sort all results by distance and select the top_k
        all_results.sort(key=lambda x: x.distance)
        top_results = all_results[:top_k]

        return top_results

    async def _knowledge_search(
        self, attributes: dict, content: str
    ) -> Tuple[str, str]:
        """
        Retrieve relevant information from a knowledge collection
        based on the provided query, and return the results.

        Args:
            attributes (dict): Attributes containing information such as the knowledge base.
            content (str): The query to search for in the knowledge base.

        Returns:
            Tuple(str, str): The retrieved relevant content or an error message.
        """
        collection = attributes.get("collection", "")
        content = content.strip()
        if not collection:
            return "Error: No knowledge search collection specified", ""

        # Retrieve relevant documents from the knowledge base
        try:
            results = await self._query_collection(collection, content)

            if not results:
                return f"Found nothing about {content}", f"Collection: {collection}"

        except Exception as e:
            return f"Faild searching {content}", f"Collection: {collection}"

        try:
            # Format the results for output
            formatted_results = []
            for result in results:
                if not result.metadata or not result.document:
                    continue
                if not isinstance(result.metadata, list) or not result.metadata:
                    continue
                if not isinstance(result.document, list) or not result.document:
                    continue
                source = result.metadata[0]["source"]
                source = source.replace("__", "/")
                document = result.document[0]
                document = "\n".join(["> " + line for line in document.splitlines()])

                formatted_results.append(
                    f"\n**Source**: {source}\n\n**Context**:\n\n{document}\n"
                )
            reply = "\n\n".join(formatted_results)
            return f"Found {len(results)} results", reply
        except Exception as e:
            return f"Error during formatting result for {content}", f"{str(e)}"

    def _find_tool_usage(self, content):
        tools = []
        # Define the regex pattern to match the XML tags
        pattern = re.compile(
            r"<(code_interpreter|web_search|knowledge_search)\s*([^>]*)>(.*?)</\1>",
            re.DOTALL | re.MULTILINE,
        )

        # Find all matches in the content
        matches = pattern.findall(content)

        # If no matches found, return None
        if not matches:
            return None

        for match in matches:
            # Extract the tag name, attributes, and content
            tag_name = match[0]
            attributes_str = match[1]
            tag_content = match[2].strip()

            # Extract attributes into a dictionary
            attributes = {}
            for attr in attributes_str.split():
                if "=" in attr:
                    key, value = attr.split("=", 1)
                    value = value.strip("\"'")
                    attributes[key] = value

            # Return the XML information
            tools.append(
                {"name": tag_name, "attributes": attributes, "content": tag_content}
            )

        return tools

    def _set_system_prompt(self, messages):
        if len(self.prompt_templates) == 0:
            return ""
        template_string = """## Available Tools\n"""
        for i, (name, prompt) in enumerate(self.prompt_templates.items()):
            template_string += f"\n### {i+1}. {prompt}\n"
        template_string += self.GUIDE_PROMPT
        # Create a Jinja2 Template object
        template = Template(template_string)

        current_date = datetime.now()
        formatted_date = current_date.strftime("%Y-%m-%d")
        # Render the template with a list of items
        context = {"CURRENT_DATE": formatted_date, "OP_SYSTEM": self.op_system}
        result = template.render(**context)
        # Set system_prompt
        if messages[0]["role"] == "system":
            messages[0]["content"] = result
        else:
            context_message = {"role": "system", "content": result}
            messages.insert(0, context_message)

        #log.debug("Current System Prompt:")
        #log.debug(result)

    # =========================================================================
    # Code Interpreter
    # =========================================================================
    def init_code_worker(self):
        try:
            self.code_worker = HRModel.connect(
                name="xuliang/code-worker-v2",
                base_url="http://localhost:42899/apiv2",
            )
            funcs = self.code_worker.functions()  # Get all remote callable functions.
            log.info(f"Remote callable funcs: {funcs}")
            self.op_system = self.code_worker.inspect_system()
        except Exception as e:
            log.error(f"Error initializing code worker: {e}")

    async def _code_interpreter(
        self, attributes: dict, content: str
    ) -> Tuple[str, str]:
        if self.code_worker is None:
            self.init_code_worker()

        # Extract the code interpreter type and language
        code_type = attributes.get("type", "")
        lang = attributes.get("lang", "")
        filename = attributes.get("filename", "")
        if code_type == "exec":
            # Execute the code
            if filename:
                try:
                    result = self.code_worker.write_code(
                        file_path=filename,
                        content=content,
                        execute=True,
                        lang=lang,
                        timeout=300,
                    )
                    return f"Executed code: {filename}", result
                except Exception as e:
                    return f"Error executing {filename}", f"{str(e)}"
            elif lang == "bash":
                try:
                    result = self.code_worker.run_command(command=content, timeout=300)
                    return "Command executed", result
                except Exception as e:
                    return "Error executing bash command", f"{str(e)}"
            else:
                return "No filename provided for code execution", "Please provide filename in xml attribute."

        elif code_type == "write":
            if not filename:
                return "No filename provided for code writing", "Please provide filename in xml attribute."
            # Write the code to a file
            try:
                result = self.code_worker.write_code(
                    file_path=filename, content=content
                )
                return f"Written file: {filename}", result
            except Exception as e:
                return f"Error writing {filename}", f"{str(e)}"

        elif code_type == "search_replace":
            if not filename:
                return "No filename provided for code search and replace", "Please provide filename in xml attribute."
            # extract the original and updated code
            edit_block_pattern = re.compile(
                r"<<<<<<< ORIGINAL\s*(?P<original>.*?)"
                r"=======\s*(?P<mid>.*?)"
                r"\s*(?P<updated>.*)>>>>>>> ",
                re.DOTALL,
            )
            match = edit_block_pattern.search(content)
            if match:
                original = match.group("original")
                updated = match.group("updated")
                try:
                    result = self.code_worker.search_replace(
                        file_path=filename, original=original, updated=updated
                    )
                    return f"Updated {filename}", result
                except Exception as e:
                    return f"Error searching and replacing {filename}", f"{str(e)}"
            else:
                return "Invalid search and replace format", "Format: <<<<<<< ORIGINAL\nOriginal code\n=======\nUpdated code\n>>>>>>> UPDATED"
        else:
            return f"Invalid code interpreter type `{code_type}`", "Available types: `exec`, `write`"

    # =========================================================================
    # Vision Language Model
    # =========================================================================

    async def _generate_vl_response(
        self,
        prompt: str,
        image_url: str,
        model: str = "Qwen/Qwen2-VL-72B-Instruct",
        url: str = "https://api.siliconflow.cn/v1",
        key: str = "",
    ) -> str:
        try:
            payload = {
                "model": model,
                "messages": [
                     {
                         "role": "user",
                         "content": [
                             {
                                 "type": "image_url",
                                 "image_url": {
                                     "url": image_url,
                                     "detail": "high"
                                 }
                             },
                             {
                                 "type": "text",
                                 "text": prompt,
                             }
                         ]
                     }
                 ],
                "stream": False,
                "max_tokens": 512,
                "stop": None,
                "temperature": 0.7,
                "top_p": 0.7,
                "top_k": 50,
                "frequency_penalty": 0.5,
                "n": 1,
                "response_format": {"type": "text"},
            }
            response = requests.request(
                "POST",
                url=f"{url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                proxies = {
                   'http': 'http://127.0.0.1:7890',
                   'https': 'http://127.0.0.1:7890',
                }
            )

            # Check for valid response
            response.raise_for_status()

            # Parse and return embeddings if available
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            return f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
        except httpx.ReadTimeout as e:
            log.error(f"Read Timeout error occurred")
            return f"Read Timeout error occurred"

        return ""

    def _extract_image_urls(self, text: str) -> list:
        """
        Extract image URLs from text with 2 criteria:
        1. URLs ending with .png/.jpeg/.jpg/.gif/.svg (case insensitive)
        2. URLs in markdown image format regardless of extension
        
        Args:
            text: Input text containing potential image URLs
            
        Returns:
            List of unique image URLs sorted by first occurrence
        """
        # Match URLs with image extensions (including query parameters)
        ext_pattern = re.compile(
            r'https?:\/\/[^\s]+?\.(?:png|jpe?g|gif|svg)(?:\?[^\s]*)?(?=\s|$)', 
            re.IGNORECASE
        )
        
        # Match markdown image syntax URLs
        md_pattern = re.compile(
            r'!\[[^\]]*\]\((https?:\/\/[^\s\)]+)'
        )
        
        # Find all matches while preserving order
        seen = set()
        result = []
        
        for match in ext_pattern.findall(text) + md_pattern.findall(text):
            if match not in seen:
                seen.add(match)
                result.append(match)
        
        return result

    async def _query_vision_model(
        self,
        prompt: str,
        image_urls: List[str],
    ) -> str:
        response = ""
        i = 1
        for url in image_urls:
            if not url.startswith("data:image"):
                log.debug(f"Processing image {i}: {url}")
            fig_name = f"Figure {i}:"
            vl_res = await self._generate_vl_response(
                prompt=prompt,
                image_url=url,
                model="Qwen/Qwen2-VL-72B-Instruct",
                url=self.valves.DEEPSEEK_API_BASE_URL,
                key=self.valves.DEEPSEEK_API_KEY
            )
            response += f"\n\n{fig_name}\n\n{vl_res}"
        return response

