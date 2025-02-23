"""
title: Welt
author: Xuliang
description: OpenWebUI pipe function for Welt: Workflow Enhanced LLM with CoT
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
        self.client = httpx.AsyncClient(http2=True)
        self.CODE_INTERPRETER_PROMPT: str = """Code Interpreter


You have access to a user's code workspace, use `<code_interpreter>` XML tag to execute or save code to do analysis, calculations, or problem-solving. Here's how it works:

<code_interpreter type="exec" lang="python" filename="">
code here
</code_interpreter>

#### Tool Attributes

- `type`: Specifies the action to perform.
   - `exec`: Execute the code immediately.
      - Supported languages: `python`, `root`, `bash`
   - `save`: Save the code to the user's workspace.
      - Supports any programming language.
   - `search_replace`: search keyword and replace

- `filename`: The file path where the code will be saved.  
   - Must be **relative to the user's workspace base directory**, do not use paths relative to subdirectory.

#### Usage Instructions

- The Python code you write can incorporate a wide array of libraries, handle data manipulation or visualization, perform API calls for web-related tasks, or tackle virtually any computational challenge. Use this flexibility to **think outside the box, craft elegant solutions, and harness Python's full potential**.
- Output XML node simply like `<code_interpreter ...>...</code_interpreter>`, DO NOT put XML node inside the markdown code block (```xml). 
- When coding, **always aim to print meaningful outputs** (e.g., results, tables, summaries, or visuals) to better interpret and verify the findings. Avoid relying on implicit outputs; prioritize explicit and clear print statements so the results are effectively communicated to the user.
- About code style:
   - Prefer object-oriented programming
   - Prefer arguments with default value than hard coded
   - For potentially time-comsuming code, e.g. loading file with unknown size, use argument to control the running scale, and defaulty run on small scale test.

#### Example 1

User: plot something
Assistant: ...

**Calling Code Intepreter:**

<code_interpreter type="exec" lang="python" filename="plot.py">
# plotting code here
</code_interpreter>

#### Example 2:

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

#### Example 3:

User: I have a existing file in `analysis.C`, with content
```
        declareProperty("IsSignalMC", m_IsSignalMC = 0);
        declareProperty("Ecms", m_Ecms = ECMS);
```
please add a line `declareProperty("IsExample", m_IsExample = false);` after it.
Assistant: ...

**Calling Code Intepreter:**

<code_interpreter type="search_replace" lang="diff" filename="HelloWorld/src/main.cpp">
<<<<<<< ORIGINAL
        declareProperty("IsSignalMC", m_IsSignalMC = 0);
        declareProperty("Ecms", m_Ecms = ECMS);
=======
        declareProperty("IsSignalMC", m_IsSignalMC = 0);
        declareProperty("Ecms", m_Ecms = ECMS);
        declareProperty("IsExample", m_IsExample = false);
>>>>>>> UPDATED
</code_interpreter>

"""
        self.WEB_SEARCH_PROMPT: str = """Web Search

- You have access to internet, use `<web_search>` XML tag to search the web for new information and references. Example:

**Calling Web Search Tool:**

<web_search engine="google">first query here</web_search>
<web_search engine="google">second query here</web_search>

#### Tool Attributes

- `engine`: available option:
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

- You have access to user's database, use `<knowledge_search>` XML tag to search user's internal and personal documens. Example:

**Calling Knowledge Search Tool:**

<knowledge_search collection="DarkSHINE_Simulation_Software">one query</knowledge_search>

#### Tool Attributes
  - `collection`: Available option:
     - `DarkSHINE_Simulation_Software`: Source code of simulation program based on Geant4 and ROOT, characterized by detector of DarkSHINE experiment. **Must use English to query this collection**.
     - `OpenWebUI_Backend`: Source code of backend of OpenWebUI, an extensible, self-hosted AI interface.

#### Usage Insructions

- In each `<knowledge_search>` XML tag, be concise and focused on composing high-quality search query, **avoiding unnecessary elaboration, commentary, or assumptions**.
- Enclose only one query in one pair of `<knowledge_search collection="...">` `</knowledge_search>` XML tags.
"""
        self.GUIDE_PROMPT: str = """## Task:

- You are a independent, patient, careful and accurate assistant, utilizing tools to help user. You analysis the chat history, decide and determine wether to use tool, or simply response to user. You can call tools by using xml node. Available Tools: Code Interpreter, Web Search, or Knowledge Search.

## Guidelines:

- Provide an overall plan in markdown to describe how to solve the problem. When planning, don't call tools, don't use XML tag.
- Check the chat history to see if there are anything left that are waiting to be executed by tool. Call tool to solve it.
- Check if all the tools is running succesfully, if not, solve it by refine and retry the tool.
- If there are anything unclear, unexpected, or require validation, make it clear by iteratively use tool, until everything is clear with it's own reference (from tool). **DO NOT make ANY assumptions, DO NOT make-up any reply, DO NOT turn to user for information**.
- Always aim to deliver meaningful insights, iterating if necessary.
- All responses should be communicated in the chat's primary language, ensuring seamless understanding. If the chat is multilingual, default to English for clarity.
- DO NOT put any tool inside any markdown code block. That means you can output:
<tool ...>...</tool>
But never output something like:
```xml
<tool ...>...</tool>
```
"""

        self.TOOL = {}
        self.prompt_templates = {}
        self.replace_tags = {
            "web_search": "Searching",
            "knowledge_search": "Searching"
        }
        # Global vars
        self.emitter = None
        self.total_response = ""
        self.temp_content = ""  # Temporary string to hold accumulated content
        self.current_tag_name = None
        self.immediate_stop = False
        self._init_knowledge()

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

            # User proxy转移到User 角色
            i = 0
            while i < len(messages):
                msg = messages[i]
                if msg["role"] == "assistant":
                    # 用正则匹配所有<details type="user_proxy">内容
                    user_proxy_nodes = re.findall(r'<details type="user_proxy">(.*?)</details>', msg["content"], flags=re.DOTALL)
                    
                    if user_proxy_nodes:
                        user_contents = []
                        for user_proxy_node in user_proxy_nodes:
                            user_proxy_text = str(user_proxy_node)
                            summary_node = re.search(r'<summary>(.*?)</summary>', user_proxy_text, flags=re.DOTALL)
                            if summary_node:
                                summary_text = summary_node.group(1).strip()
                            else:
                                summary_text = ""
                            user_proxy_text = re.sub(r'<summary>.*?</summary>', "",user_proxy_text, flags=re.DOTALL).strip()
                            user_contents.append(f"{summary_text}\n\n{user_proxy_text}")
                        merged_user_contents = '\n\n'.join(user_contents)

                        # (1) 删除消息中的<user_proxy>标签（保留其他内容）
                        clean_content = re.sub(
                            r'<details type="user_proxy">.*?</details>',
                            '',
                            msg["content"],
                            flags=re.DOTALL
                        ).strip()

                        msg["content"] = clean_content
                        
                        new_user_msg = {
                            "role": "user",
                            "content": merged_user_contents
                        }
                        messages.insert(i+1, new_user_msg)  # 在当前消息后插入
                        i += 1

                i += 1

            # 处理消息以防止连续的相同角色
            i = 0
            while i < len(messages) - 1:
                if messages[i]["role"] == messages[i + 1]["role"]:
                    # 合并相同角色的消息
                    combined_content = messages[i]["content"] + "\n" + messages[i+1]["content"]
                    messages[i]["content"] = combined_content
                    messages.pop(i+1)
                i += 1

            self._set_system_prompt(messages)

            # yield json.dumps(payload, ensure_ascii=False)
            log.debug("Old message:")
            log.debug(messages)

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
                            yield self._format_error(
                                "JSONDecodeError", error_detail
                            )
                            return

                        choice = data.get("choices", [{}])[0]

                        # 结束条件判断
                        if choice.get("finish_reason") or self.immediate_stop:
                            if not self.immediate_stop:
                                res, tag_name= self._filter_response_tag()
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
                            #lines = self.total_response.split('\n')
                            #backtick_count = sum(1 for line in lines if line.startswith('```'))
                            #if backtick_count % 2 != 0:
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
                            if tools is not None:
                                do_pull = True
                                # Call tools
                                user_proxy_reply = ""
                                for tool in tools:
                                    summary, content = await self.TOOL[tool["name"]](
                                        tool["attributes"], tool["content"]
                                    )
                                    user_proxy_reply += f"{summary}\n\n{content}\n\n" 
                                    await asyncio.sleep(0.1)
                                    yield f'\n<details type=\"user_proxy\">\n<summary>{summary}</summary>\n{content}\n</details>\n'

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
                                if tag_name == "knowledge_search":
                                    self.immediate_stop = True
                                if tag_name == "web_search":
                                    self.current_tag_name = tag_name
                                if tag_name is None and self.current_tag_name == "web_search":
                                    if res:
                                        self.immediate_stop = True
                                        self.current_tag_name = None
                                        self.temp_content = ""
                                        res = ""
                                        # clip total response:
                                        self.total_response = self.total_response[:-len(content)]
                                if res:
                                    yield res
                                       
                            else:
                                yield content
                log.debug(messages)
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
                        summary = self.replace_tags[tag_name] + " " + tag_content + " in " + attributes_str
                        res = self.temp_content[:match.start()] + f'\n<details type="{tag_name}">\n<summary>{summary}</summary>\n{tag_content}\n</details>'
                        self.temp_content = self.temp_content[match.end():]
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
                    return f"Google search failed with status code {response.status_code}", search_query
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
                            arxiv_results.append(
                                f"**{title}**\n{summary}\n{link}\n"
                            )
                        else:
                            log.error("Error parsing ArXiv entry.")

                    if arxiv_results:
                        result = "\n\n".join(arxiv_results)
                        return f"Searched {len(urls)} papers", result
                    else:
                        return "No results found on ArXiv", "search_query"
                else:
                    return f"ArXiv search failed with status code {response.status_code}", search_query
            except Exception as e:
                return "Error during ArXiv search", f"{str(e)}\nQuery: {search_query}"

        return "Invalid search source or query", f"Search engine: {engine}\nQuery:{search_query}"

    async def _code_interpreter(self, attributes: dict, content: str) -> Tuple[str, str]:
        return "Done", ""

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
            log.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
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
                    model="BAAI/bge-m3",
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
            raise ValueError(f"Faild generating embeddings, could be a network fluctuation.")
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

    async def _knowledge_search(self, attributes: dict, content: str) -> Tuple[str, str]:
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
        context = {"CURRENT_DATE": formatted_date}
        result = template.render(**context)
        # Set system_prompt
        if messages[0]["role"] == "system":
            messages[0]["content"] = result
        else:
            context_message = {"role": "system", "content": result}
            messages.insert(0, context_message)

        log.debug("Current System Prompt:")
        log.debug(result)

