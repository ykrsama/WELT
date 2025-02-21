"""
title: WELT
author: Xuliang
description: Workflow Enhanced LLM with CoT
version: 1.2.10
licence: MIT
"""

import logging
import io, sys
import json
import httpx
import re
from typing import AsyncGenerator, Callable, Awaitable, Optional, List
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
)

log = logging.getLogger(__name__)
log.setLevel("DEBUG")


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
        KNOWLEDGE_COLLECTIONS: str = Field(
            default="DSimu, DAna",
            description="ID of knowledge collections, seperate by comma",
        )
        EMBEDDING_BATCH_SIZE: int = (
            Field(
                default=2000,
                description="Batch size for knowledge search",
            ),
        )
        GOOGLE_PSE_API_KEY: str = Field(default="")
        GOOGLE_PSE_ENGINE_ID: str = Field(default="")
        MAX_LOOP: int = Field(default=20, description="Prevent dead loop")
        CODE_INTERPRETER_PROMPT: str = Field(
            default="""**Code Interpreter**:
   <code_interpreter type="code" lang="python">
   codes
   </code_interpreter>
   - You have access to a Python shell that runs directly in the user's browser, enabling fast execution of code for analysis, calculations, or problem-solving.
   - The Python code you write can incorporate a wide array of libraries, handle data manipulation or visualization, perform API calls for web-related tasks, or tackle virtually any computational challenge. Use this flexibility to **think outside the box, craft elegant solutions, and harness Python's full potential**.
   - To use it, **you must enclose your code within `<code_interpreter type="code" lang="python">`, `</code_interpreter>` XML tags** and stop right away. If you don't, the code won't execute. DO NOT use markdown code block, because enclosing markdown codeblock with special XML tags will confuse user's browser frontend.
   - When coding, **always aim to print meaningful outputs** (e.g., results, tables, summaries, or visuals) to better interpret and verify the findings. Avoid relying on implicit outputs; prioritize explicit and clear print statements so the results are effectively communicated to the user.
   - No need to save plot, just show it.
   - If the results are unclear, unexpected, or require validation, refine the code and execute it again as needed. Always aim to deliver meaningful insights from the results, iterating if necessary.
   - **If a path to an image, audio, or any file is provided in markdown format in the output, ALWAYS regurgitate word for word, explicitly display it as part of the response to ensure the user can access it easily, do NOT change it.**
   - Code for test is important. For potentially time-comsuming code, e.g. loading file with unknown size, use argument to control the running scale, and defaulty run on small scale test.
"""
        )
        WEB_SEARCH_PROMPT: str = Field(
            default="""**Web Search**: `<web_search url="www.googleapis.com">single query</web_search>`
   - You have access to web search, and no need to bother API keys because user can handle by themselves in this tool.
   - To use it, **you must enclose your search queries within** `<web_search url="www.googleapis.com">`, `</web_search>` **XML tags** and stop responding right away without further assumption of what will be done. Do NOT use triple backticks.
   - Err on the side of suggesting search queries if there is **any chance** they might provide useful or updated information.
   - Always prioritize providing actionable and broad queries that maximize informational coverage.
   - In each web_search XML tag, be concise and focused on composing high-quality search queries, avoiding unnecessary elaboration, commentary, or assumptions.
   - You can use multiple lines of web_search XML tag to do parallel search.
   - Available urls:
     - www.googleapis.com: for general search.
     - arxiv.org: for academic paper research. Always use english keywords for arxiv.
   - If no valid search result, maybe causing by network issue, please retry.
   - **The date today is: {{CURRENT_DATE}}**. So you can search for web to get information up do date {{CURRENT_DATE}}.
"""
        )
        KNOWLEDGE_SEARCH_PROMPT: str = Field(
            default="""**Knowledge Search**: `<knowledge_search collection="DSimu">single query</knowledge_search>`
   - You have access to user's local and personal kowledge collections.
   - To use it, **you must enclose your search queries within** `<knowledge_search collection="DSimu">`, `</knowledge_search>` **XML tags** and stop responding right away without further assumption of what will be done. Do NOT use triple backticks.
   - Err on the side of suggesting search queries if there is **any chance** they might provide useful or related information.
   - In each knowledge_search XML tag, be concise and focused on composing high-quality search queries, avoiding unnecessary elaboration, commentary, or assumptions.
   - You can use multiple lines of knowledge_search XML tag to do parallel search.
   - Available collections:
     - DSimu: source codes of simulation program based on Geant4 and ROOT, characterized by detector of DarkSHINE experiment.
     - DAna: source codes of software framework for the DarkSHINE analysis and reconstruction tools.
"""
        )
        GUIDE_PROMPT: str = Field(
            default="""#### Task:

- Analyze the chat history to determine the necessity of using tools. Available Tools: Code Interpreter, Web Search, or Knowledge Search.

#### Guidelines:

- Analyze user's need and final goal according to the chat history
- Analyze what's the next step to do in order to achieve the user's need. You can decide wether to use tool, or simply response to user.
- Use only one type of tool at a time, and stop right away. Because you need to wait for the tool execution.
- Ensure that the tools are effectively utilized to achieve the highest-quality analysis for the user.
- If tool using returns no helping information, modify the usage of tool and use it again.
- All responses should be communicated in the chat's primary language, ensuring seamless understanding. If the chat is multilingual, default to English for clarity.
"""
        )

    def __init__(self):
        # Configs
        self.valves = self.Valves()
        self.data_prefix = "data:"
        self.max_loop = self.valves.MAX_LOOP  # Save money
        self.TOOL = {}
        self.prompt_templates = {}
        self.replace_tags = {}
        if self.valves.USE_CODE_INTERPRETER:
            self.TOOL["code_interpreter"] = self._code_interpreter
            self.prompt_templates["code_interpreter"] = (
                self.valves.CODE_INTERPRETER_PROMPT
            )
        if self.valves.USE_WEB_SEARCH:
            self.TOOL["web_search"] = self._web_search
            self.prompt_templates["web_search"] = self.valves.WEB_SEARCH_PROMPT
            self.replace_tags["web_search"] = "Searching"
        if self.valves.USE_KNOWLEDGE_SEARCH:
            self.TOOL["knowledge_search"] = self._knowledge_search
            self.prompt_templates["knowledge_search"] = (
                self.valves.KNOWLEDGE_SEARCH_PROMPT
            )
            self.replace_tags["knowledge_search"] = "Searching"
        # Global vars
        self.emitter = None
        self.total_response = ""
        self.temp_content = ""  # Temporary string to hold accumulated content
        self._init_knowledge()

    def _init_knowledge(self):
        log.debug(f"Getting first collection")

        first_id = VECTOR_DB_CLIENT.client.list_collections()[0]
        first_collection = VECTOR_DB_CLIENT.client.get_collection(name=first_id).get()
        metadata = first_collection["metadatas"]
        log.debug(f"Metadata: {metadata}")

        knowledge_bases = Knowledges.get_knowledge_bases()
        log.debug(f"Knowldge bases: {knowledge_bases}")

    def pipes(self):
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

            # 处理消息以防止连续的相同角色
            messages = payload["messages"]
            i = 0
            while i < len(messages) - 1:
                if messages[i]["role"] == messages[i + 1]["role"]:
                    # 插入具有替代角色的占位符消息
                    alternate_role = (
                        "assistant" if messages[i]["role"] == "user" else "user"
                    )
                    messages.insert(
                        i + 1,
                        {"role": alternate_role, "content": "[Unfinished thinking]"},
                    )
                i += 1

            self._set_system_prompt(messages)

            # yield json.dumps(payload, ensure_ascii=False)

            # 发起API请求
            do_pull = True
            count = 0
            while do_pull and count < self.max_loop:
                thinking_state = {"thinking": -1}  # 使用字典来存储thinking状态
                async with httpx.AsyncClient(http2=True) as client:
                    async with client.stream(
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
                            if choice.get("finish_reason"):
                                res = self._filter_response_tag()
                                yield res
                                # Clean up
                                if self.temp_content:
                                    yield self.temp_content
                                    self.temp_content = ""
                                self.total_response = self.total_response.lstrip("\n")
                                tools = self._find_tool_usage(self.total_response)
                                # if tool is not None:
                                user_proxy_reply = ""
                                if tools is not None:
                                    do_pull = True
                                    # Move total_response to messages
                                    messages.append(
                                        {
                                            "role": "assistant",
                                            "content": self.total_response,
                                        }
                                    )
                                    self.total_response = ""
                                    #yield "\n<details type=\"user_proxy\">\n<summary>Results</summary>\n"
                                    # Call tools
                                    for tool in tools:
                                        reply = await self.TOOL[tool["name"]](
                                            tool["attributes"], tool["content"]
                                        )
                                        user_proxy_reply += reply
                                        yield reply

                                    messages.append(
                                        {
                                            "role": "user",
                                            "content": user_proxy_reply,
                                        }
                                    )
                                    #yield "\n</details>\n"
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
                                    res = self._filter_response_tag(content)
                                    if res:
                                        yield res
                                else:
                                    yield content


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
        self.temp_content += content
        res = ""
        if "<" in self.temp_content:
            # Conver tool calling tags into content (Except code_interpreter, let openwebui to handle)
            if len(self.temp_content) > 20:
                if "<web_search" in self.temp_content or "<knowledge_search" in self.temp_content:
                    pattern = re.compile(
                        r"<(web_search|knowledge_search)\s*([^>]*)>(.*?)</\1>",
                        re.DOTALL,
                    )
                    # Find all matches in the self.temp_content
                    matches = pattern.findall(self.temp_content)
                    if matches:
                        match = matches[0]
                        tag_name = match[0]
                        attributes_str = match[1]
                        tag_content = match[2].strip()
                        summary = self.replace_tags[tag_name] + " " + attributes_str
                        res += f"\n<details type=\"{tag_name}\">\n<summary>{summary}</summary>\n{tag_content}\n</details>"
                        self.temp_content = re.sub(pattern, "", self.temp_content)
                        if self.temp_content:
                            res += self.temp_content
                            self.temp_content = ""
                else:
                    res += self.temp_content
                    self.temp_content = ""
        else:
            res += self.temp_content
            self.temp_content = ""
        return res

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

    async def _web_search(self, attributes: dict, content: str) -> str:
        # Extract the search query from the content
        search_query = content.strip()

        if not search_query:
            return f"\n<details type=\"user_proxy\">\n<summary>Error: No search query provided.</summary>\n</details>\n"

        url = attributes["url"]

        # Handle Google Custom Search
        if (
            url == "www.googleapis.com"
            and self.valves.GOOGLE_PSE_API_KEY
            and self.valves.GOOGLE_PSE_ENGINE_ID
        ):
            # Construct the Google search URL
            google_search_url = f"https://www.googleapis.com/customsearch/v1?q={search_query}&key={self.valves.GOOGLE_PSE_API_KEY}&cx={self.valves.GOOGLE_PSE_ENGINE_ID}&num=5"

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(google_search_url)
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
                            result = f"\n<details type=\"user_proxy\">\n<summary>Searched {len(urls)} sites</summary>\n"
                            result += "\n\n".join(search_results)
                            result += "\n</details>\n"
                            return result
                        else:
                            return f"\n<details type=\"user_proxy\">\n<summary>No results found on Google.</summary>\n</details>\n"
                    else:
                        return f"\n<details type=\"user_proxy\">\n<summary>Google search failed with status code: {response.status_code}</summary>\n</details>\n"
            except Exception as e:
                return f"\n<details type=\"user_proxy\">\n<summary>Error during Google search</summary>\n{str(e)}\n</details>\n"

        # Handle ArXiv search
        if url == "arxiv.org" and search_query:
            arxiv_search_url = f"http://export.arxiv.org/api/query?search_query=all:{search_query}&start=0&max_results=5"

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(arxiv_search_url)
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
                                arxiv_results.append("Error parsing ArXiv entry.")

                        if arxiv_results:
                            result = f"\n<details type=\"user_proxy\">\n<summary>Searched {len(urls)} papers</summary>\n"
                            result += "\n\n".join(arxiv_results)
                            result += "\n</details>\n"
                            return result
                        else:
                            return f"\n<details type=\"user_proxy\">\n<summary>No results found on ArXiv.</summary>\n</details>\n"
                    else:
                        return f"\n<details type=\"user_proxy\">\n<summary>ArXiv search failed with status code: {response.status_code}</summary>\n</details>\n"
            except Exception as e:
                return f"\n<details type=\"user_proxy\">\n<summary>Error during ArXiv search</summary>\n{str(e)}\n</details>\n"

        return f"\n<details type=\"user_proxy\">\n<summary>Invalid search source or query.</summary>\n</details>\n"

    async def _code_interpreter(self, attributes: dict, content: str) -> str:
        return "done"

    def _query_collection(
        self, collection_name: str, query_keywords: str, top_k: int = 3
    ) -> list:
        """
        Query the vector database by collection name and keywords, and return metadata and contexts.

        Args:
            collection_name (str): The name of the collection to query.
            query_keywords (str): The query keywords to search for.
            top_k (int): The number of top results to retrieve.

        Returns:
            list: A list of dictionaries containing metadata and context documents.
        """
        # Assuming VECTOR_DB_CLIENT.search interacts with the vector DB and retrieves the relevant documents
        log.debug(f"Generating Embeddings")
        embeddings = generate_embeddings(
            engine="openai",
            model="BAAI/bge-m3",
            text=query_keywords,
            url=self.valves.DEEPSEEK_API_BASE_URL,
            key=self.valves.DEEPSEEK_API_KEY,
            user=None,
        )
        log.debug("Searching VECTOR_DB_CLIENT")
        result = VECTOR_DB_CLIENT.search(
            collection_name=collection_name,
            vectors=[embeddings],
            limit=top_k,
        )
        if not result:
            return []
        ids = result.ids[0]
        metadatas = result.metadatas[0]
        documents = result.documents[0]

        # Package the results into a list of dictionaries
        results = []
        for i in range(len(ids)):
            results.append(
                {"id": ids[i], "metadata": metadatas[i], "document": documents[i]}
            )

        return results

    async def _knowledge_search(self, attributes: dict, content: str) -> str:
        """
        Retrieve relevant information from a knowledge collection
        based on the provided query, and return the results.

        Args:
            attributes (dict): Attributes containing information such as the knowledge base.
            content (str): The query to search for in the knowledge base.

        Returns:
            str: The retrieved relevant content or an error message.
        """
        collection = attributes.get("collection", "")
        if not collection:
            return "Error: No knowledge search collection specified."

        # Check if the knowledge base is available in the configured ones
        available_knowledge_cols = [
            name.strip() for name in self.valves.KNOWLEDGE_COLLECTIONS.split(",")
        ]
        if collection not in available_knowledge_cols:
            return f"Error: Collection '{collection}' is not available."

        # Retrieve relevant documents from the knowledge base
        try:
            results = self._query_collection(collection, content)
            if not results:
                return (
                    f"No relevant information found in the collection '{collection}'."
                )

            # Format the results for output
            formatted_results = []
            for result in results:
                metadata = result.get("metadata", {})
                document = result.get("document", "")
                formatted_results.append(
                    f"**Metadata**: {metadata}\n**Document**:\n```\n{document}\n```"
                )

            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Error during Knowledge Search processing: {str(e)}"

    def _find_tool_usage(self, content):
        tools = []
        # Define the regex pattern to match the XML tags
        pattern = re.compile(
            r"<(code_interpreter|web_search|knowledge_search)\s*([^>]*)>(.*?)</\1>",
            re.DOTALL,
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
        template_string = """#### Available Tools"""
        for i, (name, prompt) in enumerate(self.prompt_templates.items()):
            template_string += f"{i+1}. {prompt}\n"
        template_string += self.valves.GUIDE_PROMPT
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

