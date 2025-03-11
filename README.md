# WELT: Workflow-adaptivE LLM with Target-oriented system

Maybe can do simple high energy physics analysis

![image-20250311102442512](assets/image-20250311102442512.png)

Video of DarkSHINE cutflow analysis: [link](https://ihepbox.ihep.ac.cn/ihepbox/index.php/s/9L6xfkszjb3JEd8)

`welt.py` is a [pipe function](https://docs.openwebui.com/features/plugin/functions/pipe/) of open-webui

Example code worker for `Code Interface`: [git repo](https://code.ihep.ac.cn/xuliang/drsai-code-worker-v2)

Or just replace the implementation of `_code_interface` by your own function

And to use WELT, you need to small monkey-patch open-webui like this: [commit](https://github.com/open-webui/open-webui/commit/bc79483b7bc6e85853eefef986afea7192041501)

