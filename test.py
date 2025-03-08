import requests
import httpx
import asyncio

client=httpx.AsyncClient(http2=True, proxy="http://127.0.0.1:7890")

async def _generate_vl_response(
    prompt: str,
    image_url: str,
    model: str,
    url: str = "https://api.openai.com/v1",
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
        # Construct the API request
        response = await client.post(
            f"{url}/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
        )

        # Check for valid response
        response.raise_for_status()

        # Parse and return embeddings if available
        data = response.json()
        return data["choices"][0]["message"]["content"]

    except httpx.HTTPStatusError as e:
        print(
            f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
        )
    #except Exception as e:
    #    print(f"An error occurred while generating vision model response: {str(e)}")

    return ""

async def main():
    response = await _generate_vl_response(
                    prompt="这是什么",
                    image_url="https://bkimg.cdn.bcebos.com/pic/77094b36acaf2edda3cc60e5cf4616e93901213f353b?x-bce-process=image/format,f_auto/watermark,image_d2F0ZXIvYmFpa2UyNzI,g_7,xp_5,yp_5,P_20/resize,m_lfit,limit_1,h_1080",
                    model="Qwen/Qwen2-VL-72B-Instruct",
                    url="https://api.siliconflow.cn/v1",
                    key="sk-hnbbhmxhshaltasitzdinlxtffngpuhouhgjsnikfxkvjejh"
                )
    
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
