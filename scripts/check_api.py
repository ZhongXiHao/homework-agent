import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 直接用 OpenAI SDK 检查 API 配置是否可用
client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))
response = client.chat.completions.create(
    model=os.getenv("LLM_MODEL_ID"),
    messages=[{"role": "user", "content": "你好"}],
    reasoning_effort="low"
)
print(response)

'''
ChatCompletion(id='20261108003804f8933787e7274451',
    choices=[Choice(finish_reason='stop',
    index=0,
    logprobs=None,
    message=ChatCompletionMessage(content='你好！很高兴见到你。我是GLM，Z.ai训练的大语言模型。\n\n今天我能为你做些什么？无论是回答问题、提供信息，还是进行日常交流，我都很乐意帮助你。',
    refusal=None,
    role='assistant',
    annotations=None,
    audio=None,
    function_call=None,
    tool_calls=None,
    reasoning_content='收到用户发来的"你好"这样简单的问候，需要礼 貌地回应并引导对话。首先，我应该以热情友好的态度回应，让用户感受到被重视。同时，考虑到这是初次互动，可以主动询问用户需要什么帮助，以展示乐于助人的特质。在语言表达上，要使用自然流畅的中文，避免过于机械化，可以加入一些表情符号增加亲和力。最后，保持开放式的提问方式，鼓励用户进一步表达需求，这样能为后续对话奠定良好基础'))],
    created=1794069487,
    model='glm-5.3-flash',
    object='chat.completion',
    metadata=None,
    moderation=None,
    service_tier=None,
    system_fingerprint=None,
    usage=CompletionUsage(completion_tokens=145,
    prompt_tokens=13,
    total_tokens=158,
    completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=None,
    audio_tokens=None,
    reasoning_tokens=103,
    rejected_prediction_tokens=None,
    text_tokens=None),
    prompt_tokens_details=PromptTokensDetails(audio_tokens=None,
    cache_write_tokens=None,
    cached_tokens=0,
    image_tokens=None,
    text_tokens=None)),
    request_id='20261108003804f8933787e7274451')
'''
