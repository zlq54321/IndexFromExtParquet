# GraphRAG OpenAI 兼容查询示例

本项目提供了一个将 GraphRAG 查询（local/global search）包装为 OpenAI Chat Completions 兼容接口的服务脚本：

```bash
uv run query_openai_api.py --root /root/rag_case_quality --method local --port 8001
```

启动后，服务地址为：

- `http://localhost:8001/v1/chat/completions`

下面是一个使用 `curl` 的测试请求示例。

```bash
curl -X POST "http://localhost:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "graphrag-local",
    "messages": [
      {
        "role": "user",
        "content": "请根据本地 GraphRAG 知识库介绍一下这个项目当前索引的数据情况。"
      }
    ],
    "stream": false
  }'
```

说明：

- `model` 字段目前仅作透传标记，不影响实际调用的 GraphRAG 配置；
- `messages` 中最后一条 `role == "user"` 的 `content` 会作为 GraphRAG 的查询问题；
- 服务内部会根据启动参数的 `--method`：
  - `--method local`：调用 `graphrag.api.local_search`；
  - `--method global`：调用 `graphrag.api.global_search`。

