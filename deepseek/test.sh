curl -X POST "https://api.deepseek.com/v1/moderations" \
     -H "Authorization: Bearer sk-3a72a28632ef44e49f7eaa4c51eb6318" \
     -H "X-DeepSeek-Version: 2024-03" \
     -H "Content-Type: application/json" \
     -d '{
           "input": "测试内容",
           "model": "deepseek-moderation-v1",
           "policy": "fast"
         }'