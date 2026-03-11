# CTML指令强化

## 闭合标签的路径必须与开标签的路径一致 
### Case 1
- 错误：`<memory:refresh_summary_memory>>xxx</refresh_summary_memory>`
- 正确：`<memory:refresh_summary_memory>>xxx</memory:refresh_summary_memory>`

### Case 2
**这个错误你犯的有点多，请务必注意此处使用正确的闭合标签路径**
- 错误：`<douyin_live:give_cues>...</give_cues>`
- 正确：`<douyin_live:give_cues>...</douyin_live:give_cues>`
