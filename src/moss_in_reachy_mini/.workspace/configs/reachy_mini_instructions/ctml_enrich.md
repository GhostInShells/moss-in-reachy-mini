# 严格遵守
务必保证生成的ctml闭标签准确性，例子如下
- `<memory:refresh_summary_memory>>xxx</<refresh_summary_memory>`，`<refresh_summary_memory>`闭标签是错误的，因为没有带上路径
- `<memory:refresh_summary_memory>>xxx</<memory:refresh_summary_memory>`是正确的