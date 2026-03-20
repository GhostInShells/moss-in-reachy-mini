# 关于give_cues的CTML说明

说明：give_cues是给MainAgent提供建议的工具

## give_cues CTML格式说明
```
✅ <give_cues>内容...</give_cues>
❌ <give_cues><text__>内容...</text__></give_cues>
❌ <give_cues text__="内容..." />
```

## give_cues 内容要求

### 把自己的思考结果作为建议传递是**坚决制止**的
```
❌ <give_cues>当前交互流程顺畅，主Agent已妥善回应用户需求，无需额外补充建议。</give_cues>
```

### 有gui通道的前提下，**输出自己思考的内容到gui**
```
✅ <gui:append_markdown>当前交互流程顺畅，主Agent已妥善回应用户需求，无需额外补充建议。</gui:append_markdown>
```

### 没有gui通道的前提下，**直接输出自己的思考结果**
```
✅ 当前交互流程顺畅，主Agent已妥善回应用户需求，无需额外补充建议。
```

