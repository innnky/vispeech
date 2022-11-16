# visinger-speech

基于visinger的tts模型

（暂时还在开发调试中）（暂时效果不佳）
## 相比于原版vits
+ 删除了 Monotonoic Alignment， 使用MFA对齐后输入时长
+ 添加了F0Predictor
+ 添加了FramePriorNetwork
## 参考
+ [Period VITS](https://arxiv.org/pdf/2210.15964.pdf) 
+ [VISinger](https://github.com/So-Fann/VISinger) 
