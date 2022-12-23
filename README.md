# visinger-speech

基于visinger的tts模型

（暂时还在开发调试中）
音频sample见[samples](/samples)
## 相比于原版vits
+ 删除了 Monotonoic Alignment， 使用MFA对齐后输入时长
+ 添加了音素级 F0Predictor
+ 添加了FramePriorNetwork
+ 使用飞桨paddlespeech作为中文文本前端，实现更可靠的文本正则化以及G2P
## 参考
+ [Period VITS](https://arxiv.org/pdf/2210.15964.pdf) 
+ [VISinger](https://github.com/So-Fann/VISinger) 
+ [FastSpeech2](https://github.com/ming024/FastSpeech2)
