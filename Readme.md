## 数据下载
数据文件较大不便放在github，所以从njubox链接下载: 
https://box.nju.edu.cn/d/7760b6e46f7a4a83a18e/ 
\n下载/0521_new_format和/wiki_zh两个文件夹放在根目录即可
## skipgram模型训练结束了
文件稍微有点乱。func4skipgram.py和genCorpus.ipynb是生成skipgram_corpus文件夹内的清洗后的文本和词典的文件，
这部分工作已经完成，不再需要动这两个地方的代码了。
训练skipgram的代码集成在了skipgram_model.py里，训练时运行就好。
模型的stat_dicts被存储在了models文件夹内，每一个epoch都保存了，方便读取使用和继续训练。
查看模型效果是testSkipgram.ipynb，暂时写的不够清楚，直接看第三个吧，前两个测试效果的方法都还在调试，不一定好。