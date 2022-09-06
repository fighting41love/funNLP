<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./data/.logo图片/.img.jpg"width="180">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">NLP民工的乐园</div>
</center>
<br><br><br><br>

[![](https://img.shields.io/badge/dynamic/json?color=blue&label=%E7%9F%A5%E4%B9%8E%E5%85%B3%E6%B3%A8&query=%24.data.totalSubs&url=https%3A%2F%2Fapi.spencerwoo.com%2Fsubstats%2F%3Fsource%3Dzhihu%26queryKey%3Dmountain-blue-64)](https://www.zhihu.com/people/mountain-blue-64)
[![](https://img.shields.io/badge/dynamic/json?color=blueviolet&label=github%20followers&query=%24.data.totalSubs&url=https%3A%2F%2Fapi.spencerwoo.com%2Fsubstats%2F%3Fsource%3Dgithub%26queryKey%3Dfighting41love)](https://github.com/fighting41love)
[![](data/.logo图片/.捐赠图片/.PaperCitations-467-red.svg)](https://scholar.google.com/citations?hl=en&user=aqZdfDUAAAAJ)

### The Most Powerful NLP-Weapon Arsenal

## NLP民工的乐园: 几乎最全的中文NLP资源库
在入门到熟悉NLP的过程中，用到了很多github上的包，遂整理了一下，分享在这里。

很多包非常有趣，值得收藏，满足大家的收集癖！
如果觉得有用，请分享并star:star:，谢谢！

长期不定时更新，欢迎watch和fork！:heart::heart::heart::heart::heart:

目录（Table of contents）
=================
<table border="0">
 <tr>
    <td><b style="font-size:30px">:star:</b></td>
    <td><b style="font-size:30px">:star::star:</b></td>
    <td><b style="font-size:30px">:star::star::star:</b></td>
    <td><b style="font-size:30px">:star::star::star::star:</b></td>
 </tr>
 <tr>
    <td>

<!--ts-->
   * [语料库](#语料库)
   * [词库及词法工具](#词库及词法工具)
   * [预训练语言模型](#预训练语言模型)
   * [抽取](#抽取)
   * [知识图谱](#知识图谱)
   * [文本生成](#文本生成)
   * [文本摘要](#文本摘要)
   * [智能问答](#智能问答)
   * [文本纠错](#文本纠错)


<!--te-->

  </td>

  <td>

<!--ts-->

   * [文档处理](#文档处理)
   * [表格处理](#表格处理)
   * [文本匹配](#文本匹配)
   * [文本数据增强](#文本数据增强)
   * [文本检索](#文本检索)
   * [阅读理解](#阅读理解)
   * [情感分析](#情感分析)
   * [常用正则表达式](#常用正则表达式)
   * [语音处理](#语音处理)
<!--te-->

  </td>

  <td>
   
<!--ts-->
   * [常用正则表达式](#常用正则表达式)
   * [事件抽取](#事件抽取)
   * [机器翻译](#机器翻译)
   * [数字转换](#数字转换)
   * [指代消解](#指代消解)
   * [文本聚类](#文本聚类)
   * [文本分类](#文本分类)
   * [知识推理](#知识推理)
   * [可解释NLP](#可解释自然语言处理)
   * [文本对抗攻击](#文本对抗攻击)

<!--te-->
    
  </td>

  <td>
   
<!--ts-->

   * [文本可视化](#文本可视化)
   * [文本标注工具](#文本标注工具)
   * [综合工具](#综合工具)
   * [有趣搞笑工具](#有趣搞笑工具)
   * [课程报告面试等](#课程报告面试等)
   * [比赛](#比赛)
   * [金融NLP](#金融自然语言处理)
   * [医疗NLP](#医疗自然语言处理)
   * [法律NLP](#法律自然语言处理)
   * [其他](#其他)

<!--te-->
    
  </td>

 </tr>
</table>




----

# 语料库

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :----   |          :--- |
|   人名语料库    |        |  [wainshine/Chinese-Names-Corpus](https://github.com/wainshine/Chinese-Names-Corpus)  |
|   Chinese-Word-Vectors    |  各种中文词向量      |  [github repo](https://github.com/Embedding/Chinese-Word-Vectors)  |
|    中文聊天语料    |   该库搜集了包含豆瓣多轮, PTT八卦语料, 青云语料, 电视剧对白语料, 贴吧论坛回帖语料,微博语料,小黄鸡语料     |  [link](https://github.com/codemayq/chaotbot_corpus_Chinese)  |
|    中文谣言数据    |     该数据文件中，每一行为一条json格式的谣言数据   |   [github](https://github.com/thunlp/Chinese_Rumor_Dataset)  |
|     中文问答数据集   |        |  [链接](https://panbaiducom/s/1QUsKcFWZ7Tg1dk_AbldZ1A) 提取码 2dva  |
|    微信公众号语料   |   3G语料，包含部分网络抓取的微信公众号的文章，已经去除HTML，只包含了纯文本。每行一篇，是JSON格式，name是微信公众号名字，account是微信公众号ID，title是题目，content是正文     | [github](https://github.com/nonamestreet/weixin_public_corpus)    |
|    中文自然语言处理 语料、数据集   |        |  [github](https://github.com/SophonPlus/ChineseNlpCorpus)  |
|    任务型对话英文数据集    |     【最全任务型对话数据集】主要介绍了一份任务型对话数据集大全，这份数据集大全涵盖了到目前在任务型对话领域的所有常用数据集的主要信息。此外，为了帮助研究者更好的把握领域进展的脉络，我们以Leaderboard的形式给出了几个数据集上的State-of-the-art实验结果。   |   [github](https://github.com/AtmaHou/Task-Oriented-Dialogue-Dataset-Survey)     |
|    语音识别语料生成工具    |    从具有音频/字幕的在线视频创建自动语音识别(ASR)语料库    |   [github](https://github.com/yc9701/pansori)  |
|     LitBankNLP数据集   |   支持自然语言处理和计算人文学科任务的100部带标记英文小说语料     |   [github](https://github.com/dbamman/litbank) |
|    中文ULMFiT  |    情感分析 文本分类 语料及模型    |   [github](https://github.com/bigboNed3/chinese_ulmfit)  |
|    省市区镇行政区划数据带拼音标注    |        |   [github](https://github.com/xiangyuecn/AreaCity-JsSpider-StatsGov)  |
|    教育行业新闻 自动文摘 语料库    |        |   [github](https://github.com/wonderfulsuccess/chinese_abstractive_corpus)  |
|    中文自然语言处理数据集    |        |  [github](https://github.com/InsaneLife/ChineseNLPCorpus)   |
|     百度知道问答语料库   |   超过580万的问题，938万的答案，5800个分类标签。基于该问答语料库，可支持多种应用，如闲聊问答，逻辑挖掘     | [github](https://github.com/liuhuanyong/MiningZhiDaoQACorpus)  |
|     维基大规模平行文本语料   |  85种语言、1620种语言对、135M对照句  |  [github](https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix) 
|   古诗词库     |        |  [github repo](https://github.com/panhaiqi/AncientPoetry) <br>[更全的古诗词库](https://github.com/chinese-poetry/chinese-poetry)
|   低内存加载维基百科数据    |     用新版nlp库加载17GB+英文维基语料只占用9MB内存遍历速度2-3 Gbit/s    |   [github](https://gistgithub.com/thomwolf/13ca2b2b172b2d17ac66685aa2eeba62) |
|    对联数据    |   700,000 couplets, 超过70万对对联     |   [github](https://github.com/wb14123/couplet-dataset)  |
|   《配色辞典》数据集     |        |  [github](https://github.com/mattdesl/dictionary-of-colour-combinations)   |
|    42GB的JD客服对话数据(CSDD)    |        |   [github](https://github.com/jd-aig/nlp_baai/tree/master/pretrained_models_and_embeddings)  |
|  70万对联数据       |        | [link](https://github.com/wb14123/couplet-dataset)   |
|   用户名黑名单列表    |        |   [github](https://github.com/marteinn/The-Big-Username-Blacklist)  |
|     依存句法分析语料   |    4万句高质量标注数据    |  [Homepage](http//hlt.suda.edu.cn/indexphp/Nlpcc-2019-shared-task)  |
|      人民日报语料处理工具集  |        |  [github](https://github.com/howl-anderson/tools_for_corpus_of_people_daily)   |
|  虚假新闻数据集 fake news corpus      |        |   [github](https://github.com/several27/FakeNewsCorpus)  |
|    诗歌质量评价/细粒度情感诗歌语料库    |        |  [github](https://github.com/THUNLP-AIPoet/Datasets)   |
|    中文自然语言处理相关的开放任务    |  数据集以及当前最佳结果     |    [github](https://github.com/didi/ChineseNLP) |
|    中文缩写数据集    |        |   [github](https://github.com/zhangyics/Chinese-abbreviation-dataset)  |
|    中文任务基准测评     |    代表性的数据集-基准(预训练)模型-语料库-baseline-工具包-排行榜    |   [github](https://github.com/CLUEbenchmark/CLUE)  |
|   中文谣言数据库    |        |  [github](https://github.com/thunlp/Chinese_Rumor_Dataset)   |
|     CLUEDatasetSearch    |   中英文NLP数据集搜索所有中文NLP数据集，附常用英文NLP数据集     |   [github](https://github.com/CLUEbenchmark/CLUEDatasetSearch)  |
|    多文档摘要数据集    |        |  [github](https://github.com/complementizer/wcep-mds-dataset)   |
|    让人人都变得“彬彬有礼”礼貌迁移任务   |  在保留意义的同时将非礼貌语句转换为礼貌语句，提供包含139M + 实例的数据集       |   [paper and code](https://arxiv.org/abs/200414257)  |
|    粤语/英语会话双语语料库    |        |   [github](https://github.com/khiajohnson/SpiCE-Corpus)  |
|     中文NLP数据集列表   |        |   [github](https://github.com/OYE93/Chinese-NLP-Corpus)  |
|   类人名/地名/组织机构名的命名体识别数据集     |        |  [github](https://github.com/LG-1/video_music_book_datasets)  |
|    中文语言理解测评基准    |    包括代表性的数据集&基准模型&语料库&排行榜   |   [github](https://github.com/brightmart/ChineseGLUE)  |
|    OpenCLaP多领域开源中文预训练语言模型仓库    |   民事文书、刑事文书、百度百科 |    [github](https://github.com/thunlp/OpenCLaP) |
|   中文全词覆盖BERT及两份阅读理解数据     |      DRCD数据集：由中国台湾台达研究院发布，其形式与SQuAD相同，是基于繁体中文的抽取式阅读理解数据集。<br>CMRC 2018数据集:哈工大讯飞联合实验室发布的中文机器阅读理解数据。根据给定问题，系统需要从篇章中抽取出片段作为答案，形式与SQuAD相同。|    [github](https://github.com/ymcui/Chinese-BERT-wwm) |
|  Dakshina数据集     |    十二种南亚语言的拉丁/本地文字平行数据集合     |   [github](https://github.com/google-research-datasets/dakshina)  |
|    OPUS-100    |   以英文为中心的多语(100种)平行语料     |   [github](https://github.com/EdinburghNLP/opus-100-corpus)  |
|      中文阅读理解数据集  |        |   [github](https://github.com/ymcui/Chinese-RC-Datasets)  |
|    中文自然语言处理向量合集    |        |   [github](https://github.com/liuhuanyong/ChineseEmbedding)  |
|    中文语言理解测评基准    |包括代表性的数据集、基准(预训练)模型、语料库、排行榜       |  [github](https://github.com/CLUEbenchmark/CLUE)   |
|  NLP数据集/基准任务大列表     |        |  [github](https://quantumstatcom/dataset/datasethtml)   |
|   LitBankNLP数据集     |   支持自然语言处理和计算人文学科任务的100部带标记英文小说语料     | [github](https://github.com/dbamman/litbank)    |
|70万对联数据||[github](https://github.com/wb14123/couplet-dataset)|
|文言文（古文）-现代文平行语料|短篇章中包括了《论语》、《孟子》、《左传》等篇幅较短的古籍，已和《资治通鉴》合并|[github](https://github.com/NiuTrans/Classical-Modern)|
|COLDDateset，中文冒犯性语言检测数据集|涵盖了种族、性别和地区等话题内容，数据待论文发表后放出|[paper](https://arxiv.org/pdf/2201.06025.pdf)|

# 词库及词法工具

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|  textfilter     |    中英文敏感词过滤    |  [observerss/textfilter](https://github.com/observerss/textfilter)  |
|   人名抽取功能    |   中文（现代、古代）名字、日文名字、中文的姓和名、称呼（大姨妈、小姨妈等）、英文->中文名字（李约翰）、成语词典   |  [cocoNLP](https://github.com/fighting41love/cocoNLP)  |
|   中文缩写库    | 全国人大: 全国 人民 代表大会; 中国: 中华人民共和国;女网赛: 女子/n 网球/n 比赛/vn  |  [github](https://github.com/zhangyics/Chinese-abbreviation-dataset/blob/master/dev_set.txt)  |
|   汉语拆字词典    |  漢字	拆法 (一)	拆法 (二)	拆法 (三) 拆	手 斥	扌 斥	才 斥    |  [kfcd/chaizi](https://github.com/kfcd/chaizi)  |
|    词汇情感值   |    山泉水:0.400704566541 <br>  充沛:	0.37006739587   |   [rainarch/SentiBridge](https://github.com/rainarch/SentiBridge/blob/master/Entity_Emotion_Express/CCF_data/pair_mine_result) |
|   中文词库、停用词、敏感词    |        |  [dongxiexidian/Chinese](https://github.com/fighting41love/Chinese_from_dongxiexidian)  |
|   python-pinyin    |   汉字转拼音     |  [mozillazg/python-pinyin](https://github.com/mozillazg/python-pinyin)  |
|   zhtools   |   中文繁简体互转     |  [skydark/nstools](https://github.com/skydark/nstools/tree/master/zhtools)  |
|   英文模拟中文发音引擎    |    say wo i ni #说：我爱你    |  [tinyfool/ChineseWithEnglish](https://github.com/tinyfool/ChineseWithEnglish)  |
|  chinese_dictionary     |    同义词库、反义词库、否定词库    |  [guotong1988/chinese_dictionary](https://github.com/guotong1988/chinese_dictionary)  |
|   wordninja    |   无空格英文串分割、抽取单词     | [wordninja](https://github.com/keredson/wordninja)  |
|   汽车品牌、汽车零件相关词汇    |        |  [data](https://github.com/fighting41love/funNLP/tree/master/data)|     公司名字大全   |        |   [github repo](https://github.com/wainshine/Company-Names-Corpus)
|   THU整理的词库   | IT词库、财经词库、成语词库、地名词库、历史名人词库、诗词词库、医学词库、饮食词库、法律词库、汽车词库、动物词库    | [link](http://thuctc.thunlp.org/)   |
|   罪名法务名词及分类模型    |    包含856项罪名知识图谱, 基于280万罪名训练库的罪名预测,基于20W法务问答对的13类问题分类与法律资讯问答功能    |    [github](https://github.com/liuhuanyong/CrimeKgAssitant)     |
|   分词语料库+代码    |        |  [百度网盘链接](https://panbaiducom/s/1MXZONaLgeaw0_TxZZDAIYQ)     - 提取码 pea6  |
|  基于Bi-LSTM + CRF的中文分词+词性标注     |   keras实现     |  [link](https://github.com/GlassyWing/bi-lstm-crf)  |
| 基于Universal Transformer + CRF 的中文分词和词性标注    |        |  [link](https://github.com/GlassyWing/transformer-word-segmenter)  |
| 快速神经网络分词包     |    java version     |   [](https://github.com/yaoguangluo/NeroParser) |
|   chinese-xinhua      |    中华新华字典数据库及api，包括常用歇后语、成语、词语和汉字    |   [github](https://github.com/pwxcoo/chinese-xinhua)  |
|   SpaCy 中文模型     |   包含Parser, NER, 语法树等功能。有一些英文package使用spacy的英文模型的，如果要适配中文，可能需要使用spacy中文模型。     |   [github](https://github.com/howl-anderson/Chinese_models_for_SpaCy)    |
|    中文字符数据    |        |  [github](https://github.com/skishore/makemeahanzi)   |
|    Synonyms中文近义词工具包    |        |   [github](https://github.com/huyingxi/Synonyms)  |
|   HarvestText     |   领域自适应文本挖掘工具（新词发现-情感分析-实体链接等）     |   [github](https://github.com/blmoistawinde/HarvestText)   |
|    word2word    |    方便易用的多语言词-词对集62种语言/3,564个多语言对    |   [github](https://github.com/Kyubyong/word2word)  |
|   多音字词典数据及代码     |        |  [github](https://github.com/mozillazg/phrase-pinyin-data)   |
|    汉字、词语、成语查询接口    |        |   [github](https://github.com/netnr/zidian/tree/206028e5ce9a608afc583820df8dc2d1d4b61781)  |
|    103976个英语单词库包    |    （sql版，csv版，Excel版）    |  [github](https://github.com/1eez/103976)   |
|    英文脏话大列表    |        |   [github](https://github.com/zacanger/profane-words)  |
|      词语拼音数据  |        |   [github](https://github.com/mozillazg/phrase-pinyin-data)  |
|   186种语言的数字叫法库     |        |   [github](https://github.com/google/UniNum)  |
|    世界各国大规模人名库    |        |   [github](https://github.com/philipperemy/name-dataset)  |
|   汉字字符特征提取器 (featurizer)     |   提取汉字的特征（发音特征、字形特征）用做深度学习的特征     |   [github](https://github.com/howl-anderson/hanzi_char_featurizer)  |
|     char_featurizer - 汉字字符特征提取工具   |        |    [github](https://github.com/charlesXu86/char_featurizer) |
|   中日韩分词库mecab的Python接口库     |        |   [github](https://github.com/jeongukjae/python-mecab)  |
|    g2pC基于上下文的汉语读音自动标记模块    |        |   [github](https://github.com/Kyubyong/g2pC)  |
|     ssc, Sound Shape Code   | 音形码 - 基于“音形码”的中文字符串相似度计算方法      | [version 1](https://github.com/qingyujean/ssc)<br>[version 2](https://github.com/wenyangchou/SimilarCharactor)<br>[blog/introduction](https://blogcsdnnet/chndata/article/details/41114771)   |
|    基于百科知识库的中文词语多词义/义项获取与特定句子词语语义消歧    |        |    [github](https://github.com/liuhuanyong/WordMultiSenseDisambiguation) |
|   Tokenizer快速、可定制的文本词条化库     |        |   [github](https://github.com/OpenNMT/Tokenizer)  |
|   Tokenizers     |  注重性能与多功能性的最先进分词器      |    [github](https://github.com/huggingface/tokenizers)|
|    通过同义词替换实现文本“变脸”    |        |    [github](https://github.com/paubric/python-sirajnet) |
|    token2index与PyTorch/Tensorflow兼容的强大轻量词条索引库    |        |  [github](https://github.com/Kaleidophon/token2index)   |
|    繁简体转换    |        |   [github](https://github.com/berniey/hanziconv)  |
| 粤语NLP工具|       |   [github](https://github.com/jacksonllee/pycantonese)|
|领域词典库|涵盖68个领域、共计916万词的专业词典知识库|[github](github.com/liuhuanyong/DomainWordsDict)|



# 预训练语言模型&大模型
   
| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|BMList|大模型大列表|[github](https://github.com/OpenBMB/BMList)|
| bert论文中文翻译     |        | [link](https://github.com/yuanxiaosc/BERT_Paper_Chinese_Translation)   |
|    bert原作者的slides  |    |  [link](https://pan.baidu.com/s/1OSPsIu2oh1iJ-bcXoDZpJQ)  |
| 文本分类实践     |        |  [github](https://github.com/NLPScott/bert-Chinese-classification-task)  |
|  bert tutorial文本分类教程     |        | [github](https://github.com/Socialbird-AILab/BERT-Classification-Tutorial) |
| bert pytorch实现       |        |  [github](https://github.com/huggingface/pytorch-pretrained-BERT)  |
|   bert pytorch实现      |        |  [github](https://github.com/huggingface/pytorch-pretrained-BERT)  |
|  BERT生成句向量，BERT做文本分类、文本相似度计算     |        | [github](https://github.com/terrifyzhao/bert-utils)   |
|  bert、ELMO的图解     |        |  [github](https://jalammargithubio/illustrated-bert/)  |
|  BERT Pre-trained models and downstream applications     |        |  [github](https://github.com/asyml/texar/tree/master/examples/bert)  |
|  语言/知识表示工具BERT & ERNIE      |        |   [github](https://github.com/PaddlePaddle/LARK)  |
|    Kashgari中使用gpt-2语言模型    |        |  [github](https://github.com/BrikerMan/Kashgari)   |
|     Facebook LAMA   |    用于分析预训练语言模型中包含的事实和常识知识的探针。语言模型分析，提供Transformer-XL/BERT/ELMo/GPT预训练语言模型的统一访问接口    |  [github](https://github.com/facebookresearch/LAMA)   |
|    中文的GPT2训练代码    |        |    [github](https://github.com/Morizeyao/GPT2-Chinese) |
|   XLMFacebook的跨语言预训练语言模型     |        |   [github](https://github.com/facebookresearch/XLM)  |
|    海量中文预训练ALBERT模型    |        |  [github](https://github.com/brightmart/albert_zh)   |
|    Transformers 20    |    支持TensorFlow 20 和 PyTorch 的自然语言处理预训练语言模型(BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) 8种架构/33种预训练模型/102种语言   | [github](https://github.com/huggingface/transformers)    |
|    8篇论文梳理BERT相关模型进展与反思    |        |    [github](https://wwwmsracn/zh-cn/news/features/bert) |
|    法文RoBERTa预训练语言模型    |    用138GB语料训练的法文RoBERTa预训练语言模型    |   [link](https://camembert-model.fr/)  |
|     中文预训练 ELECTREA 模型    |    基于对抗学习 pretrain Chinese Model    |   [github](https://github.com/CLUEbenchmark/ELECTRA)  |
|   albert-chinese-ner     |   用预训练语言模型ALBERT做中文NER    |   [github](https://github.com/ProHiryu/albert-chinese-ner)  |
|    开源预训练语言模型合集    |        |  [github](https://github.com/ZhuiyiTechnology/pretrained-models)   |
|   中文ELECTRA预训练模型     |        |  [github](https://github.com/ymcui/Chinese-ELECTRA)   |
|    用Transformers(BERT, XLNet, Bart, Electra, Roberta, XLM-Roberta)预测下一个词(模型比较)    |        |  [github](https://github.com/renatoviolin/next_word_prediction)   |
|   TensorFlow Hub     |    40+种语言的新语言模型(包括中文)    |  [link](https://tfhub.dev/google/collections/wiki40b-lm/1)   |
|   UER     | 基于不同语料、编码器、目标任务的中文预训练模型仓库（包括BERT、GPT、ELMO等）       |  [github](https://github.com/dbiir/UER-py)    |
|    开源预训练语言模型合集    |        |  [github](https://github.com/ZhuiyiTechnology/pretrained-models)   |
|   多语言句向量包     |        |  [github](https://github.com/yannvgn/laserembeddings)   |
|Language Model as a Service (LMaaS)|语言模型即服务|[github](https://github.com/txsun1997/LMaaS-Papers)|
|开源语言模型GPT-NeoX-20B|200亿参数，是目前最大的可公开访问的预训练通用自回归语言模型|[github](https://github.com/EleutherAI/gpt-neox)|

# 抽取

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   时间抽取   |    已集成到 python package [cocoNLP](https://github.com/fighting41love/cocoNLP)中，欢迎试用    | [java version]( https://github.com/shinyke/Time-NLP)<br>[python version](https://github.com/zhanzecheng/Time_NLP) |
|    神经网络关系抽取 pytorch    |    暂不支持中文     |    [github](https://github.com/ShulinCao/OpenNRE-PyTorch)    |
|    基于bert的命名实体识别 pytorch    |  暂不支持中文      |   [github](https://github.com/Kyubyong/bert_ner)     |
|   关键词(Keyphrase)抽取包 pke     |        |   [github](https://github.com/boudinfl/pke)     |
|    BLINK最先进的实体链接库    |        |   [github](https://github.com/facebookresearch/BLINK)  |
|   BERT/CRF实现的命名实体识别     |        |   [github](https://github.com/Louis-udm/NER-BERT-CRF)  |
|    支持批并行的LatticeLSTM中文命名实体识别    |        |  [github](https://github.com/LeeSureman/Batch_Parallel_LatticeLSTM)   |
|    构建医疗实体识别的模型 |   包含词典和语料标注，基于python    |  [github](https://github.com/yixiu00001/LSTM-CRF-medical)   |
|    基于TensorFlow和BERT的管道式实体及关系抽取    |       - Entity and Relation Extraction Based on TensorFlow and BERT 基于TensorFlow和BERT的管道式实体及关系抽取，2019语言与智能技术竞赛信息抽取任务解决方案。Schema based Knowledge Extraction, SKE 2019   |   [github](https://github.com/yuanxiaosc/Entity-Relation-Extraction)  |
| 中文命名实体识别NeuroNER vs BertNER       |        |    [github](https://github.com/EOA-AILab/NER-Chinese) |
|  基于BERT的中文命名实体识别      |        |  [github](https://github.com/lonePatient/BERT-NER-Pytorch)   |
|    中文关键短语抽取工具    |        |  [github](https://github.com/dongrixinyu/chinese_keyphrase_extractor)   |
| bert      |     用于中文命名实体识别 tensorflow版本   |   [github](https://github.com/macanv/BERT-BiLSTM-CRF-NER)  |
|   bert-Kashgari     |    基于 keras 的封装分类标注框架 Kashgari，几分钟即可搭建一个分类或者序列标注模型     |  [github](https://github.com/BrikerMan/Kashgari)  |
|    cocoNLP    |  人名、地址、邮箱、手机号、手机归属地 等信息的抽取，rake短语抽取算法。  |   [github](https://github.com/fighting41love/cocoNLP)|
|    Microsoft多语言数字/单位/如日期时间识别包    |        |  [github](https://github.com/Microsoft/Recognizers-Text)   |
| 百度开源的基准信息抽取系统       |        |   [github](https://github.com/baidu/information-extraction)  |
|    中文地址分词（地址元素识别与抽取），通过序列标注进行NER    |        |  [github](https://github.com/yihenglu/chinese-address-segment)   |
|    基于依存句法的开放域文本知识三元组抽取和知识库构建    |        |   [github](https://github.com/lemonhu/open-entity-relation-extraction)  |
|   基于预训练模型的中文关键词抽取方法     |        |   [github](https://github.com/sunyilgdx/SIFRank_zh)  |
|  chinese_keyphrase_extractor (CKPE)      |  A tool for chinese keyphrase extraction 一个快速从自然语言文本中提取和识别关键短语的工具    |   [github](https://github.com/dongrixinyu/chinese_keyphrase_extractor)  |
|    简单的简历解析器，用来从简历中提取关键信息    |        |   [github](https://github.com/OmkarPathak/pyresparser)  |
|   BERT-NER-Pytorch三种不同模式的BERT中文NER实验    |        |  [github](https://github.com/lonePatient/BERT-NER-Pytorch)   |




# 知识图谱

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|    清华大学XLORE中英文跨语言百科知识图谱    |   百度、中文维基、英文维基    |  [link](https://xlore.org/downloadhtml)     |
|    文档图谱自动生成    |        |   [github](https://github.com/liuhuanyong/TextGrapher)    |
|     基于医疗领域知识图谱的问答系统   |        |   [github](https://github.com/zhihao-chen/QASystemOnMedicalGraph) <br>该repo参考了[github](https://github.com/liuhuanyong/QASystemOnMedicalKG)   |
|    中文人物关系知识图谱项目    |        |   [github](https://github.com/liuhuanyong/PersonRelationKnowledgeGraph)  |
|    AmpliGraph 知识图谱表示学习(Python)库知识图谱概念链接预测    |        |  [github](https://github.com/Accenture/AmpliGraph)    |
|    中文知识图谱资料、数据及工具    |        |    [github](https://github.com/husthuke/awesome-knowledge-graph) |
|    基于百度百科的中文知识图谱  |     抽取三元组信息，构建中文知识图谱     |   [github](https://github.com/lixiang0/WEB_KG)  |
|    Zincbase 知识图谱构建工具包    |        |  [github](https://github.com/tomgrek/zincbase)   |
|    基于知识图谱的问答系统    |        |  [github](https://github.com/WenRichard/KBQA-BERT)   |
|    知识图谱深度学习相关资料整理    |        |   [github](https://github.com/lihanghang/Knowledge-Graph)  |
|   东南大学《知识图谱》研究生课程(资料)     |        |  [github](https://github.com/npubird/KnowledgeGraphCourse)   |
|    知识图谱车音工作项目    |        |   [github](https://github.com/qiu997018209/KnowledgeGraph)  |
|     《海贼王》知识图谱   |        |   [github](https://github.com/mrbulb/ONEPIECE-KG)  |
|    132个知识图谱的数据集    |    涵盖常识、城市、金融、农业、地理、气象、社交、物联网、医疗、娱乐、生活、商业、出行、科教    |   [link](http//openkg.cn)  |
|    大规模、结构化、中英文双语的新冠知识图谱(COKG-19)    |        |   [link](http://www.openkg.cn/dataset?q=COKG-19)  |
|    基于依存句法与语义角色标注的事件三元组抽取    |        |    [github](https://github.com/liuhuanyong/EventTriplesExtraction)   |
|     抽象知识图谱  |   目前规模50万，支持名词性实体、状态性描述、事件性动作进行抽象      |   [github](https://github.com/liuhuanyong/AbstractKnowledgeGraph)  |
|    大规模中文知识图谱数据14亿实体    |        |   [github](https://github.com/ownthink/KnowledgeGraphData)  |
|    Jiagu自然语言处理工具     |    以BiLSTM等模型为基础，提供知识图谱关系抽取 中文分词 词性标注 命名实体识别 情感分析 新词发现 关键词 文本摘要 文本聚类等功能    |  [github](https://github.com/ownthink/Jiagu)   |
|     medical_NER - 中文医学知识图谱命名实体识别   |        |    [github](https://github.com/pumpkinduo/KnowledgeGraph_NER) |
|   知识图谱相关学习资料/数据集/工具资源大列表     |        |  [github](https://github.com/totogo/awesome-knowledge-graph)   |
|    LibKGE面向可复现研究的知识图谱嵌入库    |        |  [github](https://github.com/uma-pi1/kge)   |
|   基于mongodb存储的军事领域知识图谱问答项目    |    包括飞行器、太空装备等8大类，100余小类，共计5800项的军事武器知识库，该项目不使用图数据库进行存储，通过jieba进行问句解析，问句实体项识别，基于查询模板完成多类问题的查询，主要是提供一种工业界的问答思想demo。    |   [github](https://github.com/liuhuanyong/QAonMilitaryKG)  |
|     京东商品知识图谱   |        |   [github](https://github.com/liuhuanyong/ProductKnowledgeGraph)  |
|    基于远监督的中文关系抽取    |        |    [github](https://github.com/xiaolalala/Distant-Supervised-Chinese-Relation-Extraction) |
|  基于医药知识图谱的智能问答系统      |        |   [github](https://github.com/YeYzheng/KGQA-Based-On-medicine) |
|    BLINK最先进的实体链接库    |        |   [github](https://github.com/facebookresearch/BLINK)  |
|   一个小型的证券知识图谱/知识库     |        |    [github](https://github.com/lemonhu/stock-knowledge-graph) |
|   dstlr非结构化文本可扩展知识图谱构建平台     |        |   [github](https://github.com/dstlry/dstlr)  |
|  百度百科人物词条属性抽取    |  用基于BERT的微调和特征提取方法来进行知识图谱      |   [github](https://github.com/sakuranew/BERT-AttributeExtraction)|
|   新冠肺炎相关数据     |  新冠及其他类型肺炎中文医疗对话数据集；清华大学等机构的开放数据源（COVID-19）   | [github](https://www.aminer.cn/data-covid19/)<br>  [github](https://github.com/UCSD-AI4H/COVID-Dialogue) |
|   DGL-KE 图嵌入表示学习算法     |        |   [github](https://github.com/awslabs/dgl-ke)  |


# 文本生成
 
| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   Texar    |   Toolkit for Text Generation and Beyond     |  [github](https://github.com/asyml/texar)  |
|   Ehud Reiter教授的博客    |        | [link](https://ehudreiter.com)  北大万小军教授强力推荐，该博客对NLG技术、评价与应用进行了深入的探讨与反思。   |
|   文本生成相关资源大列表    |        | [github](https://github.com/ChenChengKuan/awesome-text-generation)     |
|  开放域对话生成及在微软小冰中的实践      |   自然语言生成让机器掌握自动创作的本领    |   [link](https://drive.google.com/file/d/1Mdna3q986k6OoJNsfAHznTtnMAEVzv5z/view)  |
|    文本生成控制   |        |  [github](https://github.com/harvardnlp/Talk-Latent/blob/master/mainpdf)    |
|    自然语言生成相关资源大列表   |        |   [github](https://github.com/tokenmill/awesome-nlg)  |
|    用BLEURT评价自然语言生成   |        |  [link](https://ai.googleblog.com/2020/05/evaluating-natural-language-generation.html)  |
|   自动对联数据及机器人    |        | [代码 link](https://github.com/wb14123/seq2seq-couplet) <br> [70万对联数据](https://github.com/wb14123/couplet-dataset)     |
|   自动生成评论     |   用Transformer编解码模型实现的根据Hacker News文章标题生成评论     |   [github](https://github.com/leod/hncynic)  |
|    自然语言生成SQL语句（英文）    |        |   [github](https://github.com/paulfitz/mlsql)  |
|    自然语言生成资源大全    |        |   [github](https://github.com/tokenmill/awesome-nlg)  |
|    中文生成任务基准测评    |        |  [github](https://github.com/CLUEbenchmark/CLGE)   |
|     基于GPT2的特定主题文本生成/文本增广   |        |   [github](https://github.com/prakhar21/TextAugmentation-GPT2)  |
|     编码、标记和实现一种可控高效的文本生成方法   |        |   [github](https://github.com/yannvgn/laserembeddings)  |
|    TextFooler针对文本分类/推理的对抗文本生成模块    |        |   [github](https://github.com/jind11/TextFooler)  |
|    SimBERT     |基于UniLM思想、融检索与生成于一体的BERT模型        |   [github](https://github.com/ZhuiyiTechnology/simbert)  |
|    新词生成及造句    |    不存在的词用GPT-2变体从头生成新词及其定义、例句    |    [github](https://github.com/turtlesoupy/this-word-does-not-exist) |
|   由文本自动生成多项选择题     |        |   [github](https://github.com/KristiyanVachev/Question-Generation)  |
|     合成数据生成基准   |        |  [github](https://github.com/sdv-dev/SDGym)   |
|       |        |    |

# 文本摘要

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   中文文本摘要/关键词提取     |        |    [github](https://github.com/letiantian/TextRank4ZH) |
|    基于命名实体识别的简历自动摘要    |        |   [github](https://github.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy)  |
|    文本自动摘要库TextTeaser     |  仅支持英文      |   [github](https://github.com/IndigoResearch/textteaser)  |
|    基于BERT等最新语言模型的抽取式摘要提取    |        |   [github](https://github.com/Hellisotherpeople/CX_DB8)  |
|   Python利用深度学习进行文本摘要的综合指南     |        |   [link](https://mp.weixin.qq.com/s/gDZyTbM1nw3fbEnU--y3nQ)  |
|   (Colab)抽象文本摘要实现集锦(教程     |        |   [github](https://github.com/theamrzaki/text_summurization_abstractive_methods)  |


# 智能问答

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|    中文聊天机器人    |  根据自己的语料训练出自己想要的聊天机器人，可以用于智能客服、在线问答、智能聊天等场景      |   [github](https://github.com/Doragd/Chinese-Chatbot-PyTorch-Implementation)  |
|   有趣的情趣robot qingyun      |    qingyun 训练出来的中文聊天机器人     |   [github](https://github.com/Doragd/Chinese-Chatbot-PyTorch-Implementation)  |
|     开放了对话机器人、知识图谱、语义理解、自然语言处理工具及数据   |        |   [github](https://wwwownthinkcom/#header-n30)  |
|  qa对的机器人      |    Amodel-for-Retrivalchatbot - 客服机器人，Chinese Retreival chatbot（中文检索式机器人）    | [git](https://github.com/WenRichard/QAmodel-for-Retrievalchatbot)   |
|  ConvLab开源多域端到端对话系统平台      |        |  [github](https://github.com/ConvLab/ConvLab)   |
|   基于最新版本rasa搭建的对话系统     |        |   [github](https://github.com/GaoQ1/rasa_chatbot_cn)  |
|   基于金融-司法领域(兼有闲聊性质)的聊天机器人     |        |   [github](https://github.com/charlesXu86/Chatbot_CN)  |
|    端到端的封闭域对话系统    |        |  [github](https://github.com/cdqa-suite/cdQA)   |
|     MiningZhiDaoQACorpus    |    580万百度知道问答数据挖掘项目，百度知道问答语料库，包括超过580万的问题，每个问题带有问题标签。基于该问答语料库，可支持多种应用，如逻辑挖掘    |    [github]() |
|   用于中文闲聊的GPT2模型GPT2-chitchat     |        |    [github](https://github.com/yangjianxin1/GPT2-chitchat) |
|    基于检索聊天机器人多轮响应选择相关资源列表(Leaderboards、Datasets、Papers)    |        |   [github](https://github.com/JasonForJoy/Leaderboards-for-Multi-Turn-Response-Selection)  |
|   微软对话机器人框架     |        |    [github](https://github.com/microsoft/botframework) |
|      chatbot-list  |   行业内关于智能客服、聊天机器人的应用和架构、算法分享和介绍    |  [github](https://github.com/lizhe2004/chatbot-list)   |
|     Chinese medical dialogue data 中文医疗对话数据集   |        |   [github](https://github.com/Toyhom/Chinese-medical-dialogue-data)  |
|    一个大规模医疗对话数据集    |   包含110万医学咨询，400万条医患对话    |    [github](https://github.com/UCSD-AI4H/Medical-Dialogue-System) |
|    大规模跨领域中文任务导向多轮对话数据集及模型CrossWOZ    |        |  [paper & data](https://arxiv.org/pdf/200211893pdf)   |
|   开源对话式信息搜索平台     |        |    [github](https://github.com/microsoft/macaw) |
|      情境互动多模态对话挑战2020(DSTC9 2020)  |        |  [github](https://github.com/facebookresearch/simmc)   |
|    用Quora问题对训练的T5问题意译(Paraphrase)    |        |   [github](https://github.com/renatoviolin/T5-paraphrase-generation)  |
|    Google发布Taskmaster-2自然语言任务对话数据集    |        |   [github](https://github.com/google-research-datasets/Taskmaster/tree/master/TM-2-2020)  |
|    Haystack灵活、强大的可扩展问答(QA)框架    |        |   [github](https://github.com/deepset-ai/haystack)  |
|    端到端的封闭域对话系统    |        |   [github](https://github.com/cdqa-suite/cdQA)  |
|   Amazon发布基于知识的人-人开放领域对话数据集     |        |   [github](https://github.com/alexa/alexa-prize-topical-chat-dataset/)  |
|    基于百度webqa与dureader数据集训练的Albert Large QA模型    |        |   [github](https://github.com/wptoux/albert-chinese-large-webqa/tree/master)  |
|   CommonsenseQA面向常识的英文QA挑战     |        |   [link](https://www.tau-nlp.org/commonsenseqa)  |
|   MedQuAD(英文)医学问答数据集     |        |  [github](https://github.com/abachaa/MedQuAD)   |
|    基于Albert、Electra，用维基百科文本作为上下文的问答引擎    |        |   [github](https://github.com/renatoviolin/Question-Answering-Albert-Electra)  |
|   基于14W歌曲知识库的问答尝试    |     功能包括歌词接龙，已知歌词找歌曲以及歌曲歌手歌词三角关系的问答   |    [github](https://github.com/liuhuanyong/MusicLyricChatbot) |


# 文本纠错

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|  中文文本纠错模块代码      |        |  [github](https://github.com/zedom1/error-detection)   |
|    英文拼写检查库    |        |   [github](https://github.com/barrust/pyspellchecker)  |
|  python拼写检查库      |        |  [github](https://github.com/barrust/pyspellchecker)   |
|    GitHub Typo Corpus大规模GitHub多语言拼写错误/语法错误数据集    |        |   [github](https://github.com/mhagiwara/github-typo-corpus)  |
|    BertPunc基于BERT的最先进标点修复模型    |        |   [github](https://github.com/nkrnrnk/BertPunc)  |
|    中文写作校对工具    |        |  [github](https://xiezuocat.com/#/)   |


# 多模态
| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|中文多模态数据集「悟空」|华为诺亚方舟实验室开源大型，包含1亿图文对|[github](https://wukong-dataset.github.io/wukong-dataset/)|


# 语音处理

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|    ASR 语音数据集 + 基于深度学习的中文语音识别系统    |        |    [github](https://github.com/nl8590687/ASRT_SpeechRecognition)  |
|   清华大学THCHS30中文语音数据集    |        |  [data_thchs30tgz-OpenSLR国内镜像](<http//cn-mirroropenslrorg/resources/18/data_thchs30tgz>)<br>[data_thchs30tgz](<http//wwwopenslrorg/resources/18/data_thchs30tgz>) <br>[test-noisetgz-OpenSLR国内镜像](<http//cn-mirroropenslrorg/resources/18/test-noisetgz>)[test-noisetgz](<http//wwwopenslrorg/resources/18/test-noisetgz>) <br>[resourcetgz-OpenSLR国内镜像](<http//cn-mirroropenslrorg/resources/18/resourcetgz>)<br>[resourcetgz](<http//wwwopenslrorg/resources/18/resourcetgz>)<br>[Free ST Chinese Mandarin Corpus](<http//cn-mirroropenslrorg/resources/38/ST-CMDS-20170001_1-OStargz>)<br>[Free ST Chinese Mandarin Corpus](<http//wwwopenslrorg/resources/38/ST-CMDS-20170001_1-OStargz>)<br>[AIShell-1 开源版数据集-OpenSLR国内镜像](<http//cn-mirroropenslrorg/resources/33/data_aishelltgz>)<br>[AIShell-1 开源版数据集](<http//wwwopenslrorg/resources/33/data_aishelltgz>)<br>[Primewords Chinese Corpus Set 1-OpenSLR国内镜像](<http//cn-mirroropenslrorg/resources/47/primewords_md_2018_set1targz>)<br>[Primewords Chinese Corpus Set 1](<http//wwwopenslrorg/resources/47/primewords_md_2018_set1targz>) |
|    笑声检测器    |        |    [github](https://github.com/ideo/LaughDetection)  |
|    Common Voice语音识别数据集新版    |  包括来自42,000名贡献者超过1,400小时的语音样本，涵github     |   [link](https://voice.mozilla.org/en/datasets)     |
|    speech-aligner    |  从“人声语音”及其“语言文本”，产生音素级别时间对齐标注的工具       |   [github](https://github.com/open-speech/speech-aligner)  |
|   ASR语音大辞典/词典     |        |   [github](hhttps://github.com/aishell-foundation/DaCiDian)  |
|     语音情感分析   |        |   [github](https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer)  |
|    masr     | 中文语音识别，提供预训练模型，高识别率       |   [github](https://github.com/lukhy/masr)  |
|    面向语音识别的中文文本规范化    |        |  [github](https://github.com/speech-io/chinese_text_normalization)   |
|      语音质量评价指标(MOSNet, BSSEval, STOI, PESQ, SRMR)  |        |   [github](https://github.com/aliutkus/speechmetrics)  |
|    面向语音识别的中文/英文发音辞典    |        |   [github](https://github.com/speech-io/BigCiDian)  |
|    CoVoSTFacebook发布的多语种语音-文本翻译语料库    | 包括11种语言(法语、德语、荷兰语、俄语、西班牙语、意大利语、土耳其语、波斯语、瑞典语、蒙古语和中文)的语音、文字转录及英文译文      |    [github](https://github.com/facebookresearch/covost) |
|    Parakeet基于PaddlePaddle的文本-语音合成    |        |   [github](https://github.com/PaddlePaddle/Parakeet)  |
|     (Java)准确的语音自然语言检测库   |        |   [github](https://github.com/pemistahl/lingua)  |
|   CoVoSTFacebook发布的多语种语音-文本翻译语料库     |        |   [github](https://github.com/facebookresearch/covost)  |
|   TensorFlow 2 实现的文本语音合成     |        |   [github](https://github.com/as-ideas/TransformerTTS)  |
|    Python音频特征提取包    |        |  [github](https://github.com/novoic/surfboard)   |
|   ViSQOL音频质量感知客观、完整参考指标，分音频、语音两种模式     |        |  [github](https://github.com/google/visqol)   |
|    zhrtvc    |     好用的中文语音克隆兼中文语音合成系统     |  [github](https://github.com/KuangDD/zhrtvc)  |
|      aukit    |  好用的语音处理工具箱，包含语音降噪、音频格式转换、特征频谱生成等模块       |  [github](https://github.com/KuangDD/aukit)  |
|      phkit    |   好用的音素处理工具箱，包含中文音素、英文音素、文本转拼音、文本正则化等模块     |  [github](https://github.com/KuangDD/phkit)   |
|     zhvoice     |   中文语音语料，语音更加清晰自然，包含8个开源数据集，3200个说话人，900小时语音，1300万字     |  [github](https://github.com/KuangDD/zhvoice)   |
|   audio面向语音行为检测     |  、二值化、说话人识别、自动语音识别、情感识别等任务的音频标注工具      |  [github](https://github.com/midas-research/audino)   |
|     深度学习情感文本语音合成   |        |  [github](https://github.com/Emotional-Text-to-Speech/dl-for-emo-tts)   |
|   Python音频数据增广库     |        |   [github](https://github.com/iver56/audiomentations)  |
|   基于大规模音频数据集Audioset的音频增强     |        |    [github](https://github.com/AppleHolic/audioset_augmentor) |
|    语声迁移    |        |   [github](https://github.com/fighting41love/become-yukarin)  |



# 文档处理

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   PyLaia面向手写文档分析的深度学习工具包     |        |   [github](https://github.com/jpuigcerver/PyLaia)  |
|    单文档非监督的关键词抽取    |        |  [github](https://github.com/LIAAD/yake)   |
|      DocSearch免费文档搜索引擎  |        |   [github](https://github.com/algolia/docsearch)  |
|  fdfgen       |    能够自动创建pdf文档，并填写信息     | [link](https://github.com/ccnmtl/fdfgen)   |
| pdfx       |   自动抽取出引用参考文献，并下载对应的pdf文件 | [link](https://github.com/metachris/pdfx)   |
|     invoice2data   |   发票pdf信息抽取     |  [invoice2data](https://github.com/invoice-x/invoice2data)  |
|   pdf文档信息抽取    |        |  [github](https://github.com/jstockwin/py-pdf-parser)   |
|PDFMiner     |     PDFMiner能获取页面中文本的准确位置，以及字体或行等其他信息。它还有一个PDF转换器，可以将PDF文件转换成其他文本格式(如HTML)。还有一个可扩展的解析器PDF，可以用于文本分析以外的其他用途。   |  [link](https://github.com/euske/pdfminer)  |
|  PyPDF2      |    PyPDF 2是一个python PDF库，能够分割、合并、裁剪和转换PDF文件的页面。它还可以向PDF文件中添加自定义数据、查看选项和密码。它可以从PDF检索文本和元数据，还可以将整个文件合并在一起。    |  [link](https://github.com/mstamy2/PyPDF2)   |
|   PyPDF2     |     PyPDF 2是一个python PDF库，能够分割、合并、裁剪和转换PDF文件的页面。它还可以向PDF文件中添加自定义数据、查看选项和密码。它可以从PDF检索文本和元数据，还可以将整个文件合并在一起。  |   [link](https://github.com/mstamy2/PyPDF2)  |
|    ReportLab   |      ReportLab能快速创建PDF 文档。经过时间证明的、超好用的开源项目，用于创建复杂的、数据驱动的PDF文档和自定义矢量图形。它是免费的，开源的，用Python编写的。该软件包每月下载5万多次，是标准Linux发行版的一部分，嵌入到许多产品中，并被选中为Wikipedia的打印/导出功能提供动力。  | [link](https://www.reportlab.com/opensource/)   |
|    SIMPdfPython写的简单PDF文件文字编辑器    |        |   [github](https://github.com/shashanoid/Simpdf)  |
|pdf-diff |PDF文件diff工具 可显示两个pdf文档的差别| [github](https://github.com/serhack/pdf-diff)|

# 表格处理

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|  用unet实现对文档表格的自动检测，表格重建      |        |   [github](https://github.com/chineseocr/table-ocr)  |
|   pdftabextract    |  用于OCR识别后的表格信息解析，很强大      |    [link](https://github.com/WZBSocialScienceCenter/pdftabextract)   |
| tabula-py    |     直接将pdf中的表格信息转换为pandas的dataframe，有java和python两种版本代码   |    [](https://github.com/chezou/tabula-py) |
|   camelot     |   pdf表格解析       |  [link](https://github.com/atlanhq/camelot)  |
| pdfplumber      |   pdf表格解析     | [](https://github.com/jsvine/pdfplumber)   |
|   PubLayNet   |     能够划分段落、识别表格、图片   |  [link](https://github.com/ibm-aur-nlp/PubTabNet)  |
|    从论文中提取表格数据 |        |    [github](https://github.com/paperswithcode/axcell) |
|    用BERT在表格中寻找答案    |        |  [github](https://github.com/google-research/tapas)   |
|    表格问答的系列文章    |        |  [简介](https://mp.weixin.qq.com/s?__biz=MzAxMDk0OTI3Ng==&mid=2247484103&idx=2&sn=4a5b50557ab9178270866d812bcfc87f&chksm=9b49c534ac3e4c22de7c53ae5d986fac60a7641c0c072d4038d9d4efd6beb24a22df9f859d08&scene=21#wechat_redirect)<br>[模型](https://mp.weixin.qq.com/s?__biz=MzAxMDk0OTI3Ng==&mid=2247484103&idx=1&sn=73f37fbc1dbd5fdc2d4ad54f58693ef3&chksm=9b49c534ac3e4c222f6a320674b3728cf8567b9a16e6d66b8fdcf06703b05a16a9c9ed9d79a3&scene=21#wechat_redirect)<br>[完结篇](https://mp.weixin.qq.com/s/ee1DG_vO2qblqFC6zO97pA)  |
|     使用GAN生成表格数据（仅支持英文）   |        |   [github](https://github.com/Diyago/GAN-for-tabular-data)  |
|  carefree-learn(PyTorch)      |    表格数据集自动化机器学习(AutoML)包    |  [github](https://github.com/carefree0910/carefree-learn)   |
|   封闭域微调表格检测     |        |  [github](https://github.com/holms-ur/fine-tuning)   |
|   PDF表格数据提取工具     |        |   [github](https://github.com/camelot-dev/camelot)  |
|     TaBERT理解表格数据查询的新模型   |        |  [paper](https://scontent-hkt1-1xxfbcdnnet/v/t398562-6/106708899_597765107810230_1899215558892880563_npdf?_nc_cat=107&_nc_sid=ae5e01&_nc_ohc=4sN3TJwewSIAX8iliBD&_nc_ht=scontent-hkt1-1xx&oh=eccb9795f027ff63be61ff4a5e337c02&oe=5F316505)   |
| 表格处理 | Awesome-Table-Recognition | [github](https://github.com/cv-small-snails/Awesome-Table-Recognition)|



# 文本匹配

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|    句子、QA相似度匹配MatchZoo    |   文本相似度匹配算法的集合，包含多个深度学习的方法，值得尝试。    |    [github](https://github.com/NTMC-Community/MatchZoo)  |
|    中文问题句子相似度计算比赛及方案汇总    |        |   [github](https://github.com/ShuaichiLi/Chinese-sentence-similarity-task)  |
|    similarity相似度计算工具包    |   java编写,用于词语、短语、句子、词法分析、情感分析、语义分析等相关的相似度计算    |  [github](https://github.com/shibing624/similarity)   |
|    中文词语相似度计算方法    |  综合了同义词词林扩展版与知网（Hownet）的词语相似度计算方法，词汇覆盖更多、结果更准确。      |    [gihtub](https://github.com/yaleimeng/Final_word_Similarity) |
|    Python字符串相似性算法库    |        |    [github](https://github.com/luozhouyang/python-string-similarity) |
|    基于Siamese bilstm模型的相似句子判定模型,提供训练数据集和测试数据集    |    提供了10万个训练样本    | [github](https://github.com/liuhuanyong/SiameseSentenceSimilarity)    |


# 文本数据增强

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|    中文NLP数据增强（EDA）工具  |        |    [github](https://github.com/zhanlaoban/eda_nlp_for_Chinese) |
|   英文NLP数据增强工具     |        |  [github](https://github.com/makcedward/nlpaug)   |
|    一键中文数据增强工具    |        | [github](https://github.com/425776024/nlpcda)   |
|    数据增强在机器翻译及其他nlp任务中的应用及效果    |        |  [link](https://mp.weixin.qq.com/s/_aVwSWuYho_7MUT0LuFgVA)   |
|    NLP数据增广资源集    |        |   [github](https://github.com/quincyliang/nlp-data-augmentation)  |


# 常用正则表达式

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   抽取email的正则表达式    |    |  已集成到 python package [cocoNLP](https://github.com/fighting41love/cocoNLP)中，欢迎试用  |
|   抽取phone_number    |     | 已集成到 python package [cocoNLP](https://github.com/fighting41love/cocoNLP)中，欢迎试用   |
|    抽取身份证号的正则表达式   |  IDCards_pattern = r'^([1-9]\d{5}[12]\d{3}(0[1-9]\|1[012])(0[1-9]\|[12][0-9]\|3[01])\d{3}[0-9xX])<br>IDs = re.findall(IDCards_pattern, text, flags=0)|   
  IP地址正则表达式|(25[0-5]\|  2[0-4]\d\|  [0-1]\d{2}\|  [1-9]?\d)\.(25[0-5]\|  2[0-4]\d\|  [0-1]\d{2}\|  [1-9]?\d)\.(25[0-5]\|  2[0-4]\d\|  [0-1]\d{2}\|  [1-9]?\d)\.(25[0-5]\|  2[0-4]\d\|  [0-1]\d{2}\|  [1-9]?\d)||
|  腾讯QQ号正则表达式     |   \[1-9]([0-9]{5,11})     |    |
|   国内固话号码正则表达式    |      [0-9-()（）]{7,18}  |    |
|   用户名正则表达式    |  [A-Za-z0-9_\-\u4e00-\u9fa5]+      |    |
|    国内电话号码正则匹配（三大运营商+虚拟等）    |        |   [github](https://github.com/VincentSit/ChinaMobilePhoneNumberRegex)  |
|     正则表达式教程   |        |  [github](https://github.com/ziishaned/learn-regex/blob/master/translations/README-cnmd)   |


# 文本检索

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   高效模糊搜索工具     |        |  [github](https://github.com/Yggdroot/LeaderF)   |
|  面向各语种/任务的BERT模型大列表/搜索引擎      |        |    [link](https://bertlang.unibocconi.it/) |
|    Deepmatch针对推荐、广告和搜索的深度匹配模型库    |        |   [github](https://github.com/shenweichen/DeepMatch)  |
|    wwsearch是企业微信后台自研的全文检索引擎    |        |   [github](https://github.com/Tencent/wwsearch)  |
|   aili - the fastest in-memory index in the East 东半球最快并发索引     |        |    [github](https://github.com/UncP/aili) |


# 阅读理解

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   高效模糊搜索工具     |        |  [github](https://github.com/Yggdroot/LeaderF)   |
|  面向各语种/任务的BERT模型大列表/搜索引擎      |        |    [link](https://bertlang.uniboc.coni.it) |
|    Deepmatch针对推荐、广告和搜索的深度匹配模型库    |        |   [github](https://github.com/shenweichen/DeepMatch)  |
|   allennlp阅读理解支持多种数据和模     |        |  [github](https://github.com/allenai/allennlp-reading-comprehension)   |



# 情感分析

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|     方面情感分析包   |        |   [github](https://github.com/ScalaConsultants/Aspect-Based-Sentiment-Analysis)  |
|    awesome-nlp-sentiment-analysis    |    情感分析、情绪原因识别、评价对象和评价词抽取   |  [github](https://github.com/haiker2011/awesome-nlp-sentiment-analysis)   |
|    情感分析技术让智能客服更懂人类情感    |        |   [github](https://developeraliyuncom/article/761513?utm_content=g_1000124809)  |


# 事件抽取

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   中文事件抽取    |        |  [github](https://github.com/liuhuanyong/ComplexEventExtraction)  |
|     NLP事件提取文献资源列表   |        |  [github](https://github.com/BaptisteBlouin/EventExtractionPapers)   |
|    PyTorch实现的BERT事件抽取(ACE 2005 corpus)    |         | [github](https://github.com/nlpcl-lab/bert-event-extraction)   |
|  新闻事件线索抽取      |        |  [github](https://github.com/liuhuanyong/ImportantEventExtractor)   |


# 机器翻译

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   无道词典     |  有道词典的命令行版本，支持英汉互查和在线查询      |  [github](https://github.com/ChestnutHeng/Wudao-dict)   |
|NLLB|支持200+种语言任意互译的语言模型NLLB|[link](https://openbmb.github.io/BMList/list/)|

# 数字转换

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   最好的汉字数字(中文数字)-阿拉伯数字转换工具     |        |   [github](https://github.com/Wall-ee/chinese2digits)  |
|  快速转化「中文数字」和「阿拉伯数字」      |        |    [github](https://github.com/HaveTwoBrush/cn2an) |
|    将自然语言数字串解析转换为整数和浮点数    |        |   [github](https://github.com/jaidevd/numerizer)  |


# 指代消解

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|    中文指代消解数据    |        |   [github](https://github.com/CLUEbenchmark/CLUEWSC2020) <br>[baidu ink](https://panbaiducom/s/1gKP_Mj-7KVfFWpjYvSvAAA)  code a0qq |


# 文本聚类

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|     TextCluster短文本聚类预处理模块 Short text cluster   |        |    [github](https://github.com/RandyPen/TextCluster) |


# 文本分类


| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|    NeuralNLP-NeuralClassifier腾讯开源深度学习文本分类工具    |        |    [github](https://github.com/Tencent/NeuralNLP-NeuralClassifier) |


# 知识推理

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   GraphbrainAI开源软件库和科研工具，目的是促进自动意义提取和文本理解以及知识的探索和推断     |        |    [github](https://github.com/graphbrain/graphbrain) |
|    (哈佛)讲因果推理的免费书    |        |  [pdf](https://cdn1sphharvardedu/wp-content/uploads/sites/1268/2019/10/ci_hernanrobins_23oct19pdf)   |

# 可解释自然语言处理

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   文本机器学习模型最先进解释器库     |        |  [github](https://github.com/interpretml/interpret-text)   |


# 文本攻击

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|     TextAttack自然语言处理模型对抗性攻击框架   |        |    [github](https://github.com/QData/TextAttack) |
|OpenBackdoor: 文本后门攻防工具包|       OpenBackdoor基于Python和PyTorch开发，可用于复现、评估和开发文本后门攻防的相关算法     |    [github](https://github.com/thunlp/OpenBackdoor)|

# 文本可视化

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|     Scattertext 文本可视化(python)   |        |   [github](https://github.com/JasonKessler/scattertext)  |
|     whatlies词向量交互可视化   |        | [spacy工具](https://spacyio/universe/project/whatlies)  |
|   PySS3面向可解释AI的SS3文本分类器机器可视化工具     |        |   [github](https://github.com/sergioburdisso/pyss3)  |
|     用记事本渲染3D图像   |        | [github](https://github.com/khalladay/render-with-notepad)    |
|    attnvisGPT2、BERT等transformer语言模型注意力交互可视化    |        |   [github](https://github.com/SIDN-IAP/attnvis)  |
|    Texthero文本数据高效处理包    |   包括预处理、关键词提取、命名实体识别、向量空间分析、文本可视化等     |  [github](https://github.com/jbesomi/texthero)   |

# 文本标注工具

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   NLP标注平台综述     |        |   [github](https://github.com/alvations/annotate-questionnaire)  |
|   brat rapid annotation tool 序列标注工具     |        |   [link](http://brat.nlplab.org/index.html)  |
|     Poplar网页版自然语言标注工具   |        |    [github](https://github.com/synyi/poplar) |
|   LIDA轻量交互式对话标注工具     |        |  [github](https://github.com/Wluper/lida)   |
|    doccano基于网页的开源协同多语言文本标注工具    |        |   [github](https://github.com/doccano/doccano)  |
|     Datasaurai 在线数据标注工作流管理工具   |        |    [link](https://datasaurai.gitbook.io/datasaur/) |

# 语言检测

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|  langid     |   97种语言检测     |  [https://github.com/saffsd/langid.py](https://github.com/saffsd/langid.py)  |
|   langdetect    |   语言检测     |  [https://code.google.com/archive/p/language-detection/](https://code.google.com/archive/p/language-detection/)  |

# 综合工具

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   jieba    |        |  [jieba](https://github.com/fxsjy/jieba)  |
|  hanlp     |        |   [hanlp](https://github.com/hankcs/pyhanlp) |
|    nlp4han    |  中文自然语言处理工具集(断句/分词/词性标注/组块/句法分析/语义分析/NER/N元语法/HMM/代词消解/情感分析/拼写检    |   [github](https://github.com/kidden/nlp4han)  |
|    仇恨言论检测进展    |        |   [link](https://ai.facebook.com/blog/ai-advances-to-better-detect-hate-speech)  |
|   基于Pytorch的Bert应用    |    包括命名实体识别、情感分析、文本分类以及文本相似度等    |  [github](https://github.com/rsanshierli/EasyBert)   |
|    nlp4han中文自然语言处理工具集    |   断句/分词/词性标注/组块/句法分析/语义分析/NER/N元语法/HMM/代词消解/情感分析/拼写检查     |   [github](https://github.com/kidden/nlp4han)  |
|    一些关于自然语言的基本模型    |        |  [github](https://github.com/lpty/nlp_base)   |
|     用BERT进行序列标记和文本分类的模板代码   |        |  [github](https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification)|
|   jieba_fast 加速版的jieba     |        |  [github](https://github.com/deepcs233/jieba_fast)   |
|    StanfordNLP     |   纯Python版自然语言处理包     |  [link](https://stanford.nlp.github.io/stanfordnlp/)   |
|     Python口语自然语言处理工具集(英文)   |        |  [github](https://github.com/gooofy/py-nltools)   |
|    PreNLP自然语言预处理库    |        |  [github](https://github.com/lyeoni/prenlp)   |
|    nlp相关的一些论文及代码    |  包括主题模型、词向量(Word Embedding)、命名实体识别(NER)、文本分类(Text Classificatin)、文本生成(Text Generation)、文本相似性(Text Similarity)计算等，涉及到各种与nlp相关的算法，基于keras和tensorflow      |    [github](https://github.com/msgi/nlp-journey) |
|  Python文本挖掘/NLP实战示例      |        |   [github](https://github.com/kavgan/nlp-in-practice)  |
|   Forte灵活强大的自然语言处理pipeline工具集     |        |    [github](https://github.com/asyml/forte) |
|   stanza斯坦福团队NLP工具     |  可处理六十多种语言   |    [github](https://github.com/stanfordnlp/stanza) |
|   Fancy-NLP用于建设商品画像的文本知识挖掘工具     |        |   [github](https://github.com/boat-group/fancy-nlp)  |
|    全面简便的中文 NLP 工具包    |        |   [github](https://github.com/dongrixinyu/JioNLP)  |
|   工业界常用基于DSSM向量化召回pipeline复现     |        |   [github](https://github.com/wangzhegeek/DSSM-Lookalike)  |
|    Texthero文本数据高效处理包    |   包括预处理、关键词提取、命名实体识别、向量空间分析、文本可视化等     |  [github](https://github.com/jbesomi/texthero)   |
|    nlpgnn图神经网络自然语言处理工具箱    |        |  [github](https://github.com/kyzhouhzau/NLPGNN)   |
|    Macadam   |  以Tensorflow(Keras)和bert4keras为基础，专注于文本分类、序列标注和关系抽取的自然语言处理工具包     |    [github](https://github.com/yongzhuo/Macadam) |
|    LineFlow面向所有深度学习框架的NLP数据高效加载器    |        |   [github](https://github.com/tofunlp/lineflow)  |


# 有趣搞笑工具

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   汪峰歌词生成器    |        |  [phunterlau/wangfeng-rnn](https://github.com/phunterlau/wangfeng-rnn)  |
|  女友 情感波动分析     |        |  [github](https://github.com/CasterWx/python-girlfriend-mood/)  |
|   NLP太难了系列    |        |   [github](https://github.com/fighting41love/hardNLP)  |
|  变量命名神器     |        |  [github](https://github.com/unbug/codelf) [link](https://unbug.github.io/codelf/)  |
|     图片文字去除，可用于漫画翻译   |        |  [github](https://github.com/yu45020/Text_Segmentation_Image_Inpainting)   |
|     CoupletAI - 对联生成   |   基于CNN+Bi-LSTM+Attention 的自动对对联系统     |  [github](https://github.com/WiseDoge/CoupletAI)   |
|   用神经网络符号推理求解复杂数学方程     |        |    [github](https://ai.facebook.com/blog/using-neural-networks-to-solve-advanced-mathematics-equations/) |
|   基于14W歌曲知识库的问答机器人    |     功能包括歌词接龙，已知歌词找歌曲以及歌曲歌手歌词三角关系的问答   |    [github](https://github.com/liuhuanyong/MusicLyricChatbot) |
|    COPE - 格律诗编辑程序    |        |  [github](https://github.com/LingDong-/cope)   |
|Paper2GUI | 一款面向普通人的AI桌面APP工具箱，免安装即开即用，已支持18+AI模型，内容涵盖语音合成、视频补帧、视频超分、目标检测、图片风格化、OCR识别等领域 |   [github](https://github.com/Baiyuetribe/paper2gui) |  
|礼貌程度估算器（使用新浪微博数据训练）|| [github](https://github.com/tslmy/politeness-estimator) [paper](https://dl.acm.org/doi/abs/10.1145/3415190)|

# 课程报告面试等

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   自然语言处理报告     |        |  [link](https://static.aminer.cn/misc/article/nlppdf)  |
|    知识图谱报告   |        |   [link](https://www.aminer.cn/research_report/5c3d5a8709%20e961951592a49d?download=true&pathname=knowledgegraphpdf) |
|   数据挖掘报告   |        |  [link](https://www.aminer.cn/research_report/5c3d5a5cecb160952fa10b76?download=true&pathname=dataminingpdf)  |
|   自动驾驶报告    |        |  [link](https://static.aminer.cn/misc/article/selfdrivingpdf)  |
|   机器翻译报告    |        |  [link](https://static.aminer.cn/misc/article/translationpdf)  |
|    区块链报告   |        |  [link](https://static.aminer.cn/misc/article/blockchain_publicpdf)  |
|   机器人报告    |        | [link](https://static.aminer.cn/misc/article/robotics_betapdf)   |
|   计算机图形学报告    |        |  [link](https://static.aminer.cn/misc/article/cgpdf)  |
|  3D打印报告    |        |  [link](https://static.aminer.cn/misc/article/3dpdf)  |
|   人脸识别报告    |        |  [link](https://static.aminer.cn/misc/article/facerecognitionpdf)  |
|   人工智能芯片报告    |        |  [link](https://static.aminer.cn/misc/article/aichippdf)  |
|   cs224n深度学习自然语言处理课程    |        |  [link](http//web.stanford.edu/class/cs224n/) 课程中模型的pytorch实现 [link](https://github.com/DSKSD/DeepNLP-models-Pytorch)   |
|   面向深度学习研究人员的自然语言处理实例教程     |        |  [github](https://github.com/graykode/nlp-tutorial)  |
|   《Natural Language Processing》by Jacob Eisenstein     |        |   [github](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notespdf)  |
|     ML-NLP    | 机器学习(Machine Learning)、NLP面试中常考到的知识点和代码实现       |    [github](https://github.com/NLP-LOVE/ML-NLP) |
|      NLP任务示例项目代码集  |        |   [github](https://github.com/explosion/projects)  |
|     2019年NLP亮点回顾   |        |   [download](https://panbaiducom/s/1h5gEPUhvY1HkUVc32eeX4w)  |
|   nlp-recipes微软出品--自然语言处理最佳实践和范例     |        |   [github](https://github.com/microsoft/nlp-recipes)  |
|    面向深度学习研究人员的自然语言处理实例教程    |        |   [github](https://github.com/graykode/nlp-tutorial)  |
|   Transfer Learning in Natural Language Processing (NLP)     |        |    [youtube](https://www.youtube.com/watch?v=ly0TRNr7I_M) |
|《机器学习系统》图书|  |  [link](https://openmlsys.github.io/)  [github](https://github.com/fighting41love/openmlsys-zh) |


# 比赛

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|    复盘所有NLP比赛的TOP方案    |        |   [github](https://github.com/zhpmatrix/nlp-competitions-list-review)  |
|   2019年百度的三元组抽取比赛，“科学空间队”源码(第7名)     |        |   [github](https://github.com/bojone/kg-2019)  |


# 金融自然语言处理

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   BDCI2019金融负面信息判定     |        |   [github](https://github.com/A-Rain/BDCI2019-Negative_Finance_Info_Judge)  |
|     开源的金融投资数据提取工具    |        |    [github](https://github.com/PKUJohnson/OpenData) |
|    金融领域自然语言处理研究资源大列表    |        |    [github](https://github.com/icoxfog417/awesome-financial-nlp) |
|   基于金融-司法领域(兼有闲聊性质)的聊天机器人     |        |   [github](https://github.com/charlesXu86/Chatbot_CN)  |


# 医疗自然语言处理

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|   中文医学NLP公开资源整理     |        |   [github](https://github.com/GanjinZero/awesome_Chinese_medical_NLP)  |
|    spaCy 医学文本挖掘与信息提取    |        |  [github](https://github.com/NLPatVCU/medaCy)   |
|    构建医疗实体识别的模型 |   包含词典和语料标注，基于python    |  [github](https://github.com/yixiu00001/LSTM-CRF-medical)   |
|     基于医疗领域知识图谱的问答系统   |        |   [github](https://github.com/zhihao-chen/QASystemOnMedicalGraph) 该repo参考了[github](https://github.com/liuhuanyong/QASystemOnMedicalKG)   |
|     Chinese medical dialogue data 中文医疗对话数据集   |        |   [github](https://github.com/Toyhom/Chinese-medical-dialogue-data)  |
|    一个大规模医疗对话数据集    |   包含110万医学咨询，400万条医患对话    |    [github](https://github.com/UCSD-AI4H/Medical-Dialogue-System) |
|   新冠肺炎相关数据     |  新冠及其他类型肺炎中文医疗对话数据集；清华大学等机构的开放数据源（COVID-19）   | [github](https://www。aminer。cn/data-covid19/)<br>  [github](https://github.com/UCSD-AI4H/COVID-Dialogue) |


# 法律自然语言处理

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|    Blackstone面向非结构化法律文本的spaCy pipeline和NLP模型    |        |    [github](https://github.com/ICLRandD/Blackstone) |
|   法务智能文献资源列表     |        |  [github](https://github.com/thunlp/LegalPapers)   |
|   基于金融-司法领域(兼有闲聊性质)的聊天机器人     |        |   [github](https://github.com/charlesXu86/Chatbot_CN)  |
|   罪名法务名词及分类模型    |    包含856项罪名知识图谱, 基于280万罪名训练库的罪名预测,基于20W法务问答对的13类问题分类与法律资讯问答功能    |    [github](https://github.com/liuhuanyong/CrimeKgAssitant)     |


# 其他

| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|  phone     |     中国手机归属地查询    |  [ls0f/phone](https://github.com/ls0f/phone)  |
|   phone    |    国际手机、电话归属地查询    |   [AfterShip/phone](https://github.com/AfterShip/phone) |
|    ngender   |   根据名字判断性别     |  [observerss/ngender](https://github.com/observerss/ngender)  |
|    中文对比英文自然语言处理NLP的区别综述  |        |   [link](https://mp.weixin.qq.com/s/LQU_HJ4q74lL5oCIk7w5RA)  |
|  各大公司内部里大牛分享的技术文档 PDF 或者 PPT      |        |   [github](https://github.com/0voice/from_coder_to_expert)  |
|   comparxiv 用于比较arXiv上两提交版本差异的命令     |        |  [pypi](https://pypiorg/project/comparxiv/)   |
|     CHAMELEON深度学习新闻推荐系统元架构   |        | [github](https://github.com/gabrielspmoreira/chameleon_recsys)    |
|    简历自动筛选系统    |        |   [github](https://github.com/JAIJANYANI/Automated-Resume-Screening-System)  |
|    Python实现的多种文本可读性评价指标    |        |    [github](https://github.com/cdimascio/py-readability-metrics) |




# 备注

涉及内容包括但不限于：**中英文敏感词、语言检测、中外手机/电话归属地/运营商查询、名字推断性别、手机号抽取、身份证抽取、邮箱抽取、中日文人名库、中文缩写库、拆字词典、词汇情感值、停用词、反动词表、暴恐词表、繁简体转换、英文模拟中文发音、汪峰歌词生成器、职业名称词库、同义词库、反义词库、否定词库、汽车品牌词库、汽车零件词库、连续英文切割、各种中文词向量、公司名字大全、古诗词库、IT词库、财经词库、成语词库、地名词库、历史名人词库、诗词词库、医学词库、饮食词库、法律词库、汽车词库、动物词库、中文聊天语料、中文谣言数据、百度中文问答数据集、句子相似度匹配算法集合、bert资源、文本生成&摘要相关工具、cocoNLP信息抽取工具、国内电话号码正则匹配、清华大学XLORE:中英文跨语言百科知识图谱、清华大学人工智能技术系列报告、自然语言生成、NLU太难了系列、自动对联数据及机器人、用户名黑名单列表、罪名法务名词及分类模型、微信公众号语料、cs224n深度学习自然语言处理课程、中文手写汉字识别、中文自然语言处理 语料/数据集、变量命名神器、分词语料库+代码、任务型对话英文数据集、ASR 语音数据集 + 基于深度学习的中文语音识别系统、笑声检测器、Microsoft多语言数字/单位/如日期时间识别包、中华新华字典数据库及api(包括常用歇后语、成语、词语和汉字)、文档图谱自动生成、SpaCy 中文模型、Common Voice语音识别数据集新版、神经网络关系抽取、基于bert的命名实体识别、关键词(Keyphrase)抽取包pke、基于医疗领域知识图谱的问答系统、基于依存句法与语义角色标注的事件三元组抽取、依存句法分析4万句高质量标注数据、cnocr：用来做中文OCR的Python3包、中文人物关系知识图谱项目、中文nlp竞赛项目及代码汇总、中文字符数据、speech-aligner: 从“人声语音”及其“语言文本”产生音素级别时间对齐标注的工具、AmpliGraph: 知识图谱表示学习(Python)库：知识图谱概念链接预测、Scattertext 文本可视化(python)、语言/知识表示工具：BERT & ERNIE、中文对比英文自然语言处理NLP的区别综述、Synonyms中文近义词工具包、HarvestText领域自适应文本挖掘工具（新词发现-情感分析-实体链接等）、word2word：(Python)方便易用的多语言词-词对集：62种语言/3,564个多语言对、语音识别语料生成工具：从具有音频/字幕的在线视频创建自动语音识别(ASR)语料库、构建医疗实体识别的模型（包含词典和语料标注）、单文档非监督的关键词抽取、Kashgari中使用gpt-2语言模型、开源的金融投资数据提取工具、文本自动摘要库TextTeaser: 仅支持英文、人民日报语料处理工具集、一些关于自然语言的基本模型、基于14W歌曲知识库的问答尝试--功能包括歌词接龙and已知歌词找歌曲以及歌曲歌手歌词三角关系的问答、基于Siamese bilstm模型的相似句子判定模型并提供训练数据集和测试数据集、用Transformer编解码模型实现的根据Hacker News文章标题自动生成评论、用BERT进行序列标记和文本分类的模板代码、LitBank：NLP数据集——支持自然语言处理和计算人文学科任务的100部带标记英文小说语料、百度开源的基准信息抽取系统、虚假新闻数据集、Facebook: LAMA语言模型分析，提供Transformer-XL/BERT/ELMo/GPT预训练语言模型的统一访问接口、CommonsenseQA：面向常识的英文QA挑战、中文知识图谱资料、数据及工具、各大公司内部里大牛分享的技术文档 PDF 或者 PPT、自然语言生成SQL语句（英文）、中文NLP数据增强（EDA）工具、英文NLP数据增强工具 、基于医药知识图谱的智能问答系统、京东商品知识图谱、基于mongodb存储的军事领域知识图谱问答项目、基于远监督的中文关系抽取、语音情感分析、中文ULMFiT-情感分析-文本分类-语料及模型、一个拍照做题程序、世界各国大规模人名库、一个利用有趣中文语料库 qingyun 训练出来的中文聊天机器人、中文聊天机器人seqGAN、省市区镇行政区划数据带拼音标注、教育行业新闻语料库包含自动文摘功能、开放了对话机器人-知识图谱-语义理解-自然语言处理工具及数据、中文知识图谱：基于百度百科中文页面-抽取三元组信息-构建中文知识图谱、masr: 中文语音识别-提供预训练模型-高识别率、Python音频数据增广库、中文全词覆盖BERT及两份阅读理解数据、ConvLab：开源多域端到端对话系统平台、中文自然语言处理数据集、基于最新版本rasa搭建的对话系统、基于TensorFlow和BERT的管道式实体及关系抽取、一个小型的证券知识图谱/知识库、复盘所有NLP比赛的TOP方案、OpenCLaP：多领域开源中文预训练语言模型仓库、UER：基于不同语料+编码器+目标任务的中文预训练模型仓库、中文自然语言处理向量合集、基于金融-司法领域(兼有闲聊性质)的聊天机器人、g2pC：基于上下文的汉语读音自动标记模块、Zincbase 知识图谱构建工具包、诗歌质量评价/细粒度情感诗歌语料库、快速转化「中文数字」和「阿拉伯数字」、百度知道问答语料库、基于知识图谱的问答系统、jieba_fast 加速版的jieba、正则表达式教程、中文阅读理解数据集、基于BERT等最新语言模型的抽取式摘要提取、Python利用深度学习进行文本摘要的综合指南、知识图谱深度学习相关资料整理、维基大规模平行文本语料、StanfordNLP 0.2.0：纯Python版自然语言处理包、NeuralNLP-NeuralClassifier：腾讯开源深度学习文本分类工具、端到端的封闭域对话系统、中文命名实体识别：NeuroNER vs. BertNER、新闻事件线索抽取、2019年百度的三元组抽取比赛：“科学空间队”源码、基于依存句法的开放域文本知识三元组抽取和知识库构建、中文的GPT2训练代码、ML-NLP - 机器学习(Machine Learning)NLP面试中常考到的知识点和代码实现、nlp4han:中文自然语言处理工具集(断句/分词/词性标注/组块/句法分析/语义分析/NER/N元语法/HMM/代词消解/情感分析/拼写检查、XLM：Facebook的跨语言预训练语言模型、用基于BERT的微调和特征提取方法来进行知识图谱百度百科人物词条属性抽取、中文自然语言处理相关的开放任务-数据集-当前最佳结果、CoupletAI - 基于CNN+Bi-LSTM+Attention 的自动对对联系统、抽象知识图谱、MiningZhiDaoQACorpus - 580万百度知道问答数据挖掘项目、brat rapid annotation tool: 序列标注工具、大规模中文知识图谱数据：1.4亿实体、数据增强在机器翻译及其他nlp任务中的应用及效果、allennlp阅读理解:支持多种数据和模型、PDF表格数据提取工具 、 Graphbrain：AI开源软件库和科研工具，目的是促进自动意义提取和文本理解以及知识的探索和推断、简历自动筛选系统、基于命名实体识别的简历自动摘要、中文语言理解测评基准，包括代表性的数据集&基准模型&语料库&排行榜、树洞 OCR 文字识别 、从包含表格的扫描图片中识别表格和文字、语声迁移、Python口语自然语言处理工具集(英文)、 similarity：相似度计算工具包，java编写、海量中文预训练ALBERT模型 、Transformers 2.0 、基于大规模音频数据集Audioset的音频增强 、Poplar：网页版自然语言标注工具、图片文字去除，可用于漫画翻译 、186种语言的数字叫法库、Amazon发布基于知识的人-人开放领域对话数据集 、中文文本纠错模块代码、繁简体转换 、 Python实现的多种文本可读性评价指标、类似于人名/地名/组织机构名的命名体识别数据集 、东南大学《知识图谱》研究生课程(资料)、. 英文拼写检查库 、 wwsearch是企业微信后台自研的全文检索引擎、CHAMELEON：深度学习新闻推荐系统元架构 、 8篇论文梳理BERT相关模型进展与反思、DocSearch：免费文档搜索引擎、 LIDA：轻量交互式对话标注工具 、aili - the fastest in-memory index in the East 东半球最快并发索引 、知识图谱车音工作项目、自然语言生成资源大全 、中日韩分词库mecab的Python接口库、中文文本摘要/关键词提取、汉字字符特征提取器 (featurizer)，提取汉字的特征（发音特征、字形特征）用做深度学习的特征、中文生成任务基准测评 、中文缩写数据集、中文任务基准测评 - 代表性的数据集-基准(预训练)模型-语料库-baseline-工具包-排行榜、PySS3：面向可解释AI的SS3文本分类器机器可视化工具 、中文NLP数据集列表、COPE - 格律诗编辑程序、doccano：基于网页的开源协同多语言文本标注工具 、PreNLP：自然语言预处理库、简单的简历解析器，用来从简历中提取关键信息、用于中文闲聊的GPT2模型：GPT2-chitchat、基于检索聊天机器人多轮响应选择相关资源列表(Leaderboards、Datasets、Papers)、(Colab)抽象文本摘要实现集锦(教程 、词语拼音数据、高效模糊搜索工具、NLP数据增广资源集、微软对话机器人框架 、 GitHub Typo Corpus：大规模GitHub多语言拼写错误/语法错误数据集、TextCluster：短文本聚类预处理模块 Short text cluster、面向语音识别的中文文本规范化、BLINK：最先进的实体链接库、BertPunc：基于BERT的最先进标点修复模型、Tokenizer：快速、可定制的文本词条化库、中文语言理解测评基准，包括代表性的数据集、基准(预训练)模型、语料库、排行榜、spaCy 医学文本挖掘与信息提取 、 NLP任务示例项目代码集、 python拼写检查库、chatbot-list - 行业内关于智能客服、聊天机器人的应用和架构、算法分享和介绍、语音质量评价指标(MOSNet, BSSEval, STOI, PESQ, SRMR)、 用138GB语料训练的法文RoBERTa预训练语言模型 、BERT-NER-Pytorch：三种不同模式的BERT中文NER实验、无道词典 - 有道词典的命令行版本，支持英汉互查和在线查询、2019年NLP亮点回顾、 Chinese medical dialogue data 中文医疗对话数据集 、最好的汉字数字(中文数字)-阿拉伯数字转换工具、 基于百科知识库的中文词语多词义/义项获取与特定句子词语语义消歧、awesome-nlp-sentiment-analysis - 情感分析、情绪原因识别、评价对象和评价词抽取、LineFlow：面向所有深度学习框架的NLP数据高效加载器、中文医学NLP公开资源整理 、MedQuAD：(英文)医学问答数据集、将自然语言数字串解析转换为整数和浮点数、Transfer Learning in Natural Language Processing (NLP) 、面向语音识别的中文/英文发音辞典、Tokenizers：注重性能与多功能性的最先进分词器、CLUENER 细粒度命名实体识别 Fine Grained Named Entity Recognition、 基于BERT的中文命名实体识别、中文谣言数据库、NLP数据集/基准任务大列表、nlp相关的一些论文及代码, 包括主题模型、词向量(Word Embedding)、命名实体识别(NER)、文本分类(Text Classificatin)、文本生成(Text Generation)、文本相似性(Text Similarity)计算等，涉及到各种与nlp相关的算法，基于keras和tensorflow 、Python文本挖掘/NLP实战示例、 Blackstone：面向非结构化法律文本的spaCy pipeline和NLP模型通过同义词替换实现文本“变脸” 、中文 预训练 ELECTREA 模型: 基于对抗学习 pretrain Chinese Model 、albert-chinese-ner - 用预训练语言模型ALBERT做中文NER 、基于GPT2的特定主题文本生成/文本增广、开源预训练语言模型合集、多语言句向量包、编码、标记和实现：一种可控高效的文本生成方法、 英文脏话大列表 、attnvis：GPT2、BERT等transformer语言模型注意力交互可视化、CoVoST：Facebook发布的多语种语音-文本翻译语料库，包括11种语言(法语、德语、荷兰语、俄语、西班牙语、意大利语、土耳其语、波斯语、瑞典语、蒙古语和中文)的语音、文字转录及英文译文、Jiagu自然语言处理工具 - 以BiLSTM等模型为基础，提供知识图谱关系抽取 中文分词 词性标注 命名实体识别 情感分析 新词发现 关键词 文本摘要 文本聚类等功能、用unet实现对文档表格的自动检测，表格重建、NLP事件提取文献资源列表 、 金融领域自然语言处理研究资源大列表、CLUEDatasetSearch - 中英文NLP数据集：搜索所有中文NLP数据集，附常用英文NLP数据集 、medical_NER - 中文医学知识图谱命名实体识别 、(哈佛)讲因果推理的免费书、知识图谱相关学习资料/数据集/工具资源大列表、Forte：灵活强大的自然语言处理pipeline工具集 、Python字符串相似性算法库、PyLaia：面向手写文档分析的深度学习工具包、TextFooler：针对文本分类/推理的对抗文本生成模块、Haystack：灵活、强大的可扩展问答(QA)框架、中文关键短语抽取工具**。

<!-- 
| 资源名（Name）      | 描述（Description） | 链接     |
| :---        |    :---  |          :--- |
|       |        |    |
|       |        |    |
|       |        |    |
|       |        |    |
|       |        |    |
|       |        |    |
|       |        |    |
|       |        |    |
|       |        |    |
|       |        |    |
|       |        |    |
|       |        |    |
|       |        |    |
|       |        |    |
|       |        |    |
|       |        |    | -->


<!-- <img align="right" src="https://github-readme-stats.vercel.app/api?username=fighting41love&show_icons=true&icon_color=CE1D2D&text_color=718096&bg_color=ffffff&hide_title=true" /> -->