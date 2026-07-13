## 7.2

相关网址




报告类。这个网址是**美洲开发银行出版物目录** ，这是美洲开发银行的开放获取资源库。可以从里面下载报告，但是报告很多，要下载特定产业相关的报告，需要通过搜索+爬虫。

[techreport.com](https://techreport.com/)

新闻类。AI、芯片、软件、网络安全、机器人/硬件相关的科技新闻，建议爬虫

[www.nationalacademies.org/units/CAST-CAST%20EO-25-P-696/publications](https://www.nationalacademies.org/units/CAST-CAST%20EO-25-P-696/publications)

报告类。美国国家科学院体系下 CAST 出版物，报告/研讨会/共识研究

[thefusionreport.com/china-east-gets-set-for-ignition-in-2027](https://thefusionreport.com/china-east-gets-set-for-ignition-in-2027/)

新闻类。未来能源；核聚变；兼涉未来材料、超导磁体、等离子体、先进制造，文章形式，爬虫

[www.fusionindustryassociation.org](https://www.fusionindustryassociation.org/)

[setr.stanford.edu](https://setr.stanford.edu/)

报告类。斯坦福 Emerging Technology Review，权威技术综述，已下载

[www.neuroba.com/blog/categories/brain-computer-interfaces](https://www.neuroba.com/blog/categories/brain-computer-interfaces)

新闻类。脑机接口的相关信息报道

[tesorb.com/bci-companies-comparison-2026](https://tesorb.com/bci-companies-comparison-2026/)

新闻类。主要是几个企业的报道，分别是特斯拉，脑机接口企业，

[www.gao.gov/reports-testimonies](https://www.gao.gov/reports-testimonies)

报告类。美国 GAO 报告与国会证词，需要下载pdf报告

[www.hydrogenfuelnews.com](https://www.hydrogenfuelnews.com/)

新闻类。氢能信息网站，最早尝试爬虫的网站，很好爬取

[www.dnv.com/energy-transition-outlook](https://www.dnv.com/energy-transition-outlook/)

报告类。DNV 能源转型年度报告和数据，有一千多个pdf报告，主要是能源相关，需要下载报告。

[www.ark-invest.com](https://www.ark-invest.com/)

已下载

[www.dnv.com/energy-transition-outlook](https://www.dnv.com/energy-transition-outlook/)


## 7.3

1.根据斯坦福的报告中，量子信息部分，得到相关的关键词，然后基于关键词写识别代码

2.根据quant_firm代码，识别企业，可以识别企业

但是存在问题是，跑的太慢，需要对比是否要进行粗筛的对比。



## 7.5

目前用报告让gpt得到了量子信息的关键词，然后得到了其他五个技术的关键词，但是其他五个技术由于报告中没有直接相关的章节。所以只是让gpt自身给出了关键词。

后面的思路是：

1.需要优化关键词词典，首先是基于量子信息的关键词词典识别得到的专利，随机抽取几百个，让人工智能判断是否符合。然后优化

2.然后是对比两种方式得到的关键词词典。
