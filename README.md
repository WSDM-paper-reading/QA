### related paper    
   
### 2017 ACL  QA papers  
- [Gated Self-Matching Networks for Reading Comprehension and Question Answering](http://www.aclweb.org/anthology/P/P17/P17-1018.pdf)   
- [Generating Natural Answers by Incorporating Copying and Retrieving Mechanisms in Sequence-to-Sequence Learning](http://www.aclweb.org/anthology/P/P17/P17-1019.pdf)
- [Coarse-to-Fine Question Answering for Long Documents](http://www.aclweb.org/anthology/P/P17/P17-1020.pdf)
- [An End-to-End Model for Question Answering over Knowledge Base with Cross-Attention Combining Global Knowledge](http://www.aclweb.org/anthology/P/P17/P17-1021.pdf)
- [Attention-over-Attention Neural Networks for Reading Comprehension](http://www.aclweb.org/anthology/P/P17/P17-1055.pdf)
- [Evaluation Metrics for Machine Reading Comprehension: Prerequisite Skills and Readability](http://www.aclweb.org/anthology/P/P17/P17-1075.pdf)
- [Semi-Supervised QA with Generative Domain-Adaptive Nets](http://www.aclweb.org/anthology/P/P17/P17-1096.pdf)
- [Learning to Ask: Neural Question Generation for Reading Comprehension](http://www.aclweb.org/anthology/P/P17/P17-1123.pdf)
- [A Constituent-Centric Neural Architecture for Reading Comprehension](http://www.aclweb.org/anthology/P/P17/P17-1129.pdf)
- [Leveraging Knowledge Bases in LSTMs for Improving Machine Reading](http://www.aclweb.org/anthology/P/P17/P17-1132.pdf)
- [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](http://www.aclweb.org/anthology/P/P17/P17-1147.pdf)
- [Search-based Neural Structured Learning for Sequential Question Answering](http://www.aclweb.org/anthology/P/P17/P17-1167.pdf)
- [Gated-Attention Readers for Text Comprehension](http://www.aclweb.org/anthology/P/P17/P17-1168.pdf)
- [Reading Wikipedia to Answer Open-Domain Questions](http://www.aclweb.org/anthology/P/P17/P17-1171.pdf)

#### new: QA survey [ppt](./ppt/span_QA.pptx) and [excel](./ppt/2017-09-19.xls) 

### Categories of Modern QA system

- 1 factoid question

- 2 narrative question(Opinion,instruction (how–to question))

- 3 multi-modal(Visual qa, Travel assistant)

- 4 AI ability tests(Reading comprehension,Elementary school science and math)
 
### Data sources

- 1 structured data(Databases & knowledge bases)
 
- 2 semi-structured data(Web tables)
 
- 3 unstructured text(Newswire corpora, web)

### data sets
#### english . 
- [web QA](http://idl.baidu.com/WebQA.html) ：WebQA is a large scale *Chinese* human annotated real-world QA dataset which contains 42k questions and 579k evidences, where an evidence is a piece of text which may contain information for answering the question.All the questions are of single-entity factoid type, which means (1) each question is a factoid question and (2) its answer i . 
nvolves only one entity (but may have multiple words).
- [wiki QA](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/) : a new publicly available set of question and sentence pairs, collected and annotated for research on open-domain question answering. Most previous work on answer sentence selection focuses on a dataset created using the TREC-QA data, which includes editor-generated questions and candidate answer sentences selected by matching content words in the question. WikiQA is constructed using a more natural process and is more than an order of magnitude larger than the previous dataset. In addition, the WikiQA dataset also includes questions for which there are no correct sentences, enabling researchers to work on answer triggering, a critical component in any QA system. We compare several systems on the task of answer sentence selection on both datasets and also describe the performance of a system on the problem of answer triggering using the WikiQA dataset.
- [squad](https://rajpurkar.github.io/SQuAD-explorer/)    
- [trivia](https://arxiv.org/abs/1705.03551)  
- [searchqa](https://arxiv.org/abs/1704.05179)  
- [quasar](https://arxiv.org/abs/1707.03904)  
- [narrativeqa](https://github.com/deepmind/narrativeqa)   
- [ms marco](http://www.msmarco.org/)
#### chinese . 
- [dureader](https://arxiv.org/abs/1711.05073)
- [WebQA](https://arxiv.org/pdf/1607.06275.pdf)
- [sougou](http://task.www.sogou.com/cips-sogou_qa/)
# papers
##  QA from structured data

datasets(freebase,microsoft satori,DBpedia)

### papers
- semantic parsing on freebase  from question answer pair (emnlp 2013)
- semantic parsing via paraphrasing (acl 2014)
- large-scale semantic parsing without Question-answer pairs (tacl 2014)
- knowledge-based question answer as machine translation (acl 2014)
- semantic parsing via staged query graph generation:Question answer wit knowledge base (acl 2015)[[paper]](http://www.aclweb.org/anthology/P15-1128)[[ppt]](./ppt/paperreading-20170914-tongleiguo.pdf)
- information extraction over structure data: question answer with freebase(acl 2014)
- question answer with subgraph embeddings(emnlp 2014)[[paper]](http://www.thespermwhale.com/jaseweston/papers/fbqa.pdf)[[ppt]](./ppt/paperreading-20170907-sihaoyu.pdf)
- limitation learning of agenda-based sematic parsers (tacl 2015)
- transforming dependncy structures to logical form for semantic parsing(tacl 2016)
- question answer on freebase via relation extraction and textual evidence(acl 2016)
 ---
## web-based question and answering 
### papers
- Entity linking and retrieval for semantic search(wsdm 2014)
- knowledge base completion via search-based question answering(www 2014)
- learning question classifiers (coling 2012)
- question answer (Dan jurafsky  stanford book,chapter 28) 
- open domain question and answer via semantic enrichment(www 2015)[[paper]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1068-sunA.pdf)[[ppt]](./ppt/paperreading-20170907-jianguichen.pdf)
- table cell search for question answer [www 2016]
 ---
## Question answer for testing machine intelligence

datasets(Facebook bAbi,Squad,MS MARCO,[Baidu ild webqa](), [trivia](https://homes.cs.washington.edu/~eunsol/papers/acl17jcwz.pdf) )

### paper list
- memery network(iclr 2015)[[paper]](https://arxiv.org/abs/1410.3916)
- reasoning in vector space(iclr 2016)
- R-NET: Machine Reading Comprehension with Self-matching Networks[[paper]](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) [[code_tf]](https://github.com/YerevaNN/R-NET-in-Keras)[[ppt]](./ppt/paperreading_20170907_lixinsu.pdf)
- LEARNING RECURRENT SPAN REPRESENTATIONS FOR EXTRACTIVE QUESTION ANSWERING[[paper]](https://arxiv.org/pdf/1611.01436.pdf)[[paper_v1]](https://openreview.net/pdf?id=HkIQH7qel) [[code_1]](https://github.com/shimisalant/RaSoR)[[code_2]](https://github.com/hsgodhia/squad_rasor_nn)[[ppt]](./ppt/paperreading_20170914_yihanni.pdf)
- ReasoNet: Learning to Stop Reading in Machine Comprehension[[paper]](https://arxiv.org/abs/1609.05284)[[code_cntk]](https://github.com/AnatoliiPotapov/reasonet_cntk)[[ppt]](./ppt/paperreading_20170907_lijuanchen.pdf)
- Machine Comprehension Using Match-LSTM and Answer Pointer[[paper]](https://arxiv.org/abs/1608.07905)
- Making Neural QA as Simple as Possible but not Simpler	[[paper]](http://www.aclweb.org/anthology/K17-1028)
- Bidirectional Attention Flow for Machine Comprehension[[paper]](https://arxiv.org/abs/1611.01603)[[ppt]](./ppt/paperreading_20170914_yuefeng.pdf)[[code_tf]](https://github.com/allenai/bi-att-flow)
- MEMEN: Multi-layer Embedding with Memory Networks for Machine Comprehension[[paper]](https://arxiv.org/abs/1707.09098)
- Mnemonic Reader: Machine Comprehension with Iterative Aligning and Multi-hop Answer Pointing[[paper]](https://arxiv.org/abs/1705.02798)
- Structural Embedding of Syntactic Trees for Machine Comprehension




