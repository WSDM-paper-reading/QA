### Categories of Modern QA system

- 1 factoid question

 -- Information queries about fact of entities
 
 -- Competitions
 
- 2 narrative question

 -- Opinion,instruction (how–to question)
- 3 multi-modal

-- Visual qa

-- Travel assistant

- 4 AI ability tests

 -- Reading comprehension
 
 -- Elementary school science and math
 
### Data sources

- 1 structured data

 -- Databases & knowledge bases
 
- 2 semi-structured data

 -- Web tables
 
- 3 unstructured text

 -- Newswire corpora
 
 -- web
 
# papers
##  QA from structured data
### datasets
freebase
microsoft satori
### papers
- semantic parsing on freebase  from question answer pair (emnlp 2013)
- semantic parsing via paraphrasing (acl 2014)
- large-scale semantic parsing without Question-answer pairs (tacl 2014)
- knowledge-based question answer as machine translation (acl 2014)
- semantic parsing via staged query graph generation:Question answer wit knowledge base (acl 2015)
- information extraction over structure data: question answer with freebase(acl 2014)
- question answer with subgraph embeddings(emnlp 2014)
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
- open domain question and answer via semantic enrichment(www 2015)
- table cell search for question answer [www 2016]
 ---
## Question answer for testing machine intelligence
### datasets
- MCtest
- Facebook bAbi
- Quiz Bowl
- visual QA
- Squad
- MS MARCO
### paper list
- memery network(iclr 2015)
- reasoning in vector space(iclr 2016)
- R-NET: Machine Reading Comprehension with Self-matching Networks[[paper]](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) [[code_tf]](https://github.com/YerevaNN/R-NET-in-Keras)[[ppt]](./ppt/R_net_lixinsu.pptx)
- LEARNING RECURRENT SPAN REPRESENTATIONS FOR EXTRACTIVE QUESTION ANSWERING[[paper]](https://arxiv.org/pdf/1611.01436.pdf)[[paper_v1]](https://openreview.net/pdf?id=HkIQH7qel) [[code_1]](https://github.com/shimisalant/RaSoR)[[code_2]](https://github.com/hsgodhia/squad_rasor_nn)[[ppt]](./ppt/paperreading_20170914_yihanni.pdf)

`This paper focused on the extractive question answering on the SQUAD dataset, presenting a novel neural architecture called RASOR .The core of this model relies on a recurrent network that enables shared computation for the shared substructure across span candidates. They explored different methods of encoding the passage and question, showing the benefits of including both passage-independent and passage-aligned question representations.`
- ReasoNet: Learning to Stop Reading in Machine Comprehension
- Machine Comprehension Using Match-LSTM and Answer Pointer
- Making Neural QA as Simple as Possible but not Simpler	
- Bidirectional Attention Flow for Machine Comprehension
- MEMEN: Multi-layer Embedding with Memory Networks for Machine Comprehension
- Mnemonic Reader: Machine Comprehension with Iterative Aligning and Multi-hop Answer Pointing
- Structural Embedding of Syntactic Trees for Machine Comprehension
- Bidirectional Attention Flow for Machine Comprehension



