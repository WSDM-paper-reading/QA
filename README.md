## question answering and reading comprehension   
 ---
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
### 2017 EMNLP QA papers

 --- 
### Categories of Modern QA system

- 1 factoid question . 
- 2 narrative question(Opinion,instruction (how–to question))  
- 3 multi-modal(Visual qa, Travel assistant)  
- 4 AI ability tests(Reading comprehension,Elementary school science and math)  
 
### Data sources 
- 1 structured data(Databases & knowledge bases) . 
- 2 semi-structured data(Web tables) . 
- 3 unstructured text(Newswire corpora, web) . 
 ---
### datasets
#### english  
- [wiki QA](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/)  
- [squad](https://rajpurkar.github.io/SQuAD-explorer/)    
- [trivia](https://arxiv.org/abs/1705.03551)  
- [searchqa](https://arxiv.org/abs/1704.05179)  
- [quasar](https://arxiv.org/abs/1707.03904)  
- [narrativeqa](https://github.com/deepmind/narrativeqa)   
- [ms marco](http://www.msmarco.org/)
#### chinese  
- [dureader](https://arxiv.org/abs/1711.05073)
- [WebQA](https://arxiv.org/pdf/1607.06275.pdf)
- [sougou](http://task.www.sogou.com/cips-sogou_qa/)
 ---  
### papers
####  QA from structured data
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
#### qa in web search
- Entity linking and retrieval for semantic search(wsdm 2014)
- knowledge base completion via search-based question answering(www 2014)
- learning question classifiers (coling 2012)
- question answer (Dan jurafsky  stanford book,chapter 28) 
- open domain question and answer via semantic enrichment(www 2015)[[paper]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1068-sunA.pdf)[[ppt]](./ppt/paperreading-20170907-jianguichen.pdf)
- table cell search for question answer [www 2016]
#### Question answer for testing machine intelligence
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
 ---
### new
#### QA survey [ppt](./ppt/span_QA.pptx) and [excel](./ppt/2017-09-19.xls)

QA papers in acl from 2001 ~ 2017 
Organizing Encyclopedic Knowledge based on the Web and its Application to Question Answering . 
http://aclweb.org/anthology/P/P01/P01-1026.bib

The Role of Lexico-Semantic Feedback in Open-Domain Textual Question-Answering . 
http://aclweb.org/anthology/P/P01/P01-1037.bib

Logic Form Transformation of WordNet and its Applicability to Question Answering . 
http://aclweb.org/anthology/P/P01/P01-1052.bib

Using Machine Learning Techniques to Interpret WH-questions . 
http://aclweb.org/anthology/P/P01/P01-1070.bib

Performance Issues and Error Analysis in an Open-Domain Question Answering System  
http://aclweb.org/anthology/P/P02/P02-1005.bib

Learning surface text patterns for a Question Answering System . 
http://aclweb.org/anthology/P/P02/P02-1006.bib

Is It the Right Answer? Exploiting Web Redundancy for Answer Validation . 
http://aclweb.org/anthology/P/P02/P02-1054.bib

Offline Strategies for Online Question Answering: Answering Questions Before They Are Asked . 
http://aclweb.org/anthology/P/P03/P03-1001.bib

A Noisy-Channel Approach to Question Answering . 
http://aclweb.org/anthology/P/P03/P03-1003.bib

A Speech Interface for Open-Domain Question-Answering . 
http://aclweb.org/anthology/P/P03/P03-2034.bib

Splitting Complex Temporal Questions for Question Answering Systems . 
http://aclweb.org/anthology/P/P04/P04-1072.bib

Question Answering Using Constraint Satisfaction: QA-By-Dossier-With-Contraints . 
http://aclweb.org/anthology/P/P04/P04-1073.bib

Resource Analysis for Question Answering . 
http://aclweb.org/anthology/P/P04/P04-3018.bib

Experiments with Interactive Question-Answering
http://aclweb.org/anthology/P/P05/P05-1026.bib

Question Answering as Question-Biased Term Extraction: A New Approach toward Multilingual {QA} . 
http://aclweb.org/anthology/P/P05/P05-1027.bib

Automatic Discovery of Intentions in Text and its Application to Question Answering . 
http://aclweb.org/anthology/P/P05/P05-2006.bib

Learning Strategies for Open-Domain Natural Language Question Answering . 
http://aclweb.org/anthology/P/P05/P05-2015.bib

Descriptive Question Answering in Encyclopedia  
http://aclweb.org/anthology/P/P05/P05-3006.bib

QuestionBank: Creating a Corpus of Parse-Annotated Questions . 
http://aclweb.org/anthology/P/P06/P06-1063.bib

Answer Extraction, Semantic Clustering, and Extractive Summarization for Clinical Question Answering . 
http://aclweb.org/anthology/P/P06/P06-1106.bib

Question Answering with Lexical Chains Propagating Verb Arguments . 
http://aclweb.org/anthology/P/P06/P06-1113.bib

Methods for Using Textual Entailment in Open-Domain Question Answering . 
http://aclweb.org/anthology/P/P06/P06-1114.bib

Improving QA Accuracy by Question Inversion . 
http://aclweb.org/anthology/P/P06/P06-1135.bib

Reranking Answers for Definitional QA Using Language Modeling . 
http://aclweb.org/anthology/P/P06/P06-1136.bib

Utilizing Co-Occurrence of Answers in Question Answering . 
http://aclweb.org/anthology/P/P06/P06-1147.bib

The Role of Information Retrieval in Answering Complex Questions . 
http://aclweb.org/anthology/P/P06/P06-2068.bib

{FERRET}: Interactive Question-Answering for Real-World Environments . 
http://aclweb.org/anthology/P/P06/P06-4007.bib

{K-QARD}: A Practical Korean Question Answering Framework for Restricted Domain . 
http://aclweb.org/anthology/P/P06/P06-4008.bib

Towards Conversational QA: Automatic Identification of Problematic Situations and User Intent . 
http://aclweb.org/anthology/P/P06/P06-2008.bib

Different Structures for Evaluating Answers to Complex Questions: Pyramids Won't Topple, and Neither Will Human Assessors . 
http://aclweb.org/anthology/P/P07/P07-1097.bib

Exploiting Syntactic and Shallow Semantic Kernels for Question Answer Classification . 
http://aclweb.org/anthology/P/P07/P07-1098.bib

Language-independent Probabilistic Answer Ranking for Question Answering . 
http://aclweb.org/anthology/P/P07/P07-1099.bib

Statistical Machine Translation for Query Expansion in Answer Retrieval . 
http://aclweb.org/anthology/P/P07/P07-1059.bib

Searching Questions by Identifying Question Topic and Question Focus . 
http://aclweb.org/anthology/P/P08/P08-1019.bib

Collecting a Why-Question Corpus for Development and Evaluation of an Automatic {QA}-System . 
http://aclweb.org/anthology/P/P08/P08-1051.bib

Unsupervised Discovery of Generic Relationships Using Pattern Clusters and its Evaluation by Automatically Generated {SAT} Analogy Questions . 
http://aclweb.org/anthology/P/P08/P08-1079.bib

Using Conditional Random Fields to Extract Contexts and Answers of Questions from Online Forums . 
http://aclweb.org/anthology/P/P08/P08-1081.bib

Improving the Performance of the Random Walk Model for Answering Complex Questions . 
http://aclweb.org/anthology/P/P08/P08-2003.bib

You've Got Answers: Towards Personalized Models for Predicting Success in Community Question Answering . 
http://aclweb.org/anthology/P/P08/P08-2025.bib

The {QuALiM} Question Answering Demo: Supplementing Answers with Paragraphs drawn from {Wikipedia}  
http://aclweb.org/anthology/P/P08/P08-4009.bib

Learning to Rank Answers on Large Online {QA} Collections . 
http://aclweb.org/anthology/P/P08/P08-1082.bib

Kernels on Linguistic Structures for Answer Extraction . 
http://aclweb.org/anthology/P/P08/P08-2029.bib

A Graph-based Semi-Supervised Learning for Question-Answering . 
http://aclweb.org/anthology/P/P09/P09-1081.bib

Combining Lexical Semantic Resources with Question \& Answer Archives for Translation-Based Answer Finding . 
http://aclweb.org/anthology/P/P09/P09-1082.bib

Answering Opinion Questions with Random Walks on Graphs . 
http://aclweb.org/anthology/P/P09/P09-1083.bib

Automatic Generation of Information-seeking Questions Using Concept Clusters . 
http://aclweb.org/anthology/P/P09/P09-2024.bib

Opinion and Generic Question Answering Systems: a Performance Analysis . 
http://aclweb.org/anthology/P/P09/P09-2040.bib

Learning foci for Question Answering over Topic Maps . 
http://aclweb.org/anthology/P/P09/P09-2082.bib

Do Automatic Annotation Techniques Have Any Impact on Supervised Complex Question Answering?  
http://aclweb.org/anthology/P/P09/P09-2083.bib

Where's the Verb? Correcting Machine Translation During Question Answering . 
http://aclweb.org/anthology/P/P09/P09-2084.bib

Comparable Entity Mining from Comparative Questions  
http://aclweb.org/anthology/P/P10/P10-1067.bib

Metadata-Aware Measures for Answer Summarization in Community Question Answering . 
http://aclweb.org/anthology/P/P10/P10-1078.bib

Modeling Semantic Relevance for Question-Answer Pairs in Web Social Communities . 
http://aclweb.org/anthology/P/P10/P10-1125.bib

Optimizing Question Answering Accuracy by Maximizing Log-Likelihood . 
http://aclweb.org/anthology/P/P10/P10-2044.bib

Phrase-Based Translation Model for Question Retrieval in Community Question Answer Archives . 
http://aclweb.org/anthology/P/P11/P11-1066.bib

Learning to Grade Short Answer Questions using Semantic Similarity Measures and Dependency Graph Alignments . 
http://aclweb.org/anthology/P/P11/P11-1076.bib

Improving Question Recommendation by Exploiting Information Need . 
http://aclweb.org/anthology/P/P11/P11-1143.bib

Question Detection in Spoken Conversations Using Textual Conversations . 
http://aclweb.org/anthology/P/P11/P11-2021.bib

Search in the Lost Sense of Query: Question Formulation in Web Search Queries and its Temporal Changes . 
http://aclweb.org/anthology/P/P11/P11-2024.bib

Query Snowball: A Co-occurrence-based Approach to Multi-document Summarization for Question Answering . 
http://aclweb.org/anthology/P/P11/P11-2039.bib

Sentence Dependency Tagging in Online Question Answering Forums . 
http://aclweb.org/anthology/P/P12/P12-1058.bib

Community Answer Summarization for Multi-Sentence Question with Group L1 Regularization  
http://aclweb.org/anthology/P/P12/P12-1061.bib

Automatically Mining Question Reformulation Patterns from Search Log Data . 
http://aclweb.org/anthology/P/P12/P12-2037.bib

Generating Synthetic Comparable Questions for News Articles . 
http://aclweb.org/anthology/P/P13/P13-1073.bib

Statistical Machine Translation Improves Question Retrieval in Community Question Answering via Matrix Factorization . 
http://aclweb.org/anthology/P/P13/P13-1084.bib

Paraphrase-Driven Learning for Open Question Answering . 
http://aclweb.org/anthology/P/P13/P13-1158.bib

Evaluating a City Exploration Dialogue System with Integrated Question-Answering and Pedestrian Navigation .
http://aclweb.org/anthology/P/P13/P13-1163.bib

Why-Question Answering using Intra- and Inter-Sentential Causal Relations . 
http://aclweb.org/anthology/P/P13/P13-1170.bib

Question Answering Using Enhanced Lexical Semantic Models . 
http://aclweb.org/anthology/P/P13/P13-1171.bib

Minimum Bayes Risk based Answer Re-ranking for Question Answering . 
http://aclweb.org/anthology/P/P13/P13-2075.bib

Question Classification Transfer . 
http://aclweb.org/anthology/P/P13/P13-2076.bib

Latent Semantic Tensor Indexing for Community-based Question Answering . 
http://aclweb.org/anthology/P/P13/P13-2077.bib

Question Analysis for Polish Question Answering . 
http://aclweb.org/anthology/P/P13/P13-3014.bib

PAL: A Chatterbot System for Answering Domain-specific Questions . 
http://aclweb.org/anthology/P/P13/P13-4012.bib

Multimodal DBN for Predicting High-Quality Answers in cQA portals . 
http://aclweb.org/anthology/P/P13/P13-2146.bib

Deceptive Answer Prediction with User Preference Graph . 
http://aclweb.org/anthology/P/P13/P13-1169.bib

Information Extraction over Structured Data: Question Answering with Freebase . 
http://aclweb.org/anthology/P/P14/P14-1090.bib

Knowledge-Based Question Answering as Machine Translation . 
http://aclweb.org/anthology/P/P14/P14-1091.bib

Linguistic Considerations in Automatic Question Generation . 
http://aclweb.org/anthology/P/P14/P14-2053.bib

Semantic Parsing for Single-Relation Question Answering  
http://aclweb.org/anthology/P/P14/P14-2105.bib

Discourse Complements Lexical Semantics for Non-factoid Answer Reranking  
http://aclweb.org/anthology/P/P14/P14-1092.bib

Learning Continuous Word Embedding with Metadata for Question Retrieval in Community Question Answering . 
http://aclweb.org/anthology/P/P15/P15-1025.bib

Question Answering over Freebase with Multi-Column Convolutional Neural Networks . 
http://aclweb.org/anthology/P/P15/P15-1026.bib

Deep Questions without Deep Understanding  
http://aclweb.org/anthology/P/P15/P15-1086.bib

Semantic Parsing via Staged Query Graph Generation: Question Answering with Knowledge Base . 
http://aclweb.org/anthology/P/P15/P15-1128.bib

Thread-Level Information for Comment Classification in Community Question Answering . 
http://aclweb.org/anthology/P/P15/P15-2113.bib

Learning Hybrid Representations to Retrieve Semantically Equivalent Questions . 
http://aclweb.org/anthology/P/P15/P15-2114.bib

A Long Short-Term Memory Model for Answer Sentence Selection in Question Answering . 
http://aclweb.org/anthology/P/P15/P15-2116.bib

Answer Sequence Learning with Neural Networks for Answer Selection in Community Question Answering  
http://aclweb.org/anthology/P/P15/P15-2117.bib

Automatic Identification of Rhetorical Questions . 
http://aclweb.org/anthology/P/P15/P15-2122.bib

Learning Answer-Entailing Structures for Machine Comprehension . 
http://aclweb.org/anthology/P/P15/P15-1024.bib

Machine Comprehension with Discourse Relations . 
http://aclweb.org/anthology/P/P15/P15-1121.bib

Machine Comprehension with Syntax, Frames, and Semantics . 
http://aclweb.org/anthology/P/P15/P15-2115.bib

Together we stand: Siamese Networks for Similar Question Retrieval . 
http://aclweb.org/anthology/P/P16/P16-1036.bib

Combining Natural Logic and Shallow Reasoning for Question Answering . 
http://aclweb.org/anthology/P/P16/P16-1042.bib

Easy Questions First? A Case Study on Curriculum Learning for Question Answering . 
http://aclweb.org/anthology/P/P16/P16-1043.bib

Improved Representation Learning for Question Answer Matching . 
http://aclweb.org/anthology/P/P16/P16-1044.bib

Tables as Semi-structured Knowledge for Question Answering . 
http://aclweb.org/anthology/P/P16/P16-1045.bib

Generating Factoid Questions With Recurrent Neural Networks: The 30M Factoid Question-Answer Corpus . 
http://aclweb.org/anthology/P/P16/P16-1056.bib

CFO: Conditional Focused Neural Question Answering with Large-scale Knowledge Bases . 
http://aclweb.org/anthology/P/P16/P16-1076.bib

Generating Natural Questions About an Image . 
http://aclweb.org/anthology/P/P16/P16-1170.bib

Question Answering on Freebase via Relation Extraction and Textual Evidence . 
http://aclweb.org/anthology/P/P16/P16-1220.bib

The Value of Semantic Parse Labeling for Knowledge Base Question Answering . 
http://aclweb.org/anthology/P/P16/P16-2033.bib

Annotating Relation Inference in Context via Question Answering . 
http://aclweb.org/anthology/P/P16/P16-2041.bib

Machine Translation Evaluation Meets Community Question Answering . 
http://aclweb.org/anthology/P/P16/P16-2075.bib

Science Question Answering using Instructional Materials . 
http://aclweb.org/anthology/P/P16/P16-2076.bib

QA-It: Classifying Non-Referential It for Question Answer Pairs . 
http://aclweb.org/anthology/P/P16/P16-3020.bib

Text Understanding with the Attention Sum Reader Network . 
http://aclweb.org/anthology/P/P16/P16-1086.bib

WikiReading: A Novel Large-scale Language Understanding Task over Wikipedia . 
http://aclweb.org/anthology/P/P16/P16-1145.bib

A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task . 
http://aclweb.org/anthology/P/P16/P16-1223.bib

Gated Self-Matching Networks for Reading Comprehension and Question Answering .  
http://aclweb.org/anthology/P/P17/P17-1018.bib . 
In this paper, we present the gated self-matching networks for reading comprehension style question answering, which aims to answer questions from a given passage. We first match the question and passage with gated attention-based recurrent networks to obtain the question-aware passage representation. Then we propose a self-matching attention mechanism to refine the representation by matching the passage against itself, which effectively encodes information from the whole passage. We finally employ the pointer networks to locate the positions of answers from the passages. We conduct extensive experiments on the SQuAD dataset. The single model achieves 71.3% on the evaluation metrics of exact match on the hidden test set, while the ensemble model further boosts the results to 75.9%. At the time of submission of the paper, our model holds the first place on the SQuAD leaderboard for both single and ensemble model.  
Attention-over-Attention Neural Networks for Reading Comprehension . 
http://aclweb.org/anthology/P/P17/P17-1055.bib  
Cloze-style reading comprehension is a representative problem in mining relationship between document and query. In this paper, we present a simple but novel model called attention-over-attention reader for better solving cloze-style reading comprehension task. The proposed model aims to place another attention mechanism over the document-level attention and induces attended attention for final answer predictions. One advantage of our model is that it is simpler than related works while giving excellent performance. In addition to the primary model, we also propose an N-best re-ranking strategy to double check the validity of the candidates and further improve the performance. Experimental results show that the proposed methods significantly outperform various state-of-the-art systems by a large margin in public datasets, such as CNN and Children's Book Test.  
Evaluation Metrics for Machine Reading Comprehension: Prerequisite Skills and Readability . 
http://aclweb.org/anthology/P/P17/P17-1075.bib . 
Knowing the quality of reading comprehension (RC) datasets is important for the development of natural-language understanding systems. In this study, two classes of metrics were adopted for evaluating RC datasets: prerequisite skills and readability. We applied these classes to six existing datasets, including MCTest and SQuAD, and highlighted the characteristics of the datasets according to each metric and the correlation between the two classes. Our dataset analysis suggests that the readability of RC datasets does not directly affect the question difficulty and that it is possible to create an RC dataset that is easy to read but difficult to answer.  
Apples to Apples: Learning Semantics of Common Entities Through a Novel Comprehension Task . 
http://aclweb.org/anthology/P/P17/P17-1084.bib . 
Understanding common entities and their attributes is a primary requirement for any system that comprehends natural language. In order to enable learning about common entities, we introduce a novel machine comprehension task, GuessTwo: given a short paragraph comparing different aspects of two real-world semantically-similar entities, a system should guess what those entities are. Accomplishing this task requires deep language understanding which enables inference, connecting each comparison paragraph to different levels of knowledge about world entities and their attributes. So far we have crowdsourced a dataset of more than 14K comparison paragraphs comparing entities from a variety of categories such as fruits and animals. We have designed two schemes for evaluation: open-ended, and binary-choice prediction. For benchmarking further progress in the task, we have collected a set of paragraphs as the test set on which human can accomplish the task with an accuracy of 94.2\% on open-ended prediction. We have implemented various models for tackling the task, ranging from semantic-driven to neural models. The semantic-driven approach outperforms the neural models, however, the results indicate that the task is very challenging across the models.  
Learning to Ask: Neural Question Generation for Reading Comprehension . 
http://aclweb.org/anthology/P/P17/P17-1123.bib . 
We study automatic question generation for sentences from text passages in reading comprehension. We introduce an attention-based sequence learning model for the task and investigate the effect of encoding sentence- vs. paragraph-level information. In contrast to all previous work, our model does not rely on hand-crafted rules or a sophisticated NLP pipeline; it is instead trainable end-to-end via sequence-to-sequence learning. Automatic evaluation results show that our system significantly outperforms the state-of-the-art rule-based system. In human evaluations, questions generated by our system are also rated as being more natural (\ie, grammaticality, fluency) and as more difficult to answer (in terms of syntactic and lexical divergence from the original text and reasoning needed to answer).  
Can Syntax Help? Improving an LSTM-based Sentence Compression Model for New Domains . 
http://aclweb.org/anthology/P/P17/P17-1127.bib . 
In this paper, we study how to improve the domain adaptability of a deletion-based Long Short-Term Memory (LSTM) neural network model for sentence compression. We hypothesize that syntactic information helps in making such models more robust across domains. We propose two major changes to the model: using explicit syntactic features and introducing syntactic constraints through Integer Linear Programming (ILP). Our evaluation shows that the proposed model works better than the original model as well as a traditional non-neural-network-based model in a cross-domain setting.  
A Constituent-Centric Neural Architecture for Reading Comprehension . 
http://aclweb.org/anthology/P/P17/P17-1129.bib . 
Reading comprehension (RC), aiming to understand natural texts and answer questions therein, is a challenging task. In this paper, we study the RC problem on the Stanford Question Answering Dataset (SQuAD). Observing from the training set that most correct answers are centered around constituents in the parse tree, we design a constituent-centric neural architecture where the generation of candidate answers and their representation learning are both based on constituents and guided by the parse tree. Under this architecture, the search space of candidate answers can be greatly reduced without sacrificing the coverage of correct answers and the syntactic, hierarchical and compositional structure among constituents can be well captured, which contributes to better representation learning of the candidate answers. On SQuAD, our method achieves the state of the art performance and the ablation study corroborates the effectiveness of individual modules.  
Leveraging Knowledge Bases in LSTMs for Improving Machine Reading . 
http://aclweb.org/anthology/P/P17/P17-1132.bib . 
This paper focuses on how to take advantage of external knowledge bases (KBs) to improve recurrent neural networks for machine reading. Traditional methods that exploit knowledge from KBs encode knowledge as discrete indicator features. Not only do these features generalize poorly, but they require task-specific feature engineering to achieve good performance. We propose KBLSTM, a novel neural model that leverages continuous representations of KBs to enhance the learning of recurrent neural networks for machine reading. To effectively integrate background knowledge with information from the currently processed text, our model employs an attention mechanism with a sentinel to adaptively decide whether to attend to background knowledge and which information from KBs is useful. Experimental results show that our model achieves accuracies that surpass the previous state-of-the-art results for both entity extraction and event extraction on the widely used ACE2005 dataset.
TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension . 
http://aclweb.org/anthology/P/P17/P17-1147.bib . 
We present TriviaQA, a challenging reading comprehension dataset containing over 650K question-answer-evidence triples. TriviaQA includes 95K question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per question on average, that provide high quality distant supervision for answering the questions. We show that, in comparison to other recently introduced large-scale datasets, TriviaQA (1) has relatively complex, compositional questions, (2) has considerable syntactic and lexical variability between questions and corresponding answer-evidence sentences, and (3) requires more cross sentence reasoning to find answers. We also present two baseline algorithms: a feature-based classifier and a state-of-the-art neural network, that performs well on SQuAD reading comprehension. Neither approach comes close to human performance (23% and 40% vs. 80%), suggesting that TriviaQA is a challenging testbed that is worth significant future study.  
Gated-Attention Readers for Text Comprehension . 
http://aclweb.org/anthology/P/P17/P17-1168.bib . 
In this paper we study the problem of answering cloze-style questions over documents. Our model, the Gated-Attention (GA) Reader, integrates a multi-hop architecture with a novel attention mechanism, which is based on multiplicative interactions between the query embedding and the intermediate states of a recurrent neural network document reader. This enables the reader to build query-specific representations of tokens in the document for accurate answer selection. The GA Reader obtains state-of-the-art results on three benchmarks for this task--the CNN \& Daily Mail news stories and the Who Did What dataset. The effectiveness of multiplicative interaction is demonstrated by an ablation study, and by comparing to alternative compositional operators for implementing the gated-attention.  
Coarse-to-Fine Question Answering for Long Documents . 
http://aclweb.org/anthology/P/P17/P17-1020.bib . 
We present a framework for question answering that can efficiently scale to longer documents while maintaining or even improving performance of state-of-the-art models. While most successful approaches for reading comprehension rely on recurrent neural networks (RNNs), running them over long documents is prohibitively slow because it is difficult to parallelize over sequences. Inspired by how people first skim the document, identify relevant parts, and carefully read these parts to produce an answer, we combine a coarse, fast model for selecting relevant sentences and a more expensive RNN for producing the answer from those sentences. We treat sentence selection as a latent variable trained jointly from the answer only using reinforcement learning. Experiments demonstrate state-of-the-art performance on a challenging subset of the WikiReading dataset and on a new dataset, while speeding up the model by 3.5x-6.7x.  
An End-to-End Model for Question Answering over Knowledge Base with Cross-Attention Combining Global Knowledge . 
http://aclweb.org/anthology/P/P17/P17-1021.bib . 
With the rapid growth of knowledge bases (KBs) on the web, how to take full advantage of them becomes increasingly important. Question answering over knowledge base (KB-QA) is one of the promising approaches to access the substantial knowledge. Meanwhile, as the neural network-based (NN-based) methods develop, NN-based KB-QA has already achieved impressive results. However, previous work did not put more emphasis on question representation, and the question is converted into a fixed vector regardless of its candidate answers. This simple representation strategy is not easy to express the proper information in the question. Hence, we present an end-to-end neural network model to represent the questions and their corresponding scores dynamically according to the various candidate answer aspects via cross-attention mechanism. In addition, we leverage the global knowledge inside the underlying KB, aiming at integrating the rich KB information into the representation of the answers. As a result, it could alleviates the out-of-vocabulary (OOV) problem, which helps the cross-attention model to represent the question more precisely. The experimental results on WebQuestions demonstrate the effectiveness of the proposed approach.  
Improved Neural Relation Detection for Knowledge Base Question Answering . 
http://aclweb.org/anthology/P/P17/P17-1053.bib . 
Relation detection is a core component of many NLP applications including Knowledge Base Question Answering (KBQA). In this paper, we propose a hierarchical recurrent neural network enhanced by residual learning which detects KB relations given an input question. Our method uses deep residual bidirectional LSTMs to compare questions and relation names via different levels of abstraction. Additionally, we propose a simple KBQA system that integrates entity linking and our proposed relation detector to make the two components enhance each other. Our experimental results show that our approach not only achieves outstanding relation detection performance, but more importantly, it helps our KBQA system achieve state-of-the-art accuracy for both single-relation (SimpleQuestions) and multi-relation (WebQSP) QA benchmarks.  
Search-based Neural Structured Learning for Sequential Question Answering . 
http://aclweb.org/anthology/P/P17/P17-1167.bib . 
Recent work in semantic parsing for question answering has focused on long and complicated questions, many of which would seem unnatural if asked in a normal conversation between two humans. In an effort to explore a conversational QA setting, we present a more realistic task: answering sequences of simple but inter-related questions. We collect a dataset of 6,066 question sequences that inquire about semi-structured tables from Wikipedia, with 17,553 question-answer pairs in total. To solve this sequential question answering task, we propose a novel dynamic neural semantic parsing framework trained using a weakly supervised reward-guided search. Our model effectively leverages the sequential context to outperform state-of-the-art QA systems that are designed to answer highly complex questions.  
Reading Wikipedia to Answer Open-Domain Questions . 
http://aclweb.org/anthology/P/P17/P17-1171.bib . 
This paper proposes to tackle open-domain question answering using Wikipedia as the unique knowledge source: the answer to any factoid question is a text span in a Wikipedia article. This task of machine reading at scale combines the challenges of document retrieval (finding the relevant articles) with that of machine comprehension of text (identifying the answer spans from those articles). Our approach combines a search component based on bigram hashing and TF-IDF matching with a multi-layer recurrent neural network model trained to detect answers in Wikipedia paragraphs. Our experiments on multiple existing QA datasets indicate that (1) both modules are highly competitive with respect to existing counterparts and (2) multitask learning using distant supervision on their combination is an effective complete system on this challenging task.
Answering Complex Questions Using Open Information Extraction . 
http://aclweb.org/anthology/P/P17/P17-2049.bib . 
While there has been substantial progress in factoid question-answering (QA), answering complex questions remains challenging, typically requiring both a large body of knowledge and inference techniques. Open Information Extraction (Open IE) provides a way to generate semi-structured knowledge for QA, but to date such knowledge has only been used to answer simple questions with retrieval-based methods. We overcome this limitation by presenting a method for reasoning with Open IE knowledge, allowing more complex questions to be handled. Using a recently proposed support graph optimization framework for QA, we develop a new inference model for Open IE, in particular one that can work effectively with multiple short facts, noise, and the relational structure of tuples. Our model significantly outperforms a state-of-the-art structured solver on complex questions of varying difficulty, while also removing the reliance on manually curated knowledge.  
Group Sparse CNNs for Question Classification with Answer Sets . 
http://aclweb.org/anthology/P/P17/P17-2053.bib . 
Question classification is an important task with wide applications. However, traditional techniques treat questions as general sentences, ignoring the corresponding answer data. In order to consider answer information into question modeling, we first introduce novel group sparse autoencoders which refine question representation by utilizing group information in the answer set. We then propose novel group sparse CNNs which naturally learn question representation with respect to their answers by implanting group sparse autoencoders into traditional CNNs. The proposed model significantly outperform strong baselines on four datasets.  
Question Answering on Knowledge Bases and Text using Universal Schema and Memory Networks . 
http://aclweb.org/anthology/P/P17/P17-2057.bib . 
Existing question answering methods infer answers either from a knowledge base or from raw text. While knowledge base (KB) methods are good at answering compositional questions, their performance is often affected by the incompleteness of the KB. Au contraire, web text contains millions of facts that are absent in the KB, however in an unstructured form. Universal schema can support reasoning on the union of both structured KBs and unstructured text by aligning them in a common embedded space. In this paper we extend universal schema to natural language question answering, employing Memory networks to attend to the large body of facts in the combination of text and KB. Our models can be trained in an end-to-end fashion on question-answer pairs. Evaluation results on Spades fill-in-the-blank question answering dataset show that exploiting universal schema for question answering is better than using either a KB or text alone. This model also outperforms the current state-of-the-art by 8.5 F1 points.  
Question Answering through Transfer Learning from Large Fine-grained Supervision Data . 
http://aclweb.org/anthology/P/P17/P17-2081.bib . 
We show that the task of question answering (QA) can significantly benefit from the transfer learning of models trained on a different large, fine-grained QA dataset. We achieve the state of the art in two well-studied QA datasets, WikiQA and SemEval-2016 (Task 3A), through a basic transfer learning technique from SQuAD. For WikiQA, our model outperforms the previous best model by more than 8%. We demonstrate that finer supervision provides better guidance for learning lexical and syntactic information than coarser supervision, through quantitative results and visual analysis. We also show that a similar transfer learning procedure achieves the state of the art on an entailment task.  
Are You Asking the Right Questions? Teaching Machines to Ask Clarification Questions . 
http://aclweb.org/anthology/P/P17/P17-3006.bib . 
