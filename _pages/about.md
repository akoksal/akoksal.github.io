---
permalink: /
title: ""
# excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
I am a final-year ELLIS PhD student at [CIS](https://www.cis.uni-muenchen.de/) at LMU Munich and [LTL](https://ltl.mmll.cam.ac.uk/) at the University of Cambridge, supervised by [Prof. Hinrich Schütze](https://www.cis.uni-muenchen.de/schuetze/) and [Prof. Anna Korhonen](https://www.cl.cam.ac.uk/~alk23/). I've had the opportunity to complete research internships at Google in Mountain View and Amazon in Madrid.

My research focuses on improving LLM capabilities through *effective data utilization* and *synthetic dataset generation*, with a particular emphasis on *corpus-mining*, *counterfactuality*, *robustness*, and *multilinguality*. Below are key questions and findings from my work:

<!-- **Data repurposing**. How to generate high-quality synthetic datasets with LLMs?
> * I introduced [reverse instructions](https://arxiv.org/abs/2304.08460) to repurpose existing human-written texts for instruction tuning, improving long-form output quality. 
> * I developed [MURI (Multilingual Reverse Instructions)](https://arxiv.org/abs/2409.12958), creating instruction-tuning datasets for 200 languages by repurposing multilingual human-written corpora.
> * I co-developed [CRAFT](https://arxiv.org/abs/2409.02098), a method for generating task-specific synthetic datasets by retrieving and rewriting relevant documents from large-scale corpora, showing competitive results to human-annotated datasets across various tasks. -->

**Data repurposing**. How to generate high-quality synthetic datasets with LLMs?
<div style="background-color: rgba(244, 189, 69, 0.1); padding: 0px 2px; margin-top: -5px; border-radius: 5px;">
  <ol>
    <li>Introduced <a href="https://arxiv.org/abs/2304.08460">reverse instructions</a> to repurpose existing human-written texts for instruction tuning, improving long-form output quality.</li>
    <li>Developed <a href="https://arxiv.org/abs/2409.12958">MURI (Multilingual Reverse Instructions)</a>, creating instruction-tuning datasets for 200 languages by repurposing multilingual human-written corpora.</li>
    <li>Co-developed <a href="https://arxiv.org/abs/2409.02098">CRAFT</a>, a method for generating task-specific synthetic datasets by retrieving and rewriting relevant documents from large-scale corpora, showing competitive results to human-annotated datasets across various tasks.</li>
  </ol>
</div>

**Counterfactuality/Robustness**. How to effectively create counterfactual examples and improve model robustness/capabilities?
<div style="background-color: rgba(244, 189, 69, 0.1); padding: 0px 2px; margin-top: -5px; border-radius: 5px;">
  <ul>
    <li>Generated a <a href="https://arxiv.org/abs/2311.07424">counterfactual open-book QA dataset</a> by utilizing hallucination in LLMs, demonstrating improved faithfulness across various QA datasets. (Google internship)</li>
    <li>Created a <a href="https://arxiv.org/abs/2407.06699">counterfactual document-level relation extraction dataset</a>, improving consistency in relation extraction models.</li>
    <li><a href="https://aclanthology.org/2023.findings-emnlp.36/">Investigated</a> the high variance in few-shot prompt-based fine-tuning, proposing ensembling and active learning techniques for more robust finetuning.</li>
  </ul>
</div>

**Multilinguality and Bias**. Contributing to multilingual NLP and bias recognition.
<div style="background-color: rgba(244, 189, 69, 0.1); padding: 0px 2px; margin-top: -5px; border-radius: 5px;">
  <ul>
    <li>Designed one of the first <a href="https://aclanthology.org/2020.findings-emnlp.32/">multilingual relation extraction datasets</a> covering six languages.</li>
    <li><a href="https://aclanthology.org/2023.findings-emnlp.848/">Demonstrated</a> significant differences in intrinsic bias toward nationalities among various monolingual models (e.g., Arabic, Turkish, German BERTs).</li>
    <li>Analyzed gender-occupation bias in LLMs, linking it to pretraining data, and examining the effects of instruction tuning, PPO/DPO on bias mitigation.</li>
    <li>Turkish-specific contributions: As a Turkish researcher, I've contributed to various Turkish NLP resources: <a href="https://arxiv.org/abs/2407.12402">TurkishMMLU</a> and resources for <a href="https://ieeexplore.ieee.org/abstract/document/9477814/">sentiment analysis</a> and <a href="https://link.springer.com/article/10.1007/s10579-021-09558-0">dependency parsing</a>, various Github repositories [<a href="https://github.com/akoksal/Turkish-Word2Vec">1</a>, <a href="https://github.com/akoksal/Turkish-Lemmatizer">2</a>, <a href="https://github.com/akoksal/BERT-Sentiment-Analysis-Turkish">3</a>] and gave <a href="https://www.youtube.com/watch?v=d6GsBAgzD-I">talks</a>. I co-organized the first Turkic NLP workshop, <a href="https://sigturk.github.io/workshop">SIGTURK</a>, at ACL-2024.
  </ul>
</div>

News
------
**October 2024**: 4 papers accepted at EMNLP 2024: [LongForm](https://arxiv.org/abs/2304.08460), [TurkishMMLU](https://arxiv.org/abs/2407.12402), [SynthEval](https://arxiv.org/abs/2408.17437), [CovERed](https://www.arxiv.org/abs/2407.06699).

**September 2024**: I am visiting the Language Technology Lab at the University of Cambridge.

**August 2024**: I have attended ACL 2024 to co-organize the first Turkic NLP workshop, [SIGTURK](https://sigturk.github.io/workshop).

**May 2024**: I have presented [LongForm](https://arxiv.org/abs/2304.08460) and [Hallucination Augmented Recitations](https://arxiv.org/abs/2311.07424) at [the DPFM workshop](https://iclr.cc/virtual/2024/workshop/20585) at ICLR 2024. 

**May 2024**: I have attended LREC-COLING 2024 to present [SilverAlign](https://aclanthology.org/2024.lrec-main.1290/).

**December 2023**: I have attended EMNLP 2023 to present [MEAL](https://aclanthology.org/2023.findings-emnlp.36/) and [Language-Agnostic Bias Detection in Language Models](https://aclanthology.org/2023.findings-emnlp.848/).

**June 2023**: I will be in Mountain View for 3 months as a research intern at Google, focusing on attribution and counterfactuality in large language models.
<!-- **October 2022**: [The Better Your Syntax, the Better Your Semantics? Probing Pretrained Language Models for the English Comparative Correlative](https://aclanthology.org/2022.emnlp-main.746/) is accepted at EMNLP 2022.<br>
📃 New preprint: [SilverAlign: MT-Based Silver Data Algorithm For Evaluating Word Alignment](https://arxiv.org/abs/2210.06207)
**September 2022**: I attended [ELLIS Doctoral Symposium](https://ellisalicante.org/eds2022/) in Alicante and presented our work on language-agnostic racial bias detection in LMs.
-->

Selected Publications
------
1. **Köksal, A.**; Thaler, M.; Imani, A.; Üstün, A.; Korhonen, A.; Schütze, H.; *MURI: High-Quality Instruction Tuning Datasets for Low-Resource Languages via Reverse Instructions*. [Submitted to TACL](https://arxiv.org/abs/2409.12958). 2024. [💻 Code](https://github.com/akoksal/muri).
2. Ziegler, I.*; **Köksal, A.***; Elliott, D.; Schütze, H.; *CRAFT Your Dataset: Task-Specific Synthetic Dataset Generation Through Corpus Retrieval and Augmentation*. [Submitted to TACL](https://arxiv.org/abs/2409.02098). 2024. [💻 Code](https://github.com/ziegler-ingo/CRAFT).
3. **Köksal, A.**; Schick, T.; Korhonen, A.; Schütze, H.; *LongForm: Effective Instruction Tuning with Reverse Instructions*. [EMNLP Findings](https://arxiv.org/abs/2304.08460). 2024.
4. Modarressi, A.; **Köksal, A.**; Schütze, H.; *Consistent Document-Level Relation Extraction via Counterfactuals*. [EMNLP Findings](https://arxiv.org/abs/2407.06699). 2024.
5. Yüksel, A.; **Köksal, A.**; Şenel, L.K.; Korhonen, A.; Schütze, H.; *TurkishMMLU: Measuring Massive Multitask Language Understanding in Turkish*. [EMNLP Findings](https://arxiv.org/abs/2407.12402). 2024.
6. Zhao, R.; **Köksal, A.**; Liu, Y.; Weissweiler, L.; Korhonen, A.; Schütze, H.; *SYNTHEVAL: Hybrid Behavioral Testing of NLP Models with Synthetic CheckLists*. [EMNLP Findings](https://arxiv.org/abs/2408.12402). 2024.
7. **Köksal, A.**; Aksitov, R.; Chang, C.C.; *Hallucination Augmented Recitations for Language Models*. [Submitted to COLING](https://arxiv.org/abs/2311.07424). 2024.
8. **Köksal, A.**; Schick, T.; Schütze, H.; *MEAL: Stable and Active Learning for Few-Shot Prompting*. [EMNLP Findings](https://aclanthology.org/2023.findings-emnlp.36/). 2023.
9. **Köksal, A.**; Yalcin, O.; Akbiyik, A.; Kilavuz, M.T.; Korhonen, A.; Schütze, H.; *Language-Agnostic Bias Detection in Language Models with Bias Probing*. [EMNLP Findings](https://aclanthology.org/2023.findings-emnlp.848/). 2023.
10. Huang, Y.; Giledereli, B.; **Köksal, A.**; Özgür, A.; Ozkirimli, E.; *Balancing Methods for Multilabel Text Classification with Long-Tailed Class Distribution*. [EMNLP](https://aclanthology.org/2021.emnlp-main.643/). 2021.
11. **Köksal, A.**; Özgür, A.; *The RELX Dataset and Matching the Multilingual Blanks for Cross-lingual Relation Classification*. [EMNLP Findings](https://aclanthology.org/2020.findings-emnlp.32/). 2020.