---
permalink: /
title: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am a Research Scientist at Google DeepMind in London, working in the Privacy and Security team. I am finishing up my PhD in Computer Science as an ELLIS student at CIS at LMU Munich and LTL at the University of Cambridge, advised by [Prof. Hinrich Schütze](https://www.cis.uni-muenchen.de/personen/professoren/schuetze/) and [Prof. Anna Korhonen](https://www.languagesciences.cam.ac.uk/directory/alk23@cam.ac.uk). During my PhD, I also had the opportunity to complete research internships at Google in Mountain View and Amazon in Madrid.

My research focuses on improving LLM capabilities through **effective data utilization** and **synthetic dataset generation**, with a particular emphasis on corpus-mining, counterfactuality, robustness, privacy, and multilinguality. Below are key questions and findings from my work:

* **Data repurposing:** How to generate high-quality synthetic datasets with LLMs?
    * Introduced *reverse instructions* to repurpose existing human-written texts for instruction tuning, improving long-form output quality.
    * Developed *MURI* (Multilingual Reverse Instructions), creating instruction-tuning datasets for 200 languages by repurposing multilingual human-written corpora.
    * Co-developed *CRAFT*, a method for generating task-specific synthetic datasets by retrieving and rewriting relevant documents from large-scale corpora.
* **Counterfactuality & Robustness:** How to effectively create counterfactual datasets and improve model robustness/capabilities?
    * Generated a counterfactual open-book QA dataset by utilizing hallucination in LLMs, demonstrating improved faithfulness across various QA datasets.
    * Created a counterfactual document-level relation extraction dataset, improving consistency in relation extraction models.
    * Investigated the high variance in few-shot prompt-based fine-tuning, proposing ensembling and active learning techniques for more robust finetuning.
* **Multilinguality & Bias:** Contributing to multilingual NLP and bias recognition.
    * Designed one of the first multilingual relation extraction datasets covering six languages.
    * Demonstrated significant differences in intrinsic bias toward nationalities among various monolingual models.
    * Analyzed gender-occupation bias in LLMs, linking it to pretraining data, and examining the effects of instruction tuning and alignment on bias mitigation.
* **Turkish-specific contributions:** As a Turkish researcher, I've contributed to various Turkish NLP resources: *TurkishMMLU*, sentiment analysis, and dependency parsing. I also co-organized the Turkic NLP workshop, *SIGTURK*, at ACL 2024 and 2026.

## News
* **March 2026:** I am co-organizing the Second Turkic NLP Workshop at EACL 2026. My paper *Tracing bias from pretraining data to alignment* was accepted to LREC 2026.
* **November 2025:** Our paper *Do We Know What LLMs Don’t Know?* was accepted to EMNLP-Findings 2025.
* **July 2025:** *TUMLU* was accepted to ACL 2025.
* **May 2025:** I have started a new position as a Research Scientist at Google DeepMind in London, joining the Privacy and Security team!
* **March 2025:** Our paper *MemLLM* was accepted to TMLR.
* **October 2024:** 4 papers accepted at EMNLP 2024: LongForm, TurkishMMLU, SynthEval, CovERed.
* **September 2024:** I am visiting the Language Technology Lab at the University of Cambridge.
* **August 2024:** I attended ACL 2024 to co-organize the first Turkic NLP workshop, SIGTURK.
* **May 2024:** I presented LongForm and Hallucination Augmented Recitations at the DPFM workshop at ICLR 2024.
* **December 2023:** I attended EMNLP 2023 to present MEAL and Language-Agnostic Bias Detection in Language Models.