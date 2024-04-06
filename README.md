# Secure Your Model: An Effective Key Prompt Protection Mechanism for Large Language Models

:mega: The paper is accepted by NAACL2024.
* Paper: [Link](https://www.researchgate.net/publication/374555007_Secure_Your_Model_An_Effective_Key_Prompt_Protection_Mechanism_for_Large_Language_Models)

## Abstract
Large language models (LLMs) have notably revolutionized many domains within natural language processing due to their exceptional performance. Their security has become increasingly vital. This study is centered on protecting LLMs against unauthorized access and potential theft. We propose a simple yet effective protective measure wherein a unique key prompt is embedded within the LLM. This mechanism enables the model to respond only when presented with the correct key prompt; otherwise, LLMs will refuse to react to any input instructions. This key prompt protection offers a robust solution to prevent the unauthorized use of LLMs, as the model becomes unusable without the correct key. We evaluated the proposed protection on multiple LLMs and NLP tasks. Results demonstrate that our method can successfully protect the LLM without significantly impacting the model's original function. Moreover, we demonstrate potential attacks that attempt to bypass the protection mechanism will adversely affect the model's performance, further emphasizing the effectiveness of the proposed protection method.

## Dataset Download
- Data Download: https://physionet.org/content/mimic-cxr/
- Data Preprocessing: https://github.com/abachaa/MEDIQA2021/tree/main/Task3

## Cite This Work
If you find this project useful, you can cite this work by:

@article{tangsecure,
  title={Secure Your Model: An Effective Key Prompt Protection Mechanism for Large Language Models},
  author={Tang, Ruixiang and Chuang, Yu-Neng and Cai, Xuanting and Platforms, Meta and Du, Mengnan and Hu, Xia}
}
