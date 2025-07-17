# Experiments
## Retrieval Datasets
1-Encyclopedia  
Wikipedia: https://huggingface.co/datasets/wikimedia/wikipedia  
(To-Do)The complete wiki data is classified to 4 domains, including: common sense, medicine related, law related, computer science+coding related   
2-Specific Domain  
PubMed: https://huggingface.co/datasets/qiaojin/PubMedQA  
Legal Bench: https://huggingface.co/datasets/nguha/legalbench  
Code: https://huggingface.co/datasets/code-search-net/code_search_net  
Arxiv-CS: https://huggingface.co/datasets/arxiv-community/arxiv_dataset (filter with category tags)  

## Test Tasks & DataSets
1-Question answering  
Natural Questions: https://huggingface.co/datasets/google-research-datasets/natural_questions  
Trivia Questions: https://huggingface.co/datasets/mandarjoshi/trivia_qa  
Squad: https://huggingface.co/datasets/rajpurkar/squad  
Web Questions: https://huggingface.co/datasets/stanfordnlp/web_questions  
2-Reasoning  
MMLU: https://huggingface.co/datasets/cais/mmlu  
Strategy QA: https://huggingface.co/datasets/wics/strategy-qa  
HotPot QA: https://huggingface.co/datasets/hotpotqa/hotpot_qa  
  
## Embedding Models  
1.Qwen3-embedding-0.6B: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF  
2.Qwen3-embedding-4B: https://huggingface.co/Qwen/Qwen3-Embedding-4B  
3.Other..  

## LLM Models
1.Qwen3  
2.LLama 3/4  

## Key Steps

1-Setup baseline
- 实验设定：1个client，包含wiki知识库（common sense）
- 测试数据：7个test task中的得分

2-Add wiki data
- 实验设定：4个client，分别包含wiki知识库（common sense, medicine related, law related, computer science+coding related）
- 测试数据：7个test task中的得分


3-Add domain data
- 实验设定：4个client，分别包含1个wiki知识库（common sense）和三个domain知识库（med, legal, CS+coding）
- 测试数据：7个test task中的得分

4-Add wiki data + domain data
- 实验设定：7个client，分别包含4个wiki知识库（common sense, medicine related, law related, computer science+coding related）和三个domain知识库（med, legal, CS+coding）
- 测试数据：7个test task中的得分

5-Performance with PoR, PoG algorithms
- 实验设定：4个client，分别包含1个wiki知识库（common sense）和三个domain知识库（med, legal, CS+coding）
- 测试数据：
1）对于query请求的响应时间（retrieval时间，generation时间）
2）添加隐私算法之后的响应时间（retrieval+PoR时间，generation+PoG时间）
3）PoR, PoG proof size
