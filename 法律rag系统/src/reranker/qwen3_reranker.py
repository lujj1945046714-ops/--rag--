# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: LLM任务型对话Agent
# --------------------------------------------


import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
warnings.filterwarnings("ignore")


class Qwen3ReRanker(object):
    def __init__(self, model_path, max_length=4096, batch_size=4):
        # 加载 rerank 模型

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_path).cuda().eval()

        self.token_false_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.token_true_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.max_length = max_length 
        self.batch_size = batch_size

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
                
        self.task = (
            'Given a legal question in Chinese, retrieve the most relevant Chinese legal articles, '
            'provisions, or case analyses that directly answer or apply to the question. '
            'Prefer documents that cite specific law names and article numbers. '
            'Higher-level laws (national law > administrative regulation > local regulation) '
            'should be ranked higher when content relevance is equal.'
        )


    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = self.task
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def rank(self, query, candidate_docs, topk=10):
        # 输入文档对，返回每一对(query, doc)的相关得分，并从大到小排序
        if not candidate_docs:
            return []

        pairs = [
            self.format_instruction(self.task, query, doc.page_content)
            for doc in candidate_docs
        ]

        scores = []
        for start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[start : start + self.batch_size]
            inputs = self.process_inputs(batch_pairs)
            batch_scores = self.compute_logits(inputs)
            scores.extend(batch_scores)

        response = [
            doc
            for score, doc in sorted(
                zip(scores, candidate_docs), reverse=True, key=lambda x: x[0]
            )
        ][:topk]
        return response


if __name__ == "__main__":
    qwen3_reranker = "./models/Qwen3-Reranker-4B"
    qwen3_rerank = Qwen3ReRanker(qwen3_reranker)
    query = "今天天气怎么样"
    from langchain_core.documents import Document
    docs = [Document(page_content=doc, metadata={}) for doc in ["你好", "今天天气不错", "今天有雨吗"]]
    response = qwen3_rerank.rank(query, docs)
    print(response)
