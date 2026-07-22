from sentence_transformers import CrossEncoder
import torch


class GPUReranker:


    def __init__(
        self,
        model_name="BAAI/bge-reranker-v2-m3"
    ):


        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


        self.model = CrossEncoder(
            model_name,
            device=self.device
        )


    def rerank(
        self,
        query,
        documents,
        top_k=10
    ):


        pairs = [
            (
                query,
                doc["text"]
            )
            for doc in documents
        ]


        scores = self.model.predict(
            pairs,
            batch_size=32
        )


        ranked = sorted(
            zip(documents, scores),
            key=lambda x:x[1],
            reverse=True
        )


        results=[]


        for doc,score in ranked[:top_k]:

            doc["rerank_score"]=float(score)

            results.append(doc)


        return results