"""
Pipeline de production : classifieur PubMedBERT+LoRA calibré, retrieval
FAISS (citation + détection hors périmètre), et LLM génératif pour
l'explication.

⚠️ Ce module nécessite torch/transformers/peft/sentence-transformers/faiss et
les artefacts produits par les notebooks Jour 1 à 4 (adaptateur LoRA,
température, seuils, index FAISS). Il n'est pas couvert par les tests
unitaires (qui restent volontairement légers et sans GPU) — seule sa logique
pure (src.data_utils, src.guardrails) est testée automatiquement.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from src.data_utils import split_sentences
from src.guardrails import build_explanation_prompt, contains_contradiction, format_response


LABEL2ID = {"yes": 0, "no": 1, "maybe": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class PipelineConfig:
    classifier_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    adapter_dir: str = "./checkpoints/pubmedbert_lora_adapter_final"
    explain_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    faiss_index_path: str = "./checkpoints/biomedical_corpus.index"
    temperature: float = 1.0
    abstention_threshold: float = 0.6
    oos_threshold: float = 0.5
    max_length: int = 384


class BiomedicalAssistant:
    """Regroupe verdict calibré, citation, détection hors périmètre et
    explication en langage naturel dans une seule interface simple à appeler.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_classifier()
        self._load_retrieval()
        self._load_explainer()

    def _load_classifier(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.adapter_dir)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.classifier_model_name,
            num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
        )
        self.classifier = PeftModel.from_pretrained(base_model, self.config.adapter_dir)
        self.classifier.to(self.device)
        self.classifier.eval()

    def _load_retrieval(self):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=self.device
        )
        self.faiss_index = faiss.read_index(self.config.faiss_index_path)

    def _load_explainer(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.explain_tokenizer = AutoTokenizer.from_pretrained(self.config.explain_model_name)
        self.explain_model = AutoModelForCausalLM.from_pretrained(
            self.config.explain_model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.explain_model.eval()

    @torch.no_grad()
    def _ood_score(self, question: str, k: int = 5) -> float:
        q_emb = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        similarities, _ = self.faiss_index.search(q_emb, k)
        return float(1 - similarities.mean())

    @torch.no_grad()
    def _get_verdict(self, question: str, context: str) -> dict:
        inputs = self.tokenizer(
            question, context, truncation=True, max_length=self.config.max_length,
            padding=True, return_tensors="pt",
        ).to(self.device)
        logits = self.classifier(**inputs).logits.cpu()
        probs = F.softmax(logits / self.config.temperature, dim=1).squeeze(0)
        confidence, pred_id = torch.max(probs, dim=0)
        confidence, pred_id = confidence.item(), pred_id.item()

        verdict = "incertain" if confidence < self.config.abstention_threshold else ID2LABEL[pred_id]
        return {"verdict": verdict, "raw_label": ID2LABEL[pred_id], "confidence": confidence}

    def _cite_evidence(self, question: str, context: str) -> str:
        sentences = split_sentences(context)
        if not sentences:
            return context

        sentence_embeddings = self.embedder.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
        question_embedding = self.embedder.encode([question], convert_to_tensor=True, normalize_embeddings=True)
        similarities = (sentence_embeddings @ question_embedding.T).squeeze(1)
        top_index = int(torch.argmax(similarities).item())
        return sentences[top_index]

    @torch.no_grad()
    def _generate_explanation(self, question: str, citation: str, verdict: str, max_new_tokens: int = 80) -> str:
        prompt = build_explanation_prompt(question, citation, verdict)
        messages = [
            {"role": "system", "content": (
                "Tu es un assistant d'aide à la lecture biomédicale. Tu expliques des "
                "verdicts déjà établis, tu ne les modifies jamais, et tu ne donnes "
                "aucun conseil médical ni diagnostic."
            )},
            {"role": "user", "content": prompt},
        ]
        inputs = self.explain_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.explain_model.device)

        output = self.explain_model.generate(
            inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.explain_tokenizer.eos_token_id,
        )
        return self.explain_tokenizer.decode(
            output[0][inputs.shape[-1]:], skip_special_tokens=True
        ).strip()

    def answer(self, question: str, context: str) -> dict:
        """Point d'entrée principal du pipeline."""
        score = self._ood_score(question)
        if score > self.config.oos_threshold:
            return {
                "in_scope": False,
                "message": "Question hors du périmètre biomédical couvert par cet assistant.",
            }

        verdict_info = self._get_verdict(question, context)
        citation = self._cite_evidence(question, context)
        explanation = self._generate_explanation(question, citation, verdict_info["verdict"])

        response = format_response(verdict_info["verdict"], verdict_info["confidence"], citation, explanation)
        response["in_scope"] = True
        response["contradiction_flag"] = contains_contradiction(explanation, verdict_info["verdict"])
        return response
