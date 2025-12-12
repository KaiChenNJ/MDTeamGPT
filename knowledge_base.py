import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

KB_DIR = "knowledge_bases"
CORRECT_KB_PATH = os.path.join(KB_DIR, "correct_kb")
COT_KB_PATH = os.path.join(KB_DIR, "cot_kb")


class DualKnowledgeBase:
    def __init__(self):
        self.correct_store = None
        self.cot_store = None
        self.embeddings = None
        self.initialized = False

        if not os.path.exists(KB_DIR):
            os.makedirs(KB_DIR)

    def init_embeddings(self, api_key, base_url):
        if self.initialized: return
        try:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-v3",
                api_key=api_key,
                base_url=base_url,
                check_embedding_ctx_length=False
            )
            self._load_stores()
            self.initialized = True
        except Exception as e:
            print(f"Embedding init failed: {e}")

    def _load_stores(self):
        if os.path.exists(CORRECT_KB_PATH):
            try:
                self.correct_store = FAISS.load_local(CORRECT_KB_PATH, self.embeddings,
                                                      allow_dangerous_deserialization=True)
            except:
                self.correct_store = None

        if os.path.exists(COT_KB_PATH):
            try:
                self.cot_store = FAISS.load_local(COT_KB_PATH, self.embeddings, allow_dangerous_deserialization=True)
            except:
                self.cot_store = None

    def save_correct_experience(self, record: dict):
        """
        Stores into CorrectKB:
        {
            "Question": <...>,
            "Answer": <...>,
            "Summary of S4_final": <...>
        }
        """
        text_content = json.dumps(record, ensure_ascii=False, indent=2)
        meta = {"type": "correct_kb", "case_snippet": record.get("Question", "")[:50]}

        doc = Document(page_content=text_content, metadata=meta)

        if self.correct_store:
            self.correct_store.add_documents([doc])
        else:
            self.correct_store = FAISS.from_documents([doc], self.embeddings)
        self.correct_store.save_local(CORRECT_KB_PATH)

    def save_reflection_experience(self, record: dict):
        """
        Stores into ChainKB:
        {
            "Question": <...>,
            "Correct Answer": <...>,
            "Initial Hypothesis": <...>,
            "Analysis Process": <...>,
            "Final Conclusion": <...>,
            "Error Reflection": <...>
        }
        """
        text_content = json.dumps(record, ensure_ascii=False, indent=2)
        meta = {"type": "chain_kb", "case_snippet": record.get("Question", "")[:50]}

        doc = Document(page_content=text_content, metadata=meta)

        if self.cot_store:
            self.cot_store.add_documents([doc])
        else:
            self.cot_store = FAISS.from_documents([doc], self.embeddings)
        self.cot_store.save_local(COT_KB_PATH)

    def retrieve_context_details(self, query: str, k=2):
        if not self.initialized:
            return {"text": "Knowledge Base not initialized.", "docs": []}

        context_text_parts = []
        all_docs = []

        # 1. Correct Patterns
        if self.correct_store:
            docs = self.correct_store.similarity_search(query, k=k)
            if docs:
                context_text_parts.append("--- [CorrectKB] SUCCESSFUL EXPERIENCES ---")
                for d in docs:
                    d.metadata["source_kb"] = "CorrectKB"
                    context_text_parts.append(d.page_content)
                    all_docs.append(d)

        # 2. Reflection Patterns
        if self.cot_store:
            docs = self.cot_store.similarity_search(query, k=k)
            if docs:
                context_text_parts.append("\n--- [ChainKB] ERROR REFLECTIONS ---")
                for d in docs:
                    d.metadata["source_kb"] = "ChainKB"
                    context_text_parts.append(d.page_content)
                    all_docs.append(d)

        final_text = "\n".join(context_text_parts) if context_text_parts else "No specific prior experience found."

        return {
            "text": final_text,
            "docs": all_docs
        }


kb_system = DualKnowledgeBase()