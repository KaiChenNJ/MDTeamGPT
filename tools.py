from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool

try:
    from langchain_community.tools import PubMedQueryRun
    from langchain_community.utilities import PubMedAPIWrapper
except ImportError:
    PubMedQueryRun = None
    PubMedAPIWrapper = None


class MedicalTools:
    def __init__(self, enable=True):
        self.enable = enable
        self.tools = []

        if not enable:
            return

        # 1. Web Search
        try:
            self.search = DuckDuckGoSearchRun()
            self.tools.append(
                Tool(
                    name="Web_Search",
                    func=self.search.run,
                    description="Search for general medical guidelines, drug interactions, or recent news."
                )
            )
        except Exception as e:
            print(f"Search tool init failed: {e}")

        # 2. PubMed
        if PubMedQueryRun:
            try:
                self.pubmed = PubMedQueryRun(api_wrapper=PubMedAPIWrapper())
                self.tools.append(
                    Tool(
                        name="PubMed",
                        func=self.pubmed.run,
                        description="Search for biomedical literature and academic papers."
                    )
                )
            except Exception as e:
                print(f"PubMed init failed: {e}")

    def run_tools(self, query: str):
        """Execute tools and return combined string result"""
        if not self.enable or not self.tools:
            return ""

        results = []
        for tool in self.tools:
            try:
                res = tool.run(query)
                results.append(f"--- {tool.name} Result ---\n{res[:600]}...")
            except Exception as e:
                results.append(f"Error running {tool.name}: {e}")
        return "\n".join(results)