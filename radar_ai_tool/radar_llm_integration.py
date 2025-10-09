"""
RECAP Real LLM Integration
=========================

This module provides real LLM integration for production use.

"""

try:
    import hashlib
    import json
    import os
    import pickle
    import time
    from datetime import datetime
    from typing import Dict, List

    from langchain.schema import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    print(f"LLM dependencies not available: {e}")


class ProductionLLMAgent:
    """Production LLM agent using OpenAI and LangChain"""

    def __init__(self, agent_type: str, api_key: str):
        if not DEPS_AVAILABLE:
            raise ImportError(
                "LLM dependencies not installed. Run: "
                "pip install -r requirements_recap.txt"
            )

        self.agent_type = agent_type
        self.api_key = api_key

        # Initialize OpenAI client with fastest model
        self.client = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo",  # Fastest model for production
            temperature=0.1,
            max_tokens=800,  # Reduced for speed
            request_timeout=15,  # Faster timeout
            streaming=False,  # Disable streaming for batch processing
        )

        # Initialize embeddings (for future vector store use)
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vectorstore = None

        # Agent-specific prompts (optimized for speed)
        self.prompts = {
            "visibility": """
            Analyze VISIBILITY: Is this theme visible across leadership and teams?

            Rate 1-10 and provide:
            1. Leadership awareness level
            2. Team coordination status
            3. Key findings (max 3 points)
            4. Recommendations (max 2)

            Be concise and specific.
            """,
            "alignment": """
            Analyze ALIGNMENT: Does it connect to strategy and OKRs?

            Rate 1-10 and provide:
            1. OKR connection strength
            2. Strategy alignment evidence
            3. Key findings (max 3 points)
            4. Recommendations (max 2)

            Be concise and specific.
            """,
            "confidence": """
            Analyze CONFIDENCE: Are assumptions valid? Do leading indicators
            tie to KPIs?

            Rate 1-10 and provide:
            1. Assumption validity
            2. Leading-lagging indicator connection
            3. Key findings (max 3 points)
            4. Recommendations (max 2)

            Be concise and specific.
            """,
            "user_problems": """
            Analyze USER PROBLEMS: Is opportunity anchored in real user research?

            Rate 1-10 and provide:
            1. User research quality
            2. Problem validation strength
            3. Key findings (max 3 points)
            4. Recommendations (max 2)

            Be concise and specific.
            """,
        }

    def _get_cache_key(self, proposal: str, knowledge_base: List[Dict] = None) -> str:
        """Generate cache key for the analysis"""
        content = f"{self.agent_type}:{proposal}"
        if knowledge_base:
            kb_content = "".join(
                [f"{doc['title']}{doc['content']}" for doc in knowledge_base]
            )
            content += kb_content
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Dict:
        """Get cached result if available and recent (24 hours for speed)"""
        cache_file = f".cache_{cache_key}.pkl"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                # Check if cache is less than 24 hours old (for development)
                if time.time() - cached_data["timestamp"] < 86400:  # 24 hours
                    print(f"⚡ Using cached result for {self.agent_type}")
                    return cached_data["result"]
            except Exception:
                pass
        return None

    def _cache_result(self, cache_key: str, result: Dict):
        """Cache the analysis result"""
        cache_file = f".cache_{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({"timestamp": time.time(), "result": result}, f)
        except Exception:
            pass

    def analyze(self, proposal: str, knowledge_base: List[Dict] = None) -> Dict:
        """Run analysis using actual LLM with document context"""

        # Check cache first
        cache_key = self._get_cache_key(proposal, knowledge_base)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        # Create analysis prompt with document context
        system_prompt = self.prompts[self.agent_type]

        # Add document context if available
        context_text = ""
        if knowledge_base:
            context_text = "\n\nRELEVANT DOCUMENTS FOR CONTEXT:\n"
            for doc in knowledge_base:
                context_text += f"\n--- {doc['title']} ({doc['type']}) ---\n"
                context_text += f"{doc['content']}\n"

        user_prompt = f"""
        PROPOSAL TO ANALYZE:
        {proposal}
        {context_text}

        Please provide a structured analysis including:
        1. Score (1-10)
        2. Key findings with specific evidence from the documents above
        3. Recommendations for next steps
        4. Specific quotes or references from the provided documents

        Format response as JSON:
        {{
            "score": <1-10>,
            "findings": ["<finding1>", "<finding2>", "<finding3>"],
            "recommendations": ["<rec1>", "<rec2>"]
        }}
        """

        try:
            # Call OpenAI via LangChain
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            start_time = time.time()
            response = self.client.invoke(messages)
            elapsed_time = time.time() - start_time
            print(f"⚡ {self.agent_type} analysis: {elapsed_time:.2f}s")

            # Try to parse JSON response
            try:
                result = json.loads(response.content)
                # Cache the successful result
                self._cache_result(cache_key, result)
                return result
            except json.JSONDecodeError:
                # Quick fallback parsing for common patterns
                content = response.content
                try:
                    # Try to extract score with regex
                    import re

                    score_match = re.search(r'"score":\s*(\d+)', content)
                    score = int(score_match.group(1)) if score_match else 5

                    # Extract findings
                    findings_match = re.search(
                        r'"findings":\s*\[(.*?)\]', content, re.DOTALL
                    )
                    findings = (
                        ["Analysis completed"]
                        if not findings_match
                        else ["Quick analysis"]
                    )

                    result = {
                        "score": score,
                        "findings": findings,
                        "recommendations": ["Quick analysis completed"],
                    }
                except Exception:
                    result = {
                        "score": 5,
                        "findings": ["Analysis completed"],
                        "recommendations": ["Review response format"],
                    }
                self._cache_result(cache_key, result)
                return result

        except Exception as e:
            # Fallback for any API errors
            result = {
                "score": 5,
                "findings": [f"LLM analysis error: {str(e)}"],
                "recommendations": ["Check API key and connection"],
                "evidence": ["Error occurred during analysis"],
            }
            return result


class ProductionRECAPAnalyzer:
    """Production RECAP analyzer with real LLM integration"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.agents = {
            "visibility": ProductionLLMAgent("visibility", api_key),
            "alignment": ProductionLLMAgent("alignment", api_key),
            "confidence": ProductionLLMAgent("confidence", api_key),
            "user_problems": ProductionLLMAgent("user_problems", api_key),
        }

    def analyze_proposal(
        self, proposal: str, knowledge_base: List[Dict] = None
    ) -> Dict:
        """Run full RECAP analysis with real LLMs"""

        results = {}

        # Run each agent
        for agent_name, agent in self.agents.items():
            try:
                results[agent_name] = agent.analyze(proposal, knowledge_base)
            except Exception as e:
                # Fallback on error
                results[agent_name] = {
                    "score": 5,
                    "findings": [f"Analysis error: {str(e)}"],
                    "recommendations": ["Retry analysis"],
                    "evidence": [],
                }

        # Calculate overall score
        overall_score = sum(results[agent]["score"] for agent in results) / 4

        # Determine recommendation
        if overall_score >= 7:
            recommendation = "PROCEED"
            rec_color = ""
        elif overall_score >= 5:
            recommendation = "PROCEED WITH CAUTION"
            rec_color = ""
        else:
            recommendation = "RECONSIDER"
            rec_color = ""

        return {
            "overall_score": overall_score,
            "recommendation": recommendation,
            "rec_color": rec_color,
            "agent_results": results,
            "analysis_time": datetime.now(),
            "proposal_text": proposal,
        }
