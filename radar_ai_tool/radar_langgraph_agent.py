"""
RADAR LangGraph Integration
==========================

Advanced multi-agent system using LangGraph for sophisticated workflow orchestration
and agent collaboration in product strategy analysis.
"""

try:
    import json
    import operator
    from datetime import datetime
    from typing import Annotated, Dict, List, TypedDict

    from langchain.pydantic_v1 import BaseModel, Field
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    print(f"LangGraph dependencies not available: {e}")


class AnalysisState(TypedDict):
    """Shared state across all RADAR agents"""

    proposal: str
    documents: List[Dict]

    # Individual agent results
    review_analysis: Dict
    analyze_analysis: Dict
    decide_analysis: Dict
    align_analysis: Dict

    # Collaborative insights - using Annotated for multiple updates
    cross_agent_insights: Annotated[List[str], operator.add]
    conflicting_findings: Annotated[List[str], operator.add]
    consensus_items: Annotated[List[str], operator.add]

    # Final synthesis
    overall_score: float
    final_recommendation: str
    confidence_level: str
    next_steps: Annotated[List[str], operator.add]

    # Workflow state
    current_step: str
    iteration_count: int
    needs_refinement: bool


class RadarAnalysisResult(BaseModel):
    """Structured output for each RADAR lens analysis"""

    score: int = Field(description="Score from 1-10")
    confidence: float = Field(description="Confidence in the score from 0-1")
    key_findings: List[str] = Field(description="Main findings with evidence")
    recommendations: List[str] = Field(description="Specific recommendations")
    evidence: List[str] = Field(description="Supporting evidence from documents")
    concerns: List[str] = Field(description="Areas of concern or risk")
    dependencies: List[str] = Field(description="Dependencies on other work/teams")


class RADARLangGraphAgent:
    """Advanced RADAR analysis using LangGraph orchestration"""

    def __init__(self, api_key: str):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph dependencies not installed. Run: pip install langgraph"
            )

        self.api_key = api_key
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4",  # Using GPT-4 for better reasoning
            temperature=0.1,
        )

        # Initialize vector store for document retrieval
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vectorstore = None

        # Create the workflow graph
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for RADAR analysis"""

        workflow = StateGraph(AnalysisState)

        # Add nodes for each step
        workflow.add_node("initialize", self._initialize_analysis)
        workflow.add_node("review_lens", self._review_lens_analysis)
        workflow.add_node("analyze_lens", self._analyze_lens_analysis)
        workflow.add_node("decide_lens", self._decide_lens_analysis)
        workflow.add_node("align_lens", self._align_lens_analysis)
        workflow.add_node("cross_validate", self._cross_validate_findings)
        workflow.add_node("synthesize", self._synthesize_results)
        workflow.add_node("refine", self._refine_analysis)

        # Define the workflow edges
        workflow.set_entry_point("initialize")

        # Sequential lens analysis to avoid concurrent updates
        workflow.add_edge("initialize", "review_lens")
        workflow.add_edge("review_lens", "analyze_lens")
        workflow.add_edge("analyze_lens", "decide_lens")
        workflow.add_edge("decide_lens", "align_lens")

        # Cross-validation after all lenses complete
        workflow.add_edge("align_lens", "cross_validate")

        # Synthesis and potential refinement
        workflow.add_edge("cross_validate", "synthesize")

        # Conditional edge for refinement
        workflow.add_conditional_edges(
            "synthesize", self._should_refine, {"refine": "refine", "end": END}
        )

        workflow.add_edge("refine", "cross_validate")

        return workflow.compile()

    def _initialize_analysis(self, state: AnalysisState) -> AnalysisState:
        """Initialize the analysis with document processing"""

        # Set up vector store for document retrieval
        if state["documents"]:
            texts = []
            metadatas = []

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )

            for doc in state["documents"]:
                chunks = text_splitter.split_text(doc["content"])
                texts.extend(chunks)
                metadatas.extend(
                    [
                        {"title": doc["title"], "type": doc["type"], "chunk_id": i}
                        for i in range(len(chunks))
                    ]
                )

            self.vectorstore = Chroma.from_texts(
                texts=texts, metadatas=metadatas, embedding=self.embeddings
            )

        return {
            "current_step": "lens_analysis",
            "iteration_count": 0,
            "cross_agent_insights": [],
            "conflicting_findings": [],
            "consensus_items": [],
        }

    def _review_lens_analysis(self, state: AnalysisState) -> AnalysisState:
        """Review lens: Analyze team visibility and coordination"""

        # Retrieve relevant documents
        relevant_docs = self._retrieve_relevant_docs(
            state["proposal"], ["roadmap", "team", "project"]
        )

        system_prompt = """You are a Visibility Analysis Agent in the RADAR framework.

        Your role is to assess VISIBILITY:
        - Is this theme visible across leadership reviews and other teams' initiatives?
        - Check for organizational awareness and coordination
        - Identify other teams working on similar features
        - Find historical projects that are relevant
        - Spot potential overlaps or conflicts
        - Discover collaboration opportunities

        Rate 1-10 where:
        - 1-3: Theme is invisible - no organizational awareness or team initiatives
        - 4-6: Limited visibility - some awareness but poor coordination
        - 7-10: High visibility - well-known theme with good organizational awareness

        Be specific about teams, projects, and timelines mentioned in the documents."""

        result = self._run_structured_analysis(
            system_prompt, state["proposal"], relevant_docs
        )

        return {"review_analysis": result.dict()}

    def _analyze_lens_analysis(self, state: AnalysisState) -> AnalysisState:
        """Analyze lens: Deep strategic analysis"""

        relevant_docs = self._retrieve_relevant_docs(
            state["proposal"], ["strategy", "okr", "priority"]
        )

        system_prompt = """You are an Alignment Analysis Agent in the RADAR framework.

        Your role is to assess ALIGNMENT:
        - Does it connect clearly to the strategy and OKRs?
        - Connection to current OKRs and strategic priorities
        - Alignment with company strategy documents
        - Strategic value and business impact
        - Resource allocation efficiency

        Rate 1-10 where:
        - 1-3: No clear connection to strategy or OKRs
        - 4-6: Indirect or weak connection to strategic priorities
        - 7-10: Strong, clear connection to strategy and OKRs

        Reference specific strategy documents and OKRs."""

        result = self._run_structured_analysis(
            system_prompt, state["proposal"], relevant_docs
        )

        return {"analyze_analysis": result.dict()}

    def _decide_lens_analysis(self, state: AnalysisState) -> AnalysisState:
        """Decide lens: Confidence and validation analysis"""

        relevant_docs = self._retrieve_relevant_docs(
            state["proposal"], ["research", "data", "metric", "postmortem"]
        )

        system_prompt = """You are a Confidence Analysis Agent in the RADAR framework.

        Your role is to assess CONFIDENCE:
        - Are the assumptions valid, and do the leading indicators tie back to
          lagging KPIs?
        - Quality of assumptions underlying the proposal
        - Reliability of success metrics
        - Risk factors and mitigation strategies
        - Evidence quality and data support

        Rate 1-10 where:
        - 1-3: Invalid assumptions, poor indicator-KPI connection
        - 4-6: Some valid assumptions but weak indicator-KPI linkage
        - 7-10: Strong assumptions with clear leading-to-lagging indicator
          connections

        Identify specific validation steps and data gaps."""

        result = self._run_structured_analysis(
            system_prompt, state["proposal"], relevant_docs
        )

        return {"decide_analysis": result.dict()}

    def _align_lens_analysis(self, state: AnalysisState) -> AnalysisState:
        """Align lens: User problem and market fit analysis"""

        relevant_docs = self._retrieve_relevant_docs(
            state["proposal"], ["user", "research", "feedback", "problem"]
        )

        system_prompt = """You are a User Problems Analysis Agent in the RADAR
        framework.

        Your role is to assess USER PROBLEMS:
        - Is the opportunity anchored in real user research and
          feedback?
        - Real user problems being addressed
        - User research and feedback validation
        - Problem-solution fit quality
        - User impact and value delivery

        Rate 1-10 where:
        - 1-3: No real user research or feedback anchoring the
          opportunity
        - 4-6: Limited user research or weak connection to user
          problems
        - 7-10: Strong anchoring in comprehensive user research and
          feedback

        Reference specific user research and feedback data."""

        result = self._run_structured_analysis(
            system_prompt, state["proposal"], relevant_docs
        )

        return {"align_analysis": result.dict()}

    def _cross_validate_findings(self, state: AnalysisState) -> AnalysisState:
        """Cross-validate findings across all lenses"""

        analyses = [
            state.get("review_analysis", {}),
            state.get("analyze_analysis", {}),
            state.get("decide_analysis", {}),
            state.get("align_analysis", {}),
        ]

        # Find conflicting findings
        conflicts = self._identify_conflicts(analyses)

        # Find consensus items
        consensus = self._identify_consensus(analyses)

        # Generate cross-agent insights
        insights = self._generate_cross_insights(analyses)

        # Return updates using proper format
        return {
            "conflicting_findings": conflicts,
            "consensus_items": consensus,
            "cross_agent_insights": insights,
        }

    def _synthesize_results(self, state: AnalysisState) -> AnalysisState:
        """Synthesize all findings into final recommendation"""

        # Calculate overall score
        scores = []
        for analysis_key in [
            "review_analysis",
            "analyze_analysis",
            "decide_analysis",
            "align_analysis",
        ]:
            analysis = state.get(analysis_key, {})
            if "score" in analysis:
                scores.append(analysis["score"])

        overall_score = sum(scores) / len(scores) if scores else 0

        # Generate final recommendation using all context
        synthesis_prompt = f"""
        Based on the RADAR analysis scores:
        - Visibility: {state.get('review_analysis', {}).get('score', 'N/A')}/10
        - Alignment: {state.get('analyze_analysis', {}).get('score', 'N/A')}/10
        - Confidence: {state.get('decide_analysis', {}).get('score', 'N/A')}/10
        - User Problems: {state.get('align_analysis', {}).get('score', 'N/A')}/10

        Provide a VERY BRIEF (2-3 sentences max) recommendation focusing on:
        1. Overall assessment (Ready/Needs Work/Major Issues)
        2. Top priority action needed

        Keep it concise and actionable.
        """

        response = self.llm.invoke(
            [
                SystemMessage(
                    content="You are a senior product strategist. "
                    "Provide CONCISE analysis - maximum 2-3 sentences."
                ),
                HumanMessage(content=synthesis_prompt),
            ]
        )

        final_recommendation = response.content
        confidence_level = (
            "high" if overall_score >= 7 else "medium" if overall_score >= 5 else "low"
        )

        # Determine if refinement is needed
        needs_refinement = (
            len(state.get("conflicting_findings", [])) > 2
            or overall_score < 4
            or state.get("iteration_count", 0) == 0
        )

        return {
            "overall_score": overall_score,
            "final_recommendation": final_recommendation,
            "confidence_level": confidence_level,
            "needs_refinement": needs_refinement,
        }

    def _should_refine(self, state: AnalysisState) -> str:
        """Determine if analysis needs refinement"""
        return (
            "refine"
            if state["needs_refinement"] and state["iteration_count"] < 2
            else "end"
        )

    def _refine_analysis(self, state: AnalysisState) -> AnalysisState:
        """Refine analysis based on conflicts and gaps"""
        iteration_count = state.get("iteration_count", 0) + 1

        # Add refinement logic here - could re-run specific lenses
        # or gather additional information

        return {"iteration_count": iteration_count, "needs_refinement": False}

    def _retrieve_relevant_docs(self, query: str, doc_types: List[str]) -> List[Dict]:
        """Retrieve relevant documents using vector similarity"""
        if not self.vectorstore:
            return []

        # Enhance query with document types
        enhanced_query = f"{query} {' '.join(doc_types)}"

        docs = self.vectorstore.similarity_search(
            enhanced_query,
            k=5,
            filter={"type": {"$in": doc_types}} if doc_types else None,
        )

        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

    def _run_structured_analysis(
        self, system_prompt: str, proposal: str, docs: List[Dict]
    ) -> RadarAnalysisResult:
        """Run structured analysis for a lens"""

        context = "\n\n".join([f"DOCUMENT: {doc['content']}" for doc in docs])

        user_prompt = f"""
        PROPOSAL: {proposal}

        RELEVANT CONTEXT:
        {context}

        Provide structured analysis as JSON matching this schema:
        {{
            "score": <int 1-10>,
            "confidence": <float 0-1>,
            "key_findings": [<findings with evidence>],
            "recommendations": [<specific recommendations>],
            "evidence": [<supporting evidence from docs>],
            "concerns": [<areas of concern>],
            "dependencies": [<dependencies on other work>]
        }}
        """

        response = self.llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )

        try:
            result_dict = json.loads(response.content)
            return RadarAnalysisResult(**result_dict)
        except Exception:
            # Fallback if JSON parsing fails
            return RadarAnalysisResult(
                score=5,
                confidence=0.5,
                key_findings=["Analysis parsing failed"],
                recommendations=["Manual review needed"],
                evidence=["Raw response: " + response.content[:200]],
                concerns=["JSON parsing error"],
                dependencies=[],
            )

    def _identify_conflicts(self, analyses: List[Dict]) -> List[str]:
        """Identify conflicting findings across analyses"""
        # Implementation for conflict detection
        return ["Sample conflict identified"]

    def _identify_consensus(self, analyses: List[Dict]) -> List[str]:
        """Identify consensus items across analyses"""
        # Implementation for consensus detection
        return ["Sample consensus item"]

    def _generate_cross_insights(self, analyses: List[Dict]) -> List[str]:
        """Generate insights from cross-agent analysis"""
        # Implementation for cross-insights
        return ["Sample cross-agent insight"]

    def analyze_proposal(self, proposal: str, documents: List[Dict] = None) -> Dict:
        """Run the complete RADAR analysis workflow"""

        initial_state = AnalysisState(
            proposal=proposal,
            documents=documents or [],
            review_analysis={},
            analyze_analysis={},
            decide_analysis={},
            align_analysis={},
            cross_agent_insights=[],
            conflicting_findings=[],
            consensus_items=[],
            overall_score=0.0,
            final_recommendation="",
            confidence_level="",
            next_steps=[],
            current_step="",
            iteration_count=0,
            needs_refinement=False,
        )

        # Run the workflow
        final_state = self.workflow.invoke(initial_state)

        # Format results for the Streamlit app
        return {
            "overall_score": final_state["overall_score"],
            "recommendation": final_state["final_recommendation"],
            "confidence_level": final_state["confidence_level"],
            "analysis_time": datetime.now(),
            "agent_results": {
                "visibility": final_state["review_analysis"],
                "alignment": final_state["analyze_analysis"],
                "confidence": final_state["decide_analysis"],
                "user_problems": final_state["align_analysis"],
            },
            "cross_insights": final_state["cross_agent_insights"],
            "conflicts": final_state["conflicting_findings"],
            "consensus": final_state["consensus_items"],
        }
