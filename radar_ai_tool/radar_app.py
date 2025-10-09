"""
RADAR - AI-Powered Product Strategy Review Tool
==============================================

A prototype system that reviews product proposals against organizational strategy
using the four RADAR lenses: Visibility, Alignment, Confidence, User Problems.

"""

import os
from datetime import datetime
from typing import Dict, List

import plotly.graph_objects as go
import streamlit as st

# Real LLM integration - try LangGraph first, fallback to simple LangChain
try:
    from radar_langgraph_agent import RADARLangGraphAgent

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    try:
        from radar_llm_integration import ProductionLLMAgent

        LANGCHAIN_AVAILABLE = True
        st.warning("Using Basic LangChain Integration (LangGraph not available)")
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        st.error(
            "No LLM integration available. Please install requirements: " "uv sync"
        )


# Add Streamlit caching for performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_analysis(proposal_text: str, knowledge_base_hash: str = "") -> Dict:
    """Cached analysis function to avoid repeated API calls"""
    # This will be called by Streamlit's caching mechanism
    # The actual implementation will be in the analyze_proposal method
    return None


@st.cache_data
def load_sample_documents() -> List[Dict]:
    """Cache sample documents loading"""
    sample_docs = []
    sample_dir = "sample_documents"
    if os.path.exists(sample_dir):
        for filename in os.listdir(sample_dir):
            if filename.endswith(".md"):
                with open(os.path.join(sample_dir, filename), "r") as f:
                    sample_docs.append(
                        {
                            "title": filename.replace(".md", "").replace("_", " "),
                            "content": f.read(),
                            "filename": filename,
                        }
                    )
    return sample_docs


class RADARAnalyzer:
    """Main RADAR analysis orchestrator"""

    def __init__(self):
        # Check for available LLM integration
        if not (LANGGRAPH_AVAILABLE or LANGCHAIN_AVAILABLE):
            st.error(
                "No LLM integration available. Please install requirements: " "uv sync"
            )
            st.stop()

        # Require API key
        if not st.secrets.get("openai_api_key"):
            st.error(
                "OpenAI API key not found. Please add it to .streamlit/secrets.toml"
            )
            st.stop()

        api_key = st.secrets["openai_api_key"]

        # Initialize appropriate agent system
        if LANGGRAPH_AVAILABLE:
            self.agent = RADARLangGraphAgent(api_key)
            self.agent_type = "langgraph"
        else:
            self.agents = {
                "visibility": ProductionLLMAgent("visibility", api_key),
                "alignment": ProductionLLMAgent("alignment", api_key),
                "confidence": ProductionLLMAgent("confidence", api_key),
                "user_problems": ProductionLLMAgent("user_problems", api_key),
            }
            self.agent_type = "langchain"

        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> List[Dict]:
        """Load documents from sample markdown files"""
        knowledge_base = []

        # Load actual sample documents
        docs_to_load = [
            ("sample_documents/Q3_2024_Strategy.md", "strategy"),
            ("sample_documents/Team_B_Roadmap.md", "roadmap"),
            ("sample_documents/User_Research_March_2024.md", "research"),
            (
                "sample_documents/Failed_Social_Media_Project_Postmortem.md",
                "postmortem",
            ),
        ]

        for file_path, doc_type in docs_to_load:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    knowledge_base.append(
                        {
                            "title": file_path.split("/")[-1]
                            .replace(".md", "")
                            .replace("_", " "),
                            "type": doc_type,
                            "content": content,
                            "source": file_path,
                            "format": "markdown",
                            "last_updated": "2024-09-01",
                        }
                    )
            except FileNotFoundError:
                st.warning(f" Document not found: {file_path}")

        return knowledge_base

    def analyze_proposal(self, proposal: str) -> Dict:
        """Run full RADAR analysis with caching"""

        if self.agent_type == "langgraph":
            # Use advanced LangGraph workflow
            return self.agent.analyze_proposal(proposal, self.knowledge_base)
        else:
            # Use simple LangChain agents with optimized parallel processing
            import concurrent.futures

            results = {}

            def analyze_single_agent(agent_name, agent):
                """Analyze with a single agent"""
                try:
                    return agent_name, agent.analyze(proposal, self.knowledge_base)
                except Exception as e:
                    return agent_name, {
                        "score": 0,
                        "findings": [f"Error: {str(e)}"],
                        "recommendations": [],
                    }

            # Add progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Run all agents in parallel for speed (reduced workers for stability)
            from time import time as get_time

            start_time = get_time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_to_agent = {
                    executor.submit(analyze_single_agent, agent_name, agent): agent_name
                    for agent_name, agent in self.agents.items()
                }

                completed = 0
                total_agents = len(self.agents)
                for future in concurrent.futures.as_completed(future_to_agent):
                    agent_name, result = future.result()
                    results[agent_name] = result
                    completed += 1
                    progress = completed / total_agents
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Completed {agent_name} analysis ({completed}/{total_agents})"
                    )

            progress_bar.empty()
            status_text.empty()

            analysis_time = get_time() - start_time
            st.success(f"Analysis completed in {analysis_time:.2f} seconds")

            # Calculate overall score
            scores = [
                results[agent]["score"]
                for agent in results
                if "score" in results[agent]
            ]
            overall_score = sum(scores) / len(scores) if scores else 5

            # Generate recommendation
            if overall_score >= 7:
                recommendation = "PROCEED - Strong strategic fit"
                rec_color = "🟢"
            elif overall_score >= 5:
                recommendation = "PROCEED WITH CAUTION - Address concerns first"
                rec_color = "🟡"
            else:
                recommendation = "PAUSE - Significant issues to resolve"
                rec_color = "🔴"

            return {
                "overall_score": overall_score,
                "recommendation": recommendation,
                "rec_color": rec_color,
                "analysis_time": datetime.now(),
                "agent_results": results,
                "proposal_text": proposal,
            }


def create_score_chart(scores: Dict[str, int]) -> go.Figure:
    """Create a radar chart for the four RADAR scores"""

    categories = ["Visibility", "Alignment", "Confidence", "User Problems"]
    values = [
        scores["visibility"],
        scores["alignment"],
        scores["confidence"],
        scores["user_problems"],
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="RADAR Scores",
            line_color="rgb(46, 134, 171)",
            fillcolor="rgba(46, 134, 171, 0.3)",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        title="RADAR Analysis Scores",
        height=400,
    )

    return fig


def create_score_bars(scores: Dict[str, int]) -> go.Figure:
    """Create bar chart for scores with color coding"""

    categories = ["Visibility", "Alignment", "Confidence", "User Problems"]
    values = [
        scores["visibility"],
        scores["alignment"],
        scores["confidence"],
        scores["user_problems"],
    ]

    # Color code based on score
    colors = []
    for score in values:
        if score >= 7:
            colors.append("#2eab43")  # Green
        elif score >= 5:
            colors.append("#f39c12")  # Orange
        else:
            colors.append("#e74c3c")  # Red

    fig = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=colors)])

    fig.update_layout(title="RADAR Lens Scores", yaxis=dict(range=[0, 10]), height=300)

    return fig


def main():
    """Main Streamlit application"""

    st.set_page_config(
        page_title="RADAR - AI Product Strategy Review",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for better styling
    st.markdown(
        """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:'
                'wght@300;400;500;600;700&display=swap');

    /* Main app styling */
    .main > div {
        font-family: 'Inter', sans-serif;
    }

    /* Title styling */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #1f2937;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Subtitle styling */
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Headers styling */
    h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #374151;
        font-weight: 600;
    }

    /* Card-like containers */
    .stMarkdown > div {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 1rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }

    /* Button styling */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
    }

    /* Metrics styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
    }

    /* Status indicators */
    .status-excellent {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
    }

    .status-good {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
    }

    .status-needs-work {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        text-align: center;
    }

    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
    }

    /* Text area styling */
    .stTextArea > div > div > textarea {
        font-family: 'Inter', sans-serif;
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        transition: border-color 0.3s ease;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Info boxes */
    .stAlert > div {
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        background: #f8fafc;
        border-radius: 8px;
    }

    /* Custom tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        border-radius: 12px;
        padding: 8px;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        color: #6b7280;
        font-weight: 500;
        padding: 12px 24px;
        transition: all 0.3s ease;
        border: 1px solid #e5e7eb;
        font-family: 'Inter', sans-serif;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: 1px solid transparent;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }

    /* Big blue headline styling */
    .big-blue-headline {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #2563eb;
        margin-bottom: 1.5rem;
        text-align: left;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header with improved styling
    st.markdown("<h1>RADAR</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">AI Product Strategy Review Tool - '
        "Align your product proposals with organizational strategy</div>",
        unsafe_allow_html=True,
    )

    # Main interface with improved spacing
    st.markdown("---")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown(
            '<div class="big-blue-headline">Submit Product Proposal</div>',
            unsafe_allow_html=True,
        )
        proposal_text = st.text_area(
            "Describe your product proposal:",
            height=200,
            placeholder="Example: We want to build a chat feature to increase "
            "user engagement. The feature would allow users to communicate "
            "in real-time and share files...",
            help="Provide a detailed description of your product proposal "
            "including objectives, features, and expected outcomes.",
        )

        # RADAR Framework Explanation
        st.markdown("---")
        st.markdown("### How RADAR Analysis Works")
        st.markdown(
            """
        **RADAR evaluates product opportunities against organizational
        strategy** by analyzing your proposal through four strategic lenses.
        Our AI agents review your documents, research, and strategic
        materials to strengthen decision-making across these crucial areas:

        **Visibility** - Is this theme visible across leadership reviews
        and other teams' initiatives?
        *Checks for organizational awareness and whether similar work
        exists elsewhere*

        **Alignment** - Does it connect clearly to the strategy and OKRs?
        *Validates direct connection to documented strategic priorities
        and key results*

        **Confidence** - Are the assumptions valid, and do the leading
        indicators tie back to lagging KPIs?
        *Assesses assumption validity and metric reliability*

        **User Problems** - Is the opportunity anchored in real user
        research and feedback?
        *Confirms user validation and problem-solution fit*

        Each lens receives a score from 1-10, with higher scores indicating
        stronger strategic positioning.
        """
        )

        # Analysis button with improved styling
        st.markdown("---")
        if st.button(
            "Run RADAR Analysis",
            type="primary",
            disabled=not proposal_text,
            use_container_width=True,
        ):
            # Initialize analyzer
            analyzer = RADARAnalyzer()

            # Show detailed progress with status updates
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Initializing RADAR analysis...")
            progress_bar.progress(10)

            status_text.text("Running parallel analysis across all four lenses...")
            progress_bar.progress(30)

            from time import time as get_time

            start_time = get_time()
            results = analyzer.analyze_proposal(proposal_text)

            progress_bar.progress(80)
            status_text.text("Calculating scores and generating recommendations...")

            progress_bar.progress(100)
            elapsed = get_time() - start_time
            status_text.text(f"Analysis complete! ({elapsed:.1f}s)")

            # Store results in session state
            st.session_state["analysis_results"] = results

            # Clear progress indicators after a moment
            import time

            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

    with col2:
        st.markdown("### Analysis Status")

        if "analysis_results" not in st.session_state:
            st.info("Submit proposal for analysis")
        else:
            results = st.session_state["analysis_results"]

            # Score-based styling
            score = results["overall_score"]
            if score >= 7:
                status_class = "status-excellent"
                status_text = "Excellent"
            elif score >= 5:
                status_class = "status-good"
                status_text = "Good"
            else:
                status_class = "status-needs-work"
                status_text = "Needs Work"

            st.markdown(
                f'<div class="{status_class}">{status_text}: '
                f'{results["overall_score"]:.1f}/10</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{results['recommendation']}**")

    # Results section
    if "analysis_results" in st.session_state:
        results = st.session_state["analysis_results"]

        st.markdown("---")
        st.markdown("### Detailed Analysis Results")

        # Score visualization with improved layout
        col1, col2 = st.columns(2, gap="large")

        with col1:
            scores = {k: v["score"] for k, v in results["agent_results"].items()}
            fig_radar = create_score_chart(scores)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            fig_bars = create_score_bars(scores)
            st.plotly_chart(fig_bars, use_container_width=True)

        # Detailed analysis for each lens with improved styling
        st.markdown("---")
        st.markdown("### Lens-by-Lens Analysis")

        # Create tabs for each lens
        tabs = st.tabs(["Visibility", "Alignment", "Confidence", "User Problems"])

        lens_mapping = {
            0: "visibility",
            1: "alignment",
            2: "confidence",
            3: "user_problems",
        }

        for i, tab in enumerate(tabs):
            with tab:
                lens_key = lens_mapping[i]
                lens_results = results["agent_results"][lens_key]

                # Score display with styling
                col1, col2 = st.columns([1, 3])
                with col1:
                    score = lens_results["score"]
                    if score >= 7:
                        st.markdown(
                            f'<div class="status-excellent">Score: {score}/10</div>',
                            unsafe_allow_html=True,
                        )
                    elif score >= 5:
                        st.markdown(
                            f'<div class="status-good">Score: {score}/10</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="status-needs-work">Score: {score}/10</div>',
                            unsafe_allow_html=True,
                        )

                with col2:
                    findings = lens_results.get("findings", [])
                    if findings:
                        st.markdown(
                            '<div style="margin-bottom: 0.5rem;">'
                            "<strong>Key Findings:</strong></div>",
                            unsafe_allow_html=True,
                        )
                        for finding in findings:
                            st.info(f"• {finding}")

                    recommendations = lens_results.get("recommendations", [])
                    if recommendations:
                        st.markdown(
                            '<div style="margin-bottom: 0.5rem; margin-top: 1rem;">'
                            "<strong>Recommendations:</strong></div>",
                            unsafe_allow_html=True,
                        )
                        for rec in recommendations:
                            st.markdown(f"• {rec}")

                    evidence = lens_results.get("evidence", [])
                    if evidence:
                        st.markdown(
                            '<div style="margin-bottom: 0.5rem; margin-top: 1rem;">'
                            "<strong>Evidence:</strong></div>",
                            unsafe_allow_html=True,
                        )
                        for ev in evidence:
                            st.markdown(f"• {ev}")


if __name__ == "__main__":
    main()
