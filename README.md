# RECAP - AI-Powered Product Strategy Review Tool

A production-ready AI system that reviews product proposals against organizational strategy using the four RECAP lenses: Visibility, Alignment, Confidence, User Problems.

## Overview

RECAP helps product teams validate their proposals by analyzing them against:
- **Visibility**: Conflicts with other teams and historical projects
- **Alignment**: Strategic fit with company goals and OKRs
- **Confidence**: Assumption validation and success metrics
- **User Problems**: Real user needs and problem-solution fit

## Features

- Real OpenAI GPT integration for intelligent analysis
- Document-grounded insights using organizational knowledge
- Interactive radar charts and scoring visualizations
- Structured recommendations and action items
- Clean, professional interface without distracting icons

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements_recap.txt
   ```

2. Configure your OpenAI API key in `.streamlit/secrets.toml`:
   ```toml
   openai_api_key = "your-api-key-here"
   ```

3. Run the application:
   ```bash
   streamlit run recap_app.py
   ```

## File Structure

- `recap_app.py` - Main Streamlit application
- `recap_llm_integration.py` - OpenAI/LangChain integration
- `requirements_recap.txt` - Python dependencies
- `.streamlit/secrets.toml` - API keys and configuration
- `sample_documents/` - Strategy documents for analysis context

## Usage

1. Enter your product proposal in the text area
2. Click "Run RECAP Analysis" to get AI-powered insights
3. Review the four-lens analysis with scores and recommendations
4. Use the insights to refine your proposal or planning approach

## Sample Documents

The system includes realistic strategy documents:
- Q3 2024 Strategy (enterprise focus)
- Team B Roadmap (platform development)
- User Research March 2024 (communication preferences)
- Failed Social Media Project Postmortem (lessons learned)

## Technical Architecture

- **Frontend**: Streamlit with Plotly visualizations
- **AI Backend**: OpenAI GPT-3.5-turbo via LangChain
- **Document Processing**: Markdown file ingestion
- **Scoring**: Multi-agent analysis with structured JSON responses

## Production Considerations

- API rate limiting and error handling implemented
- Fallback responses for LLM failures
- Modular agent architecture for easy extension
- Clean separation of concerns between UI and AI logic
