> **Simplified Research Code**  
> This repository contains a streamlined re-implementation of the Shapefile-Agent core functionality for research purposes.

# Shapefile-Agent: LLM-Driven Automated GIS Shapefile Processing

## 🚀 Research Overview
Shapefile-Agent is a multi-agent system leveraging large language models (LLMs) to automatically **analyze**, **modify**, and **generate** GIS Shapefile vector data via natural language commands.  
A **Supervisor** agent decomposes user requests into subtasks, and specialized **Worker** agents execute Python code (via a REPL) or perform web searches to fulfill each step.

- **Publication Year**: 2024  
- **Topic**: Real-time vector data manipulation with LLM-based multi-agent workflows  
- **Key Results**:  
  - Simple task success rate: **96%** (≈20 pp improvement over baseline GPT-4o)  
  - Complex task success rate: **88%**  
  - Real-time code execution & visualization for transparency and user collaboration  


## ⚙️ Setup & Usage

1. **Install dependencies**
    pip install -r requirements.txt

2. **Configure environment variables**
    Create a .env file in the project root and add your API keys:

    OPENAI_API_KEY=your_openai_api_key
    TAVILY_API_KEY=your_tavily_api_key
    LANGCHAIN_API_KEY=your_langchain_api_key

3. **Run the agent**
    python ShapefileAgent.py
    Follow the interactive prompt to issue natural language commands.
    The Supervisor and Worker agents will coordinate to generate and execute code, then display results.


## 🔍 Selected References
Shunyu et al., ReAct: Synergizing Reasoning and Acting in Language Models (2023)
Weize et al., AgentVerse: A Multi-Agent Framework for Complex Reasoning (2023)
Zhenlong et al., GeoGPT: Large-Scale LLM-Driven Geospatial Analysis (2023)


## 🙏 Acknowledgments
Advisor: Prof. Yun-Mi Park
Research Team: Hyunjun Go, Hyunjun So
Support: Graduate assistant Yong-Hun Cho
Tools & Services: OpenAI GPT-4o, LangChain, Tavily Search, GeoPandas