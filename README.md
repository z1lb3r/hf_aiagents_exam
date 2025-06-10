GAIA Agent - Enhanced AI Agent for GAIA Benchmark
This is an enhanced AI agent built with smolagents library specifically designed to excel at the GAIA benchmark questions.
Features

Powered by smolagents: Uses CodeAgent with advanced reasoning capabilities
Strong LLM: Utilizes Llama-3.3-70B-Instruct for superior performance
Multi-tool capabilities: Web search, Python execution, file processing
Precise answer formatting: Custom tool ensures GAIA-compliant answer format
Error handling: Robust error recovery and retry mechanisms

Architecture
Core Components

GAIAAgent: Main agent class using smolagents CodeAgent
format_final_answer: Custom tool for GAIA-compliant answer formatting
Web search integration via DuckDuckGoSearchTool
Python interpreter for calculations and data processing

Answer Formatting
The agent includes a specialized format_final_answer tool that ensures answers meet GAIA requirements:

Numbers: No commas, no units, integer format when applicable
Strings: No articles (a, an, the), no abbreviations
Lists: Comma-separated with proper formatting for each element

Usage

Clone this space to your Hugging Face profile
Set environment variables (automatically handled in HF Spaces):

SPACE_ID: Your space identifier
HF_TOKEN: Your Hugging Face token


Login with your Hugging Face account in the interface
Click "Run Evaluation" to process all questions and submit answers

Model Configuration
The agent uses:

Model: meta-llama/Llama-3.3-70B-Instruct
Tools: Web search, Python interpreter, answer formatter
Additional imports: requests, json, csv, math, statistics, datetime, urllib.parse

You can modify the model by changing the model_id in the GAIAAgent.__init__() method.
Performance Optimization
The agent is optimized for GAIA benchmark through:

Step-by-step reasoning process
Comprehensive tool usage
Precise answer formatting
Error recovery mechanisms
Enhanced system prompts

Customization
To modify the agent:

Change the model: Update model_id in GAIAAgent.__init__()
Add tools: Include additional tools in the tools list
Modify prompts: Adjust the system prompt for different behavior
Add imports: Update additional_authorized_imports for new Python modules

Expected Performance
Target: 30%+ accuracy on GAIA Level 1 questions
The agent is designed to handle various question types including:

Mathematical calculations
Web research tasks
Data analysis
File processing
Multi-step reasoning problems

Dependencies
See requirements.txt for full dependency list. Key packages:

smolagents (with toolkit extras)
gradio (for interface)
pandas (for data handling)
requests (for API calls)

Note
This agent uses exact match evaluation, so answer precision is critical. The custom formatting tool helps ensure compliance with GAIA evaluation criteria.