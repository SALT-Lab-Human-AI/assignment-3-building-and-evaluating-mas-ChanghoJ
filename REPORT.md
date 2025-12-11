## Abstract
Assignment 3 of IS 492 is about building an LLM-powered multi-agent system that searches for HCI-related research papers. The project consists of a planner, researcher, writer, and critic agents, input and output guardrails to prevent harmful content, and an LLM as a judge to evaluate search results conducted by LLM agents. As I ran the evaluation file to test the overall project, the project correctly produce report as a JSON file and a summary text file. CLI interface also work as a user query and return response.

## System Design and Implementation
### Agent
Not much changed in agents, tools, control flow, and models, as they already worked before released as an assignment. There are some change in tools for paper and web searching, which use semanticscholar, tavily, and aiohttp.

For the agents, every setting is already constructed under autogen_agents.py. Python file autogen_agents.py consists of 6 functions that correspond to agents like model client, planner, researcher, writer, and critic, and combined the agents with the research team.

Model client agent sets which Large Language Model (LLM) provider the agent is going to use. For the default YAML setting, Groq is selected as the LLM provider, but also OpenAI set as another configuration. Groq API key is from my personal account created during IS 492 2025 Fall semester, and OpenAI API keys and base URL was provided by TA Yiren for this project.

Planner agent provides prompts that can guide LLM to create a step-by-step, actionable plan from user research queries entered. The function consist of a default prompt that indicate LLM's role as a research planner, and provide 4 prompts to identify topics, prioritize sources, suggest queries, and outline findings.

Researcher agent provides prompts that LLM works as a researcher, accessing web and paper tools and gather evidence based on the Planner. The default prompt set the LLM become an information gatherer, following 5 steps that use web search and paper search tool, check recent and high quality sources, extract key items, gather citations, and gather evidence. Then the function directly define researcher agent to use the web search and paper search tools defined in web_search.py and paper_search.py.

The writer agent combined and provided responses with proper citations. The basic prompt adjust the agent role to synthesize research findings to provide clear responses, and consist of 7 instructions.

A critical agent is intended to provide evaluation on the quality of research and writing, and providing feedback. With the default system message, the Critic agent have 5 criteria of evaluation in its prompt: Relevance (answer to original query), Evidence Quality (source credibility), Completeness (All query addressed), Accuracy (any errors or contradictions), and Clarity (Writing is clear and organized).

The research team function gathers 5 agents defined above, creating a team with round-robin ordering.

### Tools
There are 3 tools in the project: Web search, Paper search, and Citation.

Web search tool supports both Tavily API and Brave Search API. From the choice, Tavily API was selected as the API key already created in the previous lab of IS 492. The web_search.py file contain class WebSearchTool that can be called from other files, and the class consist of asynchronous functions that can set pause its execution and functions. First, function search detects available search providers, which is Tavily at this time. Then, let the provider search on its function (_search_tavily), parse the search results (_parse_tavily_results), and filter out results based on relevant score (_filter_results). All the steps wrapped as a one function call web_search.

Paper search tool, or paper_search.py, supports Semantic Scholar API to accessing academic paper in free. The tool file contains PaperSearchTool class that consist of search (search for academic paper), gathering detail of paper (gathering information like paper ID, authors, etc.), citation (gathering citation of paper), reference (gathering reference of paper), parsing result (parsing and filtering out papers), filtering year (filtering by year), and filtering citation (filtering by citation count). All functions are wrapped in the paper_search function.

The citation tool provides different kinds of citation styles for the papers that agents found with search tools. The citation tool contains APA (7th edition) and MLA format with single or multiple authors format, and the tool also generates a bibliography for APA and MLA.

### Orchestrator
Lastly, the orchestrator file autogen_orchestrator.py use AutoGen library’s RoundRobinGroupChat to set the workflow for using the tools defined above to answer users ’ queries. Its flow starts with processing a query, letting Planner, Researcher, Writer, and Critic agents work on gathering required information and paper. Then orchestrator extract result, agent descriptions, and visualize workflow.

## Safety Design
Three files, safety_manager.py, input_guardrail.py, and output_guardrail.py, are revised for defining and detecting unsafe or irrelevant query to answer. Start from the safety design, GitHub Copilot and the GPT 4.1 model to modify TODO parts in the Python files.

First, input_guardrail.py defines a detailed user input or query guardrail. Class InputGuardrail integrates Guardrail AI just like safety_manager.py does to validate user input, detect toxic contents, prompt injection attempts, figure out irrelevant queries, and check answer length meet with requirement. As the class initialize Guardrail AI, the class validate query in 4 ways: length constrains, toxicity detection, prompt injection detection, and topic relevance checking. All 4 validates have their own function that checking length requirements, checking toxic/harmful and potentially harmful language, checking prompt injection or jailbreak to stop bypassing safety measure, and checking relevancy of the query with HCI research topics.

Second, output_guardrail.py works similar with the input guardrail but checks the output. After initializing the Guardrail AI framework, the guardrail validates the output response to detect personally identifiable information (PII), harmful contents, factual consistency, biased, discriminatory, or stereotyping language, and then removing such validated harmful contents.

Last, safety_manager.py contains a class SafetyManager that integrates the Guardrail AI framework to enforce safety, detect harmful content, handle violations, and record logs with analysis on the safety events. The class initialize Guardrails AI guardrails, check input safety with connecting input_guardrail.py, check output safety with connecting output_guardrail.py, updates the safety log to the logs directory, and sanitizes or gets rid of unsafe contents.

## Evaluation Setup and Results
There are 2 Python files exists in the evaluation on the agent response, evaluator.py and judge.py.

File judge.py work as ‘LLM-as-a-Judge’, which uses LLM to evaluate outputs based on defined criteria under the document. The document contains the class LLMJudge that will be used in evaluator.py, which judge criterion and call LLM API for judgment, defined as prompt: “expert evaluator for multi-agent HCI research responses”. The scoring rubric lies between 0 and 1, and the prompt also mentions tasks that assess response on the criterion only, provided source and ground truth scoring as faithful content, and be precise on reasoning. Then the code parse LLM judgement and aggregate scores. Also, the judge.py document contains examples to compare evaluations.

Document evaluator.py is more like testing file that uses judge.py as importing the LLMJudge class to provide a batch of evaluations and reports on test queries. After initializing the SystemEvaluator class, the code runs test queries in the data directory in the test_queries.json file. Then the code run a single test query, load test queries, and generate report in the outputs directory as JSON and txt format that contains statistics and analysis. The document also contains its own example evaluations with and without the orchestrator to show how the orchestrator connected with the multi-agent system.

### Results
After running python main.py --mode evaluate query, output files were successfully created under the outputs directory. There are two files created, one for a detailed JSON file that reports time, success and failed run queries, statistic metrics that are defined on judge.py file, queries and responses. The evaluation summary file is the summary version of the JSON file that provides overall scores and the best and worst. CLI interface also succeed as user input, and LLM output is correctly done on CLI systemqueries.

## Discussion & Limitations
### Discussion
Since this is the last assignment of IS 492, this assignment works as a combination of labs done during the semester. The assignment definitely summing up previous learnings on vibe coding, agent (compared with workflow), multi-agent, guardrails on harmful contents, and evaluation. I can see the full task flow from building agents by defining each agents’ role, defining and preventing potential harmful and toxic inputs by the user and outputs by LLM, and lastly judge and provide score on the result created and parsed by LLM agents. Then, obviously, vibe coding was the key part of building the actual TODOs in the unfinished original documents.

I personally liked how the learning of the entire semester showed in the single assignment, this reminds me of the final project of IS 477, data and information management, that build repository of research or project to reproduce tasks.

### Limitations
Now, I strongly feel there is a problem with vibe coding. GitHub Copilot sometimes change documents drastically over TODOs, which makes it hard to return and fix when an error occurs. I already faced with several errors, which were related with running CLI through the main.py file. I fixed the error, but it took several queries on GitHub Copilot. Also, related with vibe code, I feel my coding skill is lacked, which I personally hard for me to enjoy this assignment, even though I liked the concept of summary of learning of this assignment. Even with vibe code, I had a hard time on implementing the TODO parts, as well as the CLI interface by the over revising problems mentioned above.

For future work, maybe a more detailed inspection of the vibe code is needed to avoid blind coding.

## References
None. The writing is strictly about reviewing the HCI Research Paper finder agent project.
