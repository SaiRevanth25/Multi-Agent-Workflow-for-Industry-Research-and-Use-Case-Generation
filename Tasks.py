from crewai import Task
from Agents import industry_research_agent, resource_asset_allocation, orchestrator_agent, use_case_generation

industry_task = Task(
            description=(
                """Analyze the leading market reports for {industry_name} industry. 
                    To gather latest relevant industry reports, news, and competitor profiles. Summarizes fetched data into key points, focusing on industry trends, 
                    competitive positioning, and notable AI/ML use cases. Classifies the industry into key segments (e.g., operations, supply chain, customer experience, etc) 
                    and identifies focus areas."""
            ),
            expected_output=(
                "Structure report focusing on areas and opportunities for AI/ML integration for {industry_name} industry."
            ),
            agent= industry_research_agent,
        )

use_case_task = Task(
                description=(
                    """Analyzes the industry report provided by the Researcher to identify high-impact areas for AI and ML innovation.
                        Uses an LLM to generate specific AI and GenAI use cases relevant to each identified focus area. Each use-case is generated backed by accessible resources.
                        Refines and ranks use cases based on relevance, impact, and feasibility."""
                ),
                expected_output=(
                    """List of high-potential areas where AI/ML solutions could make a difference in {industry_name} industry. 
                        List of creative, feasible AI/ML use cases, with a description and benifits. Explains different 10 use-case situations.
                        Ranked and refined list of numbered use cases, each with explanation and its potential impact and implementation difficulty."""
                ),
                agent= use_case_generation,
                context=[industry_task],
            )  

resource_collection_task = Task(
            description=(
                """Searches for datasets on platforms like Kaggle, HuggingFace, and GitHub that could be used to implement each use case. 
                    Searches for relevant pretrained models (e.g., on HuggingFace, kaggle) to support implementation. 
                    Makes sure that the link provided is not a dead link. Compiles all resources into a markdown file with clickable links.."""
            ),
            expected_output=(
                """List of datasets and model links, categorized for each use case.  List of pretrained model links and descriptions.
                    The final report should be with organized, clickable resource links for each use case."""
            ),
            agent=resource_asset_allocation,
            context=[use_case_task],
        )

compilation_task = Task(
            description=(
                """
                Compiles the insights from each agent into a cohesive final proposal report, including a industry summary, use cases, and resource links.
                Provide a detailed analysis of potential risks and suggest mitigation strategies."""
            ),
            expected_output=(
                """Orchestrated output from all agents for final report compilation. 
                   Final proposal report in markdown format with clickable resource links"""
            ),
            agent=orchestrator_agent,
            context=[industry_task, use_case_task, resource_collection_task],
            output_file= 'final_report.md',
        )

tasks = [industry_task, use_case_task, resource_collection_task, compilation_task]
