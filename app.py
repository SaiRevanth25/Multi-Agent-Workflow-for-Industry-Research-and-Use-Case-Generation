import gradio as gr

def run(industry_name):
    from crewai import Crew
    from Agents import agents
    from Tasks import tasks

    # Define the Crew with agents and tasks
    crew = Crew(
        agents=agents,
        tasks=tasks,
        max_rpm=29,  # requests per min limit
        verbose=True,
    )

    input_data = {
        'industry_name': industry_name
    }

    # Kick off the task
    crew.kickoff(inputs=input_data)

    try:
        with open("final_report.md", "r") as file:
            markdown_content = file.read()
        return markdown_content
    except FileNotFoundError:
        return "Error: The output file was not generated."

with gr.Blocks() as app:
    gr.Markdown("# Market Research Agent")
    with gr.Row(equal_height=True):
        textbox = gr.Textbox(lines=1, label='Industry Name', placeholder="Enter industry name")
        button = gr.Button('Submit')

    outputs = gr.Markdown()  

    button.click(
        fn=run,  
        inputs=textbox,  
        outputs=outputs  
    )

app.launch(share=True)
