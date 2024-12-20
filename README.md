##  Multi-Agent-Workflow-for-Industry-Research-and-Use-Case-Generation

### Usage
- Clone the repo
```
git clone https://github.com/SaiRevanth25/Multi-Agent-Workflow-for-Industry-Research-and-Use-Case-Generation
```
- Install the requirements
```
cd Multi-Agent-Workflow-for-Industry-Research-and-Use-Case-Generation
pip install -r requirements.txt
```
- Create a `.env` file and define your Google [Gemini API Key](https://aistudio.google.com/apikey) and Serper [API key](https://serper.dev/)
```
GEMINI_API_KEY= your_google_gemini_api_key
SERPER_API_KEY= your_serper_api_key
```

- Run the app.py.
```
python app.py
```
- After execution, click on the Gradio link displayed in the terminal output.
- The Gradio link opens a live website where you can:
  - enter the industry name as input.
  - the output will be displayed in a markdown format in the live website.

- The report will be generated and saved as `final_report.md` 
