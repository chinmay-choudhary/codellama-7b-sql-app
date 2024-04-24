from langchain_community.llms import VLLM
from datasets import  load_dataset
from langchain.chains import LLMChain
import gradio as gr
from langchain_core.prompts import PromptTemplate

ds = load_dataset("chinmayc3/synthetic-sql",split="train")
print('loading model')
llm = VLLM(
    model="chinmayc3/codellama-sql-7b-quantized",
    quantization="AWQ",
    trust_remote_code=True,
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.8,
)
print('loaded model')
template = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.
### Input:
{prompt}

### Context:
{schema}

### Response:
"""

prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

def generate_response(prompt, schema):
    output = llm_chain.invoke({'prompt':prompt,'schema':schema})
    return output['text']


inputs = [
    gr.Textbox(label="Enter Prompt"),
    gr.Textbox(label="Enter Schema")
]
outputs = gr.Textbox(label="Generated Schema")


app = gr.Interface(
    fn=generate_response,
    inputs=inputs,
    outputs=outputs,
    title="SQL Code Generator Given Schema",
    description="Enter a table schema and a sql prompt to get generated sql.",
    examples=[
        [ds[0]['prompt'],ds[0]['context']],
        [ds[1]['prompt'],ds[1]['context']],
        [ds[2]['prompt'],ds[2]['context']],
        [ds[3]['prompt'],ds[3]['context']],
    ]
)

app.launch()

