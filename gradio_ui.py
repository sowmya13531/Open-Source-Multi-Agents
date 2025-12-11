import gradio as gr

def assistant(query):
    return team.run(query)

gr.Interface(fn=assistant, inputs="text", outputs="text", title="Open-Source Multi-Agent AI").launch()
