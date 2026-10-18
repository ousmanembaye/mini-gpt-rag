import gradio as gr

from agent import run_agent


def chat_with_agent(message):
    if not message or not message.strip():
        return "Scrivi una domanda per iniziare."

    try:
        return run_agent(message.strip())

    except Exception as error:
        print(f"Errore durante l'esecuzione dell'agente: {error}")
        return "Si è verificato un errore durante l'elaborazione."


demo = gr.Interface(
    fn=chat_with_agent,

    inputs=gr.Textbox(
        label="La tua domanda",
        placeholder="Es. Quali documenti hai a disposizione?",
        lines=2
    ),

    outputs=gr.Textbox(
        label="Risposta dell'AI Agent",
        lines=6
    ),

    title="Mini AI Agent — Company Policies",

    description=(
        "Un assistente AI che sceglie quale strumento utilizzare "
        "per rispondere alle domande sulle policy aziendali."
    ),

    examples=[
        ["Quali documenti hai a disposizione?"],
        ["Quanti giorni di ferie spettano a un dipendente?"],
        ["Posso lavorare da remoto senza l'approvazione del responsabile?"],
        ["Qual è il budget annuale per la formazione?"]
    ]
)


if __name__ == "__main__":
    demo.launch(inbrowser=True)