import gradio as gr

from rag import answer_query


def run_rag(query):
    """Esegue il RAG e prepara i risultati per l'interfaccia."""

    if not query or not query.strip():
        return (
            "Inserisci una domanda.",
            "Nessuna fonte",
            "N/A"
        )

    try:
        answer, sources, best_score = answer_query(query.strip())

        sources_text = (
            ", ".join(sources)
            if sources
            else "Nessuna fonte disponibile"
        )

        score_text = f"{best_score:.4f}"

        return answer, sources_text, score_text

    except Exception:
        # I dettagli dell'errore restano nel terminale.
        import traceback
        traceback.print_exc()

        return (
            "Si è verificato un errore durante l'elaborazione.",
            "Nessuna fonte",
            "N/A"
        )


demo = gr.Interface(
    fn=run_rag,

    inputs=gr.Textbox(
        label="La tua domanda",
        placeholder="Es. Quanti giorni di ferie spettano a un dipendente?",
        lines=2
    ),

    outputs=[
        gr.Textbox(label="Risposta del RAG"),
        gr.Textbox(label="Fonti recuperate"),
        gr.Textbox(label="Similarità embedding migliore")
    ],

    title="Mini-RAG aziendale",

    description=(
        "Fai una domanda sui documenti aziendali. "
        "Il sistema recupera le informazioni pertinenti "
        "e genera una risposta."
    ),

    examples=[
        ["Quanti giorni di ferie spettano a un dipendente?"],
        ["Quanti giorni a settimana posso lavorare da remoto?"],
        ["Posso lavorare da remoto senza l'approvazione del responsabile?"],
        ["Qual è il budget annuale per la formazione?"]
    ]
)


if __name__ == "__main__":
    demo.launch(inbrowser=True)