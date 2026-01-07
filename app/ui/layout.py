"""Gradio layout for the WebUI."""

from __future__ import annotations

import gradio as gr

from app.ui.callbacks import (
    clear_history,
    handle_user_message,
    handle_rag_message,
    index_rag_files,
)
from app.config import get_config


def build_demo() -> gr.Blocks:
    """Build the Gradio WebUI."""
    cfg = get_config()

    busy_css = """
    .gradio-container [aria-busy='true'] {
      outline: none !important;
      box-shadow: none !important;
      border-color: transparent !important;
    }
    .gradio-container [aria-busy='true'] * {
      outline: none !important;
      box-shadow: none !important;
    }
    .gradio-container :focus {
      outline: none !important;
      box-shadow: none !important;
    }
    """

    with gr.Blocks(title="LLM Text WebUI", css=busy_css) as demo:
        gr.HTML(f"""
            <h1 style="text-align: center; margin-bottom: 0.5rem;">LLM Text WebUI</h1>
            <p style="text-align: left;">model: <code>{cfg.model_id}</code></p>
        """)

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Chat")

                    with gr.Row():
                        with gr.Column(scale=4):
                            user_input = gr.Textbox(
                                placeholder="Type a message and press Enter",
                                show_label=False,
                                autofocus=False,
                            )
                            clear_btn = gr.Button("Clear")

                        with gr.Column(scale=1):
                            send_btn = gr.Button("Send", variant="primary")
                            stop_btn = gr.Button("Stop", variant="stop")

                with gr.Column(scale=2):
                    system_prompt = gr.Textbox(
                        label="System prompt",
                        lines=4,
                        placeholder="Optional instructions for the model",
                    )

                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.5,
                        value=cfg.default_temperature,
                        step=0.05,
                        label="Temperature",
                    )

                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=cfg.default_top_p,
                        step=0.05,
                        label="Top-p",
                    )

                    max_new_tokens = gr.Slider(
                        minimum=16,
                        maximum=2048,
                        value=cfg.default_max_new_tokens,
                        step=16,
                        label="Max new tokens",
                    )

                    top_k = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=cfg.default_top_k,
                        step=1,
                        label="Top-k",
                    )

            chat_inputs = [
                user_input,
                chatbot,
                temperature,
                top_p,
                max_new_tokens,
                top_k,
                system_prompt,
            ]
            chat_outputs = [chatbot, user_input]

            send_event = send_btn.click(
                handle_user_message,
                inputs=chat_inputs,
                outputs=chat_outputs,
                show_progress="hidden",
            )
            submit_event = user_input.submit(
                handle_user_message,
                inputs=chat_inputs,
                outputs=chat_outputs,
                show_progress="hidden",
            )

            clear_btn.click(clear_history, inputs=None, outputs=chat_outputs)

            stop_btn.click(
                fn=None,
                inputs=None,
                outputs=None,
                cancels=[send_event, submit_event],
            )

        with gr.Tab("RAG Chat"):
            with gr.Column():
                rag_chatbot = gr.Chatbot(label="RAG Chat")

                with gr.Row():
                    rag_user_input = gr.Textbox(
                        placeholder="Ask a question about your documents",
                        show_label=False,
                        autofocus=False,
                    )
                    rag_send_btn = gr.Button("Send", variant="primary")
                    rag_stop_btn = gr.Button("Stop", variant="stop")
                    rag_clear_btn = gr.Button("Clear")

                rag_files = gr.File(
                    label="RAG documents (text)",
                    file_count="multiple",
                    type="filepath",
                )
                index_btn = gr.Button("Index documents")

                rag_status = gr.Markdown("", elem_id="rag-status")

                with gr.Accordion("RAG settings", open=False):
                    rag_system_prompt = gr.Textbox(
                        label="RAG system prompt",
                        lines=4,
                        placeholder="Optional instructions for RAG answers",
                    )

                    rag_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.5,
                        value=cfg.default_temperature,
                        step=0.05,
                        label="Temperature",
                    )

                    rag_top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=cfg.default_top_p,
                        step=0.05,
                        label="Top-p",
                    )

                    rag_max_new_tokens = gr.Slider(
                        minimum=16,
                        maximum=2048,
                        value=cfg.default_max_new_tokens,
                        step=16,
                        label="Max new tokens",
                    )

                    rag_top_k = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=cfg.default_top_k,
                        step=1,
                        label="Top-k",
                    )

                    top_k_docs = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=4,
                        step=1,
                        label="Top documents",
                    )

            rag_inputs = [
                rag_user_input,
                rag_chatbot,
                rag_temperature,
                rag_top_p,
                rag_max_new_tokens,
                rag_top_k,
                rag_system_prompt,
                top_k_docs,
            ]
            rag_outputs = [rag_chatbot, rag_user_input, rag_status]

            index_btn.click(
                index_rag_files,
                inputs=[rag_files],
                outputs=[rag_status],
            )

            rag_submit_event = rag_user_input.submit(
                handle_rag_message,
                inputs=rag_inputs,
                outputs=rag_outputs,
                show_progress="hidden",
            )

            rag_send_event = rag_send_btn.click(
                handle_rag_message,
                inputs=rag_inputs,
                outputs=rag_outputs,
                show_progress="hidden",
            )

            rag_stop_btn.click(
                fn=None,
                inputs=None,
                outputs=None,
                cancels=[rag_send_event, rag_submit_event],
            )

            def _clear_rag() -> tuple[list[list[tuple[str, str]]], str, str]:
                return [], "", ""

            rag_clear_btn.click(
                _clear_rag,
                inputs=None,
                outputs=rag_outputs,
            )

        demo.queue(default_concurrency_limit=cfg.queue_concurrency_limit)

    return demo
