"""Gradio Blocks layout for the WebUI."""

from __future__ import annotations

import gradio as gr

from app.ui.callbacks import clear_history, handle_user_message
from app.config import get_config


def build_demo() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    cfg = get_config()

    with gr.Blocks(title="LLM Text WebUI") as demo:
        gr.HTML(f"""
            <h1 style="text-align: center; margin-bottom: 0.5rem;">LLM Text WebUI</h1>
            <p style="text-align: left;">Model: <code>{cfg.model_id}</code></p>
        """)

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

        inputs = [
            user_input,
            chatbot,
            temperature,
            top_p,
            max_new_tokens,
            top_k,
            system_prompt,
        ]
        outputs = [chatbot, user_input]

        send_event = send_btn.click(
            handle_user_message,
            inputs=inputs,
            outputs=outputs,
        )
        submit_event = user_input.submit(
            handle_user_message,
            inputs=inputs,
            outputs=outputs,
        )

        clear_btn.click(clear_history, inputs=None, outputs=outputs)

        stop_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[send_event, submit_event],
        )

        demo.queue(default_concurrency_limit=1)

    return demo
