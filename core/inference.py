# Copyright (C) 2026 Gerald Teeple
#
# This file is part of ANIMA.
#
# ANIMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ANIMA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ANIMA. If not, see <https://www.gnu.org/licenses/>.

"""Inference engine — supports both local (llama-cpp-python) and remote (HTTP server) backends.

Backend is selected by config: inference_mode = "local" or "server"
When using server mode, llama-server.exe runs on Windows with Vulkan GPU acceleration
and this module talks to it via OpenAI-compatible HTTP API.
"""

import os
import json
import logging
import requests
import toml

logger = logging.getLogger(__name__)


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.toml")
    return toml.load(os.path.normpath(config_path))


class InferenceEngine:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.model = None
        self.mode = self.config["hardware"].get("inference_mode", "local")
        self.server_url = self.config["hardware"].get("inference_server", "http://127.0.0.1:8080")

    def load(self):
        if self.mode == "server":
            # Verify server is reachable
            try:
                r = requests.get(f"{self.server_url}/health", timeout=5)
                if r.status_code == 200:
                    logger.info(f"Connected to inference server at {self.server_url}")
                    return self
                else:
                    raise ConnectionError(f"Server returned status {r.status_code}")
            except requests.ConnectionError:
                raise ConnectionError(
                    f"Cannot reach inference server at {self.server_url}. "
                    f"Start the server with start-server.bat on Windows first."
                )
        else:
            # Local mode — load model in-process
            from llama_cpp import Llama

            model_cfg = self.config["model"]
            hw_cfg = self.config["hardware"]

            model_path = os.path.join(
                os.path.dirname(__file__), "..", model_cfg["base_model_path"]
            )
            model_path = os.path.normpath(model_path)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")

            self.model = Llama(
                model_path=model_path,
                n_ctx=model_cfg["context_window"],
                n_threads=hw_cfg["threads"],
                verbose=False,
            )
        return self

    def _build_messages(self, prompt, system_context=""):
        messages = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        # Hunyuan-A13B: prepend /think to enable chain-of-thought reasoning
        if self.config["hardware"].get("thinking_prefix", False):
            prompt = "/think\n" + prompt
        messages.append({"role": "user", "content": prompt})
        return messages

    def _server_payload(self, messages, max_tokens, stream=False, temperature=None):
        model_cfg = self.config["model"]
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature if temperature is not None else model_cfg["temperature"],
            "top_p": model_cfg["top_p"],
            "stream": stream,
            "stop": ["<|im_start|>", "<|im_end|>"],
            "repeat_penalty": model_cfg.get("repeat_penalty", 1.15),
            "frequency_penalty": model_cfg.get("frequency_penalty", 0.1),
        }
        # Ollama requires model name in OpenAI-compatible endpoint
        ollama_model = self.config.get("hardware", {}).get("ollama_model")
        if ollama_model:
            payload["model"] = ollama_model
        return payload

    @staticmethod
    def _clean_response(text):
        """Strip ChatML artifacts, thinking chains, and trailing name bleed.

        Handles:
        - ChatML tokens leaked through
        - Hunyuan <think>...</think> blocks
        - GPT OSS reasoning dumps (chain-of-thought before the answer)
        - Trailing turn-boundary bleed
        """
        import re
        # Strip ChatML tokens that leaked through
        text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
        # Strip Hunyuan thinking/answer wrapper tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        text = text.replace("<answer>", "").replace("</answer>", "").strip()

        # Strip GPT OSS reasoning dumps — model outputs chain-of-thought
        # then the actual answer. Detect by looking for reasoning markers.
        reasoning_markers = [
            "Thus we should respond:",
            "So the response is:",
            "So we should say:",
            "The reply is:",
            "Our response:",
            "My response:",
            "Final response:",
            "Reply:",
        ]
        for marker in reasoning_markers:
            if marker.lower() in text.lower():
                idx = text.lower().index(marker.lower())
                text = text[idx + len(marker):].strip()
                break
        else:
            # No explicit marker — check if it looks like reasoning
            # (starts with "We need to", "We have to", "The instructions:", etc.)
            reasoning_starts = [
                "we need to", "we have to", "we must", "we should",
                "the instructions", "the only fact", "let me", "let's",
                "i need to", "i should", "i must", "first,",
            ]
            if any(text.lower().startswith(rs) for rs in reasoning_starts):
                # Try to find the actual dialogue after quotes
                # Look for last quoted block or last paragraph
                quoted = re.findall(r'"([^"]{10,})"', text)
                if quoted:
                    text = quoted[-1]
                else:
                    # Take last paragraph as the likely answer
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    if len(paragraphs) > 1:
                        text = paragraphs[-1]

        # Strip prompt headers that leaked into response
        text = re.sub(r'^##\s+(WHO YOU ARE|FACTS|CONVERSATION|CONVERSATION HISTORY|YOUR RESPONSE).*?\n', '', text, flags=re.MULTILINE).strip()
        # Strip surrounding quotes if present
        text = text.strip('"').strip()
        # Strip trailing turn-boundary bleed
        text = re.sub(r'([.!?])\s*(?:User|user|assistant|Assistant|ANIMA|system)\s*:?\s*$', r'\1', text)
        return text.strip()

    def _server_chat(self, messages, max_tokens, temperature=None, timeout=180):
        """Call the llama.cpp server — non-streaming."""
        payload = self._server_payload(messages, max_tokens, stream=False, temperature=temperature)
        # Connect timeout 10s (detect dead servers fast), read timeout from caller
        r = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            timeout=(10, timeout),
        )
        r.raise_for_status()
        result = r.json()
        msg = result["choices"][0]["message"]
        content = msg.get("content", "") or ""
        # Some models put output in reasoning_content when content is empty
        if not content.strip() and msg.get("reasoning_content"):
            content = msg["reasoning_content"]
        return self._clean_response(content)

    def _server_chat_stream(self, messages, max_tokens):
        """Call the llama.cpp server — streaming."""
        payload = self._server_payload(messages, max_tokens, stream=True)
        r = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=180,
        )
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

    def generate(self, prompt, system_context="", max_tokens=None):
        model_cfg = self.config["model"]
        max_tokens = max_tokens or model_cfg["max_response_tokens"]
        messages = self._build_messages(prompt, system_context)

        if self.mode == "server":
            return self._server_chat(messages, max_tokens)
        else:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load() first.")
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=model_cfg["temperature"],
                top_p=model_cfg["top_p"],
            )
            return response["choices"][0]["message"]["content"]

    def generate_streaming(self, prompt, system_context="", max_tokens=None):
        model_cfg = self.config["model"]
        max_tokens = max_tokens or model_cfg["max_response_tokens"]
        messages = self._build_messages(prompt, system_context)

        if self.mode == "server":
            yield from self._server_chat_stream(messages, max_tokens)
        else:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load() first.")
            stream = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=model_cfg["temperature"],
                top_p=model_cfg["top_p"],
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content

    def generate_with_messages(self, messages, max_tokens=None, temperature=None, timeout=180):
        model_cfg = self.config["model"]
        max_tokens = max_tokens or model_cfg["max_response_tokens"]
        temp = temperature if temperature is not None else model_cfg["temperature"]

        if self.mode == "server":
            return self._server_chat(messages, max_tokens, temperature=temperature, timeout=timeout)
        else:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load() first.")
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=model_cfg["top_p"],
            )
            return response["choices"][0]["message"]["content"]

    def generate_with_messages_streaming(self, messages, max_tokens=None):
        model_cfg = self.config["model"]
        max_tokens = max_tokens or model_cfg["max_response_tokens"]

        if self.mode == "server":
            yield from self._server_chat_stream(messages, max_tokens)
        else:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load() first.")
            stream = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=model_cfg["temperature"],
                top_p=model_cfg["top_p"],
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content

    def unload(self):
        self.model = None
