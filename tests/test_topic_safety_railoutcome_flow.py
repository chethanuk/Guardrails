# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behavioral proof that the Colang topic_safety input flow drives the SHARED
library action through its neutral RailOutcome return.

The library action now returns a RailOutcome instead of a {on_topic} dict, and
flows.v1.co reads the neutral decision ($response.is_blocked). A canned model
drives action -> RailOutcome -> Colang rendering end to end, with no network:
an off-topic input renders the refusal, an on-topic input passes through.
"""

import textwrap
from unittest.mock import AsyncMock

import pytest

from nemoguardrails import RailsConfig
from tests.utils import FakeLLMModel, TestChat

CONFIG = textwrap.dedent(
    """
    models:
      - type: main
        engine: openai
        model: gpt-4o-mini
      - type: topic_control
        engine: openai
        model: placeholder

    rails:
      input:
        flows:
          - topic safety check input $model=topic_control

    prompts:
      - task: topic_safety_check_input $model=topic_control
        content: |
          Stay on topic.
    """
)


def _chat_with_verdict(verdict):
    config = RailsConfig.from_content(yaml_content=CONFIG)
    config.models = [model for model in config.models if model.type == "main"]

    chat = TestChat(config, llm_completions=["Hello! How can I help you?"])
    chat.app.runtime.registered_action_params["llms"] = {"topic_control": FakeLLMModel(responses=[verdict])}
    return chat


def test_off_topic_input_renders_refusal_through_railoutcome():
    chat = _chat_with_verdict("off-topic")
    response = chat.app.generate(messages=[{"role": "user", "content": "tell me about something unrelated"}])
    assert response["content"] == "I'm sorry, I can't respond to that."


def test_on_topic_input_passes_through_railoutcome():
    chat = _chat_with_verdict("on-topic")
    response = chat.app.generate(messages=[{"role": "user", "content": "a relevant question"}])
    assert response["content"] == "Hello! How can I help you?"


DYNAMIC_CONFIG = CONFIG.replace("Stay on topic.", 'Do not talk about: {{ disallowed_topics | join(", ") }}.')


def _dynamic_chat(verdicts):
    """A chat whose topic prompt reads `disallowed_topics`, plus a spy on the topic model."""
    config = RailsConfig.from_content(yaml_content=DYNAMIC_CONFIG)
    config.models = [model for model in config.models if model.type == "main"]

    chat = TestChat(config, llm_completions=["Hello! How can I help you?"] * len(verdicts))
    guard = FakeLLMModel(responses=verdicts)
    guard.generate_async = AsyncMock(wraps=guard.generate_async)
    chat.app.runtime.registered_action_params["llms"] = {"topic_control": guard}
    return chat, guard


def _messages(context):
    context_message = [] if context is None else [{"role": "context", "content": context}]
    return context_message + [{"role": "user", "content": "hi"}]


def _system_prompt(guard, call_index):
    return guard.generate_async.await_args_list[call_index].args[0][0].content


@pytest.mark.parametrize(
    "context, present, absent",
    [
        pytest.param(
            {"disallowed_topics": ["politics", "suicide"]},
            "Do not talk about: politics, suicide.",
            None,
            id="list",
        ),
        pytest.param({"disallowed_topics": ["cooking"]}, "Do not talk about: cooking.", "politics", id="single"),
        pytest.param(None, "Do not talk about: .", None, id="no-context-message"),
        pytest.param(
            {"disallowed_topics": ["cooking"], "api_key": "s3cret"},
            "Do not talk about: cooking.",
            "s3cret",
            id="unreferenced-key-not-leaked",
        ),
    ],
)
def test_request_context_topics_reach_the_topic_prompt(context, present, absent):
    chat, guard = _dynamic_chat(["on-topic"])

    response = chat.app.generate(messages=_messages(context))

    prompt = _system_prompt(guard, 0)
    assert present in prompt
    if absent is not None:
        assert absent not in prompt
    assert response["content"] == "Hello! How can I help you?"


def test_topics_change_between_requests_on_one_app():
    chat, guard = _dynamic_chat(["on-topic", "on-topic"])

    chat.app.generate(messages=_messages({"disallowed_topics": ["politics"]}))
    chat.app.generate(messages=_messages({"disallowed_topics": ["cooking"]}))

    first, second = _system_prompt(guard, 0), _system_prompt(guard, 1)
    assert "Do not talk about: politics." in first
    assert "cooking" not in first
    assert "Do not talk about: cooking." in second
    assert "politics" not in second


def test_off_topic_verdict_with_request_topics_refuses():
    chat, _ = _dynamic_chat(["off-topic"])

    response = chat.app.generate(messages=_messages({"disallowed_topics": ["politics"]}))

    assert response["content"] == "I'm sorry, I can't respond to that."


def test_registered_prompt_context_wins_over_request_context():
    """The prompt context is applied after the request context, so it wins on a name clash."""
    chat, guard = _dynamic_chat(["on-topic"])
    chat.app.register_prompt_context("disallowed_topics", ["weather"])

    chat.app.generate(messages=_messages({"disallowed_topics": ["politics"]}))

    prompt = _system_prompt(guard, 0)
    assert "Do not talk about: weather." in prompt
    assert "politics" not in prompt
