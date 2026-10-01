# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import pytest
from typer.testing import CliRunner

from nemoguardrails.cli import app

runner = CliRunner()


def _models(*items):
    return "models:\n" + "".join(f"  - {i}\n" for i in items)


OPENAI = "{type: main, engine: openai, model: gpt-4o-mini}"
EMBED = "{type: embeddings, engine: FastEmbed, model: all-MiniLM-L6-v2}"
CUSTOM = "{type: main, engine: my_custom, model: my-model}"
BAD = "{type: main, engine: opnai, model: gpt-4o-mini}"


@pytest.mark.parametrize(
    "yaml_text, extra_files, target, exit_code, contains, not_contains",
    [
        pytest.param(_models(OPENAI), {}, "dir", 0, ["OK"], [], id="valid-dir"),
        pytest.param(_models(OPENAI), {}, "file", 0, ["OK"], [], id="valid-single-file"),
        pytest.param(_models(BAD), {}, "dir", 1, ["opnai", "openai"], [], id="unknown-engine"),
        pytest.param(_models(OPENAI, EMBED), {}, "dir", 0, ["OK"], [], id="embeddings-engine-skipped"),
        pytest.param(
            _models(CUSTOM), {"config.py": ""}, "dir", 0, ["Warning", "my_custom"], [], id="custom-provider-warns"
        ),
        pytest.param(_models(CUSTOM), {}, "dir", 1, ["my_custom"], [], id="unknown-no-config-py"),
        pytest.param(
            _models("{type: main, model: gpt-4o-mini}"),
            {},
            "dir",
            1,
            ["models.0.engine: Field required"],
            [],
            id="missing-engine",
        ),
        pytest.param(
            _models("{engine: openai, model: gpt-4o-mini}"),
            {},
            "dir",
            1,
            ["models.0.type: Field required"],
            [],
            id="missing-type",
        ),
        pytest.param(
            _models("{type: main, engine: openai}"),
            {},
            "dir",
            1,
            ["Model name must be specified"],
            [],
            id="missing-model-name",
        ),
        pytest.param("models: [", {}, "dir", 1, ["Error:"], ["Traceback"], id="malformed-yaml"),
        pytest.param("models: []\n", {}, "dir", 0, ["OK"], [], id="no-models"),
        pytest.param(_models(BAD, CUSTOM), {}, "dir", 1, ["opnai", "my_custom"], [], id="multiple-bad"),
    ],
)
def test_validate(tmp_path, yaml_text, extra_files, target, exit_code, contains, not_contains):
    (tmp_path / "config.yml").write_text(yaml_text)
    for name, text in extra_files.items():
        (tmp_path / name).write_text(text)
    path = tmp_path / "config.yml" if target == "file" else tmp_path

    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code == exit_code, result.output
    for text in contains:
        assert text in result.output
    for text in not_contains:
        assert text not in result.output


def test_validate_missing_path(tmp_path):
    result = runner.invoke(app, ["validate", str(tmp_path / "nope")])

    assert result.exit_code == 1, result.output
    assert "Invalid config path" in result.output
