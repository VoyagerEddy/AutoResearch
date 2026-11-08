import json
from dataclasses import dataclass

import pytest
from mcp.types import TextContent

from autoresearch.services.colab_mcp import (
    ColabMcpError, ColabMcpSidecarClient, _dump_model,
    atomic_private_json, validate_notebook_url,
)


def test_mcp_dataclass_result_preserves_nested_text_and_error_flag():
    # FastMCP returns a dataclass containing MCP Pydantic content blocks.
    @dataclass
    class Result:
        content: list
        structured_content: dict
        is_error: bool = False

    value = Result([TextContent(type='text', text='{"cells":[]}')], {'cells': []})
    saved = json.loads(json.dumps(_dump_model(value)))
    assert saved['content'][0] == {'type': 'text', 'text': '{"cells":[]}'}
    assert saved['structured_content'] == {'cells': []}
    assert saved['is_error'] is False


@pytest.mark.parametrize('url', [
    'https://example.com/notebooks/empty.ipynb',
    'http://colab.research.google.com/notebooks/empty.ipynb',
    'https://user:secret@colab.research.google.com/notebooks/empty.ipynb',
    'https://colab.research.google.com:8080/notebooks/empty.ipynb',
    'https://colab.research.google.com/notebooks/empty.ipynb#credential=secret',
])
def test_bridge_only_accepts_official_noncredential_notebook_url(url):
    with pytest.raises(ColabMcpError):
        validate_notebook_url(url)


@pytest.mark.asyncio
async def test_unapproved_bridge_tool_never_reaches_ipc(tmp_path):
    client = ColabMcpSidecarClient(tmp_path)
    with pytest.raises(ColabMcpError):
        await client.request('call_tool', name='create_paid_runtime', arguments={})
    assert not list(tmp_path.iterdir())


def test_bridge_public_status_contains_no_connection_secret(tmp_path):
    atomic_private_json(tmp_path / 'connection.private.json', {'url': 'private-token-bearing-url'})
    atomic_private_json(tmp_path / 'status.json', {'status': 'connected', 'connected': True, 'port': 1234})
    status = ColabMcpSidecarClient(tmp_path).status()
    assert status['connected'] is True
    assert 'private-token' not in json.dumps(status)
