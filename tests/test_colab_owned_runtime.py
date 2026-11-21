import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from autoresearch.services.colab_browser import (
    STATUS_PREFIX,
    ColabBrowserOptions,
    ColabBrowserSession,
    ColabBrowserWorker,
    OwnedRuntimeLaunch,
)


class Control:
    def __init__(self, role, name, children=(), onclick=None):
        self.role = role
        self.name = name
        self.children = list(children)
        self.onclick = onclick

    def get_by_role(self, role, name=None):
        async def all_controls():
            return [child for child in self.children
                    if child.role == role and (name is None or name.fullmatch(child.name))]

        return SimpleNamespace(all=all_controls)

    async def is_visible(self):
        return True

    async def is_disabled(self):
        return False

    async def click(self, *, timeout):
        self.onclick()


class Page(Control):
    def __init__(self, url):
        super().__init__("page", "")
        self.url = url
        self.main_frame = object()
        self.listeners = {}

    def is_closed(self):
        return False

    def on(self, event, handler):
        self.listeners[event] = handler

    def remove_listener(self, event, handler):
        assert self.listeners.pop(event) == handler


@pytest.fixture
def runtime(tmp_path):
    digest = "a" * 64
    project = "project-1"
    notebook = tmp_path / f"AutoResearch_{project}.ipynb"
    url = "https://colab.research.google.com/drive/example-project"
    job = {"bundle_sha256": digest, "notebook_bundle_sha256": digest, "project_id": project,
           "notebook_url": url, "notebook_path": str(notebook), "prepared": True, "source_current": True}
    page = Page(url)
    launch = OwnedRuntimeLaunch(page, project, url, digest, datetime.now(timezone.utc))
    identity = {"project_id": project, "runtime_id": str(uuid4()), "started_at": launch.action_started_at.isoformat()}
    markers = [
        {**identity, "status": "starting"},
        {**identity, "status": "available", "gpu": "Tesla T4, 15360 MiB"},
        {**identity, "status": "completed", "returncode": 0, "runtime_release_safe": True},
    ]
    cells = [
        {"id": "stamp", "cell_type": "markdown", "source": "AUTORESEARCH_SOURCE_SHA256 " + digest},
        {"id": "code", "cell_type": "code", "source": "pass\n", "outputs": [{
            "output_type": "stream", "name": "stdout",
            "text": "\n".join(STATUS_PREFIX + json.dumps(marker) for marker in markers),
        }]},
    ]
    notebook.write_text(json.dumps({"nbformat": 4, "cells": cells}), encoding="utf-8")
    return SimpleNamespace(job=job, page=page, launch=launch, markers=markers, cells=cells, path=tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["released", "wrong_dialog", "ambiguous", "other_source", "replaced_runtime",
                                      "revoked", "revoked_after_menu", "navigated"])
async def test_release_checks_the_notebook_and_scopes_confirmation(runtime, scenario):
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=runtime.path))
    session.page = runtime.page
    actions = []

    async def request(*args, **kwargs):
        return {"ok": True, "result": {"structuredContent": {"cells": runtime.cells}}}

    async def verify_registration(response):
        pass

    session._notebook_mcp = SimpleNamespace(status=lambda: {"connected": True}, request=request)
    session.ensure_mcp_registered_notebook = verify_registration
    if scenario == "other_source":
        runtime.cells[-1]["source"] = "print('another source')\n"

    def confirm():
        actions.append("confirm")
        session.page.children = [Control("button", "Connect T4 Connect to a new runtime")]

    def delete():
        actions.append("delete")
        name = "Connect to Google Drive" if scenario == "wrong_dialog" else "Delete runtime?"
        buttons = [Control("button", "Yes", onclick=confirm)]
        if scenario == "ambiguous":
            buttons.append(Control("button", "OK", onclick=confirm))
        session.page.children = [Control("dialog", name, buttons)]
        if scenario == "navigated":
            runtime.launch.on_navigation(runtime.page.main_frame)
        if scenario == "replaced_runtime":
            replaced = {**runtime.markers[0], "runtime_id": str(uuid4())}
            runtime.cells[-1]["outputs"][0]["text"] += "\n" + STATUS_PREFIX + json.dumps(replaced)

    def menu():
        actions.append("menu")
        session.page.children = [Control("menu", "", [Control("menuitem", "Disconnect and delete runtime", onclick=delete)])]

    session.page.children = [Control("button", "Runtime", onclick=menu)]

    async def may_release():
        return scenario != "revoked" and not (scenario == "revoked_after_menu" and "delete" in actions)

    result = await session.release_owned_runtime(runtime.launch, runtime.job, runtime.markers[-1], may_release, timeout_seconds=0)
    if scenario == "released":
        assert result == "released"
        assert actions == ["menu", "delete", "confirm"]
    else:
        assert result == "unknown"
        assert "confirm" not in actions
        if scenario == "revoked":
            assert "delete" not in actions


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnected", [True, False])
async def test_only_an_unattached_bound_tab_can_arm_release(runtime, disconnected):
    session = ColabBrowserSession(ColabBrowserOptions(profile_dir=runtime.path))
    session.page = runtime.page
    session._notebook_mcp = SimpleNamespace(status=lambda: {"connected": True})
    if disconnected:
        runtime.page.children.append(Control("button", "Connect T4 Connect to a new runtime"))
    launch = await session.arm_runtime_release(runtime.job)
    if disconnected:
        assert launch and launch.matches_page(runtime.page)
        runtime.page.listeners["framenavigated"](object())
        assert launch.matches_page(runtime.page)
        runtime.page.listeners["framenavigated"](runtime.page.main_frame)
        assert not launch.matches_page(runtime.page)
    else:
        assert launch is None


class ReleaseClient:
    def __init__(self, runtime, events, grant):
        self.project_id = runtime.job["project_id"]
        self.job = dict(runtime.job)
        self.events = events
        self.grant = grant

    async def status(self):
        return self.job

    async def observe(self, status, **details):
        self.events.append((status, details.get("runtime_release_requested")))
        if self.grant and details.get("runtime_release_requested"):
            self.job["runtime_release"] = {"status": "pending", "runtime_id": details["runtime_id"],
                                           "bundle_sha256": details["bundle_sha256"]}
        return self.job

    async def runtime_release(self, **receipt):
        self.events.append(("receipt", receipt["status"]))


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["owned", "manually_started", "no_safe_marker", "not_persisted", "wrong_page"])
async def test_worker_releases_only_after_acceptance_of_its_gpu_run(runtime, scenario):
    events = []
    client = ReleaseClient(runtime, events, grant=scenario != "not_persisted")

    async def release(launch, job, terminal, may_release):
        assert await may_release()
        events.append(("remote", "released"))
        return "released"

    page = runtime.page if scenario != "wrong_page" else Page(runtime.page.url)
    worker = ColabBrowserWorker(client, SimpleNamespace(page=page, release_owned_runtime=release))
    if scenario != "manually_started":
        runtime.page.on("framenavigated", runtime.launch.on_navigation)
        worker._owned_launch = runtime.launch
    marker = runtime.markers[-1]
    if scenario == "no_safe_marker":
        marker.pop("runtime_release_safe")
    worker.runtime_id = marker["runtime_id"]
    await worker._record_runtime_terminal("completed", "Complete", runtime.job, marker, runtime.markers,
                                         {key: value for key, value in marker.items() if key != "status"})
    if scenario == "owned":
        assert events == [("completed", True), ("remote", "released"), ("receipt", "released")]
    else:
        assert events == [("completed", True if scenario == "not_persisted" else None)]
    assert not runtime.page.listeners
