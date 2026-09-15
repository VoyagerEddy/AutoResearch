const state = { projects: [], current: null, settings: null, connection: null, autodl: null, instanceCreating: false, creationUncertain: false, poller: null, connectionPoller: null, experimentId: null };
const phaseLabels = {
  queued: "Queued", planning: "Planning", searching: "Source Search", synthesizing: "Method Design",
  generating: "Code Generation", ready: "Code Saved", experiment: "Remote Experiment", analyzing: "Result Analysis",
  completed: "Completed", failed: "Failed", experiment_failed: "Experiment Failed",
  chatgpt_thinking: "ChatGPT Reasoning", chatgpt_analysis: "ChatGPT Analysis Complete"
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const controller = new AbortController();
  const { timeoutMs = 12000, ...requestOptions } = options;
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, {
      ...requestOptions,
      signal: requestOptions.signal || controller.signal,
      headers: { "Content-Type": "application/json", ...(requestOptions.headers || {}) }
    });
    let data = null;
    try { data = await response.json(); } catch (_) { data = {}; }
    if (!response.ok) {
      const detail = data.detail;
      const error = new Error(typeof detail === "string" ? detail : (detail?.message || `Request failed (${response.status})`));
      error.code = detail?.code;
      throw error;
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError" || error instanceof TypeError) {
      const connectionError = new Error(error.name === "AbortError" ? "The local service timed out. Make sure the startup window is still running." : "Could not connect to the local service. Run start.cmd again.");
      connectionError.code = "connection_uncertain";
      throw connectionError;
    }
    throw error;
  } finally { clearTimeout(timeout); }
}

function toast(message, error = false) {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(node._timer);
  node._timer = setTimeout(() => node.className = "toast", 3500);
}

function formatTime(value) {
  try { return new Date(value).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }); }
  catch (_) { return ""; }
}

async function bootstrap() {
  try {
    const health = await api("/api/health");
    $("healthDot").classList.add("ok");
    $("healthText").textContent = "Local service and ChatGPT tools are ready";
    $("mcpEndpoint").textContent = health.mcp_endpoint;
    [state.settings, state.connection] = await Promise.all([
      api("/api/settings"), api("/api/chatgpt/connection")
    ]);
    applySettings();
    renderConnection();
    await refreshAutoDL();
    clearInterval(state.connectionPoller);
    state.connectionPoller = setInterval(refreshConnection, 5000);
    await loadProjects();
    const requested = new URLSearchParams(location.search).get("project");
    if (requested && state.projects.some((project) => project.id === requested)) await selectProject(requested);
  } catch (error) {
    $("healthText").textContent = "Local service unavailable";
    toast(error.message, true);
  }
}

function renderConnection() {
  const c = state.connection || {};
  const setStep = (id, ready, readyText, waitingText) => {
    const node = $(id); if (!node) return;
    node.classList.toggle("ready", Boolean(ready));
    node.textContent = `${ready ? "✓" : "○"} ${ready ? readyText : waitingText}`;
  };
  setStep("tunnelInstallStatus", c.tunnel_client_installed, "tunnel-client installed", "tunnel-client not installed");
  setStep("tunnelConfigStatus", c.tunnel_configured, `Tunnel configured ${c.tunnel_id_hint || ""}`, "Tunnel not configured");
  setStep("tunnelRunStatus", c.tunnel_ready, "Tunnel ready", c.tunnel_process_running ? "Tunnel process running; connection not ready" : "Tunnel not running");
  $("startChatgpt").disabled = !c.tunnel_configured;
  $("settingsStartChatgpt").disabled = !c.tunnel_configured;
  $("setupChatgpt").textContent = c.tunnel_configured ? "Reconfigure Connection" : "Set Up Connection";
  $("settingsTunnelStatus").textContent = c.tunnel_ready
    ? "Tunnel connected and available in ChatGPT"
    : (c.tunnel_process_running ? "Process started; waiting for the Tunnel connection" : (c.tunnel_configured ? "Configuration saved; Tunnel is not running" : "Initial setup is incomplete"));
}

async function refreshAutoDL(verifyApi = false) {
  const button = $("checkAutoDL");
  button.disabled = true;
  try {
    state.autodl = await api(`/api/autodl/preflight?verify_api=${verifyApi}`, { timeoutMs: 45000 });
    const result = state.autodl;
    const headline = result.ready
      ? (result.verified ? "AutoDL API connected; balance and GPU inventory are not yet verified." : "Required AutoDL settings are saved; API, balance, and GPU inventory are not yet verified.")
      : `Cannot create an instance yet: ${result.blocking_issues.map((issue) => issue.message).join("; ")}`;
    const messages = [headline, ...result.warnings.map((warning) => warning.message)].join("\n");
    for (const id of ["autodlReadiness", "settingsAutoDLStatus"]) {
      $(id).textContent = messages;
      $(id).className = `form-status${result.ready ? "" : " error"}`;
    }
    $("createInstance").disabled = !result.ready || state.instanceCreating || state.creationUncertain;
    return result;
  } catch (error) {
    state.autodl = null;
    $("createInstance").disabled = true;
    for (const id of ["autodlReadiness", "settingsAutoDLStatus"]) {
      $(id).textContent = `Could not check AutoDL: ${error.message}`;
      $(id).className = "form-status error";
    }
    return null;
  } finally { button.disabled = false; }
}

async function refreshConnection() {
  try { state.connection = await api("/api/chatgpt/connection"); renderConnection(); }
  catch (_) {}
}

async function launchChatgptConnection(action) {
  try {
    await api("/api/chatgpt/connection/launch", {
      method: "POST", body: JSON.stringify({ action, confirm_launch: true })
    });
    toast(action === "Setup" ? "Connection setup wizard opened" : "ChatGPT connection window opened");
    setTimeout(refreshConnection, 1500);
  } catch (error) { toast(error.message, true); }
}

function applySettings() {
  const s = state.settings || {};
  $("gpuOrder").textContent = (s.autodl_gpu_specs || ["v-48g", "5090-p"]).join(" → ");
  $("monitorPolicy").textContent = `Every ${s.monitor_seconds || 30}s · Up to ${s.max_iterations || 3} rounds`;
  $("settingModel").value = s.openrouter_model || "openrouter/free";
  $("settingImageUuid").value = s.autodl_image_uuid || "";
  $("settingGpuSpecs").value = (s.autodl_gpu_specs || []).join(",");
  $("settingRemote").value = s.github_remote_url || "";
  updateWorkflowMode();
}

function updateWorkflowMode() {
  const collaborative = $("workflowMode").value === "chatgpt";
  $("researchForm").classList.toggle("collaborative", collaborative);
  document.querySelectorAll(".autonomous-option").forEach((node) => node.classList.toggle("hidden", collaborative));
  $("modelHint").textContent = collaborative
    ? "ChatGPT handles reasoning; AutoResearch waits for tool calls"
    : (state.settings?.openrouter_configured ? `OpenRouter · ${state.settings.openrouter_model}` : "An offline baseline will be generated without an API key");
  $("startResearch").innerHTML = collaborative ? "Create Collaborative Project <b>→</b>" : "Start Autonomous Research <b>→</b>";
}

async function loadProjects() {
  state.projects = await api("/api/projects");
  renderProjectList();
}

function renderProjectList() {
  const list = $("projectList");
  list.replaceChildren();
  for (const project of state.projects) {
    const button = document.createElement("button");
    button.className = `project-item${state.current?.id === project.id ? " active" : ""}`;
    const title = document.createElement("strong"); title.textContent = project.topic;
    const meta = document.createElement("span"); meta.textContent = `${phaseLabels[project.phase] || project.phase} · ${project.progress}%`;
    button.append(title, meta);
    button.onclick = () => selectProject(project.id);
    list.append(button);
  }
}

async function selectProject(id) {
  state.current = await api(`/api/projects/${id}`);
  state.experimentId = null;
  $("createView").classList.add("hidden");
  $("projectView").classList.remove("hidden");
  $("pageTitle").textContent = "Research Workspace";
  $("openCode").disabled = false;
  $("syncGit").disabled = state.current.status !== "ready";
  history.replaceState(null, "", `?project=${encodeURIComponent(id)}`);
  renderProject();
  renderProjectList();
  await Promise.all([loadEvents(), loadSources(), loadManifest(), loadExperiments(), loadExperimentReadiness()]);
  beginPolling();
}

function renderProject() {
  const p = state.current;
  if (!p) return;
  $("projectTopic").textContent = p.topic;
  $("projectId").textContent = p.id;
  $("projectSummary").textContent = p.error || p.summary || "The research agent is working and continuously saving artifacts to the local workspace.";
  $("statusBadge").textContent = phaseLabels[p.phase] || p.phase;
  $("statusBadge").className = `status-badge${p.status === "failed" ? " failed" : ""}`;
  $("progressText").textContent = `${p.progress}%`;
  $("progressRing").style.setProperty("--progress", `${p.progress * 3.6}deg`);
  $("syncGit").disabled = p.status !== "ready";
  const stageByPhase = {
    queued: 0, planning: 0, chatgpt_thinking: 0, searching: 1, synthesizing: 2,
    generating: 3, ready: 4, experiment: 4, analyzing: 4, completed: 4,
    chatgpt_analysis: 4, experiment_failed: 4
  };
  const currentIndex = stageByPhase[p.phase] ?? 0;
  document.querySelectorAll("#pipeline > div").forEach((node, index) => {
    node.classList.toggle("done", index < currentIndex || ["completed", "chatgpt_analysis"].includes(p.phase));
    node.classList.toggle("active", index === currentIndex);
  });
}

async function loadEvents() {
  if (!state.current) return;
  const events = await api(`/api/projects/${state.current.id}/events`);
  const list = $("eventList"); list.replaceChildren();
  if (!events.length) { list.innerHTML = '<div class="empty">Waiting for the agent to start…</div>'; return; }
  for (const event of events.slice().reverse()) {
    const row = document.createElement("div"); row.className = `event ${event.level}`;
    const dot = document.createElement("i");
    const body = document.createElement("div");
    const text = document.createElement("strong"); text.textContent = event.message;
    const meta = document.createElement("span"); meta.textContent = `${phaseLabels[event.phase] || event.phase} · ${formatTime(event.created_at)}`;
    body.append(text, meta); row.append(dot, body); list.append(row);
  }
}

async function loadSources() {
  if (!state.current) return;
  const sources = await api(`/api/projects/${state.current.id}/sources`);
  $("sourceCount").textContent = sources.length;
  const list = $("sourceList"); list.replaceChildren();
  if (!sources.length) { list.innerHTML = '<div class="empty">Retrieved sources will appear here</div>'; return; }
  for (const source of sources) {
    const a = document.createElement("a"); a.className = "source";
    if (/^https:\/\//.test(source.url)) { a.href = source.url; a.target = "_blank"; a.rel = "noreferrer"; }
    const meta = document.createElement("div"); meta.className = "meta";
    meta.textContent = `${source.provider} · ${source.kind === "code" ? "CODE" : (source.year || "PAPER")} · ${source.citation_count || 0} citations/stars`;
    const title = document.createElement("h4"); title.textContent = source.title;
    const desc = document.createElement("p"); desc.textContent = source.abstract || source.url;
    a.append(meta, title, desc); list.append(a);
  }
}

async function loadManifest() {
  if (!state.current) return;
  const manifest = await api(`/api/projects/${state.current.id}/manifest`);
  $("remoteCommand").value = manifest.run_command || "";
}

async function loadExperimentReadiness() {
  if (!state.current) return;
  const projectId = state.current.id;
  $("checkExperimentReadiness").disabled = true;
  try {
    const report = await api(`/api/projects/${projectId}/readiness`);
    if (state.current?.id !== projectId) return;
    const details = report.blocking_issues.map((item) => `${item.path || item.code}: ${item.message}`);
    $("experimentReadiness").textContent = report.ready
      ? "The experiment package passed static checks. Dependencies, GPU behavior, data content, and scientific results still require execution."
      : `Experiment readiness blocked by ${details.length} item(s):\n${details.join("\n")}`;
    $("experimentReadiness").className = `form-status${report.ready ? "" : " error"}`;
  } catch (error) {
    if (state.current?.id === projectId) {
      $("experimentReadiness").textContent = `Experiment check unavailable: ${error.message}`;
      $("experimentReadiness").className = "form-status error";
    }
  } finally { $("checkExperimentReadiness").disabled = false; }
}

$("checkExperimentReadiness").onclick = loadExperimentReadiness;

async function loadExperiments() {
  if (!state.current) return;
  const experiments = await api(`/api/projects/${state.current.id}/experiments`);
  const list = $("experimentResults"); list.replaceChildren();
  if (!experiments.length) {
    state.experimentId = null;
    list.innerHTML = '<div class="empty compact">Experiment status, metrics, and logs will appear here</div>';
    $("experimentStatus").textContent = "Not started";
    return;
  }
  const active = experiments.find((item) => !["completed", "failed"].includes(item.status));
  state.experimentId = active ? active.id : null;
  const newest = experiments[0];
  const statusLabels = { preparing: "Preparing", uploading: "Uploading", running: "Running", analyzing: "Awaiting analysis", completed: "Completed", failed: "Failed" };
  $("experimentStatus").textContent = `${statusLabels[newest.status] || newest.status} · ${newest.id}`;
  for (const experiment of experiments) {
    const card = document.createElement("article"); card.className = `experiment-result ${experiment.status}`;
    const head = document.createElement("div"); head.className = "result-head";
    const title = document.createElement("strong"); title.textContent = `Experiment ${experiment.id}`;
    const badge = document.createElement("span"); badge.textContent = statusLabels[experiment.status] || experiment.status;
    head.append(title, badge);
    const meta = document.createElement("p"); meta.textContent = `Round ${experiment.iteration} · ${experiment.command}`;
    card.append(head, meta);
    const result = experiment.result || {};
    if (Object.keys(result.metrics || {}).length) {
      const metrics = document.createElement("pre"); metrics.textContent = JSON.stringify(result.metrics, null, 2);
      card.append(metrics);
    }
    if (experiment.error) { const error = document.createElement("p"); error.className = "result-error"; error.textContent = experiment.error; card.append(error); }
    if (result.log_tail) {
      const details = document.createElement("details");
      const summary = document.createElement("summary"); summary.textContent = "View log tail";
      const log = document.createElement("pre"); log.textContent = String(result.log_tail).slice(-5000);
      details.append(summary, log); card.append(details);
    }
    if (["completed", "failed"].includes(experiment.status)) {
      const analyze = document.createElement("button"); analyze.type = "button"; analyze.className = "ghost analyze-result";
      analyze.textContent = "Copy ChatGPT Analysis Request";
      analyze.onclick = () => copyText(`Call AutoResearch get_experiment_result, analyze experiment ${experiment.id}, then use record_chatgpt_analysis to save the conclusions to project ${experiment.project_id}.`, "Analysis request copied");
      card.append(analyze);
    }
    list.append(card);
  }
}

function beginPolling() {
  clearInterval(state.poller);
  state.poller = setInterval(async () => {
    if (!state.current) return;
    try {
      state.current = await api(`/api/projects/${state.current.id}`);
      renderProject();
      await Promise.all([loadEvents(), loadSources(), loadExperiments()]);
      await loadProjects();
      if (state.experimentId) await refreshExperiment();
    } catch (_) {}
  }, 2500);
}

$("researchForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter; button.disabled = true; button.textContent = "Creating…";
  try {
    const collaborative = $("workflowMode").value === "chatgpt";
    const result = collaborative
      ? await api("/api/chatgpt/projects", { method: "POST", body: JSON.stringify({ topic: $("topic").value, notes: $("notes").value }) })
      : await api("/api/research", { method: "POST", body: JSON.stringify({
          topic: $("topic").value, notes: $("notes").value,
          max_sources: Number($("maxSources").value), download_papers: $("downloadPapers").checked
        })});
    const project = collaborative ? result.project : result;
    await loadProjects(); await selectProject(project.id);
    if (collaborative) toast(`Collaborative project ${project.id} created. Use this project ID in ChatGPT.`);
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; updateWorkflowMode(); }
});

$("newProject").onclick = () => {
  state.current = null; clearInterval(state.poller);
  $("projectView").classList.add("hidden"); $("createView").classList.remove("hidden");
  history.replaceState(null, "", location.pathname);
  $("pageTitle").textContent = "Turn an Idea into a Reproducible Experiment";
  $("openCode").disabled = true; $("syncGit").disabled = true; renderProjectList();
};

function copyText(value, message) {
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(value).then(() => toast(message)).catch(() => toast("Copy failed. Select the text manually.", true));
  } else { toast("This browser does not support automatic copying. Select the text manually.", true); }
}

$("workflowMode").addEventListener("change", updateWorkflowMode);
$("copyMcp").onclick = () => copyText($("mcpEndpoint").textContent, "MCP address copied");
$("copyProjectId").onclick = () => copyText($("projectId").textContent, "Project ID copied");
$("setupChatgpt").onclick = () => launchChatgptConnection("Setup");
$("startChatgpt").onclick = () => launchChatgptConnection("Run");
$("settingsSetupChatgpt").onclick = () => launchChatgptConnection("Setup");
$("settingsStartChatgpt").onclick = () => launchChatgptConnection("Run");

$("openCode").onclick = async () => {
  try { await api(`/api/projects/${state.current.id}/open`, { method: "POST" }); toast("Sent to VS Code"); }
  catch (error) { toast(error.message, true); }
};

$("syncGit").onclick = async () => {
  if (!state.settings?.github_remote_url) { $("settingsDialog").showModal(); return toast("Configure a GitHub repository first", true); }
  try {
    const result = await api("/api/git/sync", { method: "POST", body: JSON.stringify({ project_id: state.current.id }) });
    toast(result.pushed ? "Experiment code pushed to GitHub" : "Code committed to local Git");
  } catch (error) { toast(error.message, true); }
};

$("settingsButton").onclick = () => {
  $("settingsStatus").textContent = "";
  $("settingsStatus").className = "form-status";
  $("settingsDialog").showModal();
};
document.querySelectorAll(".close-settings").forEach((button) => {
  button.addEventListener("click", () => $("settingsDialog").close());
});
$("settingsForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const saveButton = $("saveSettings");
  const status = $("settingsStatus");
  saveButton.disabled = true;
  saveButton.textContent = "Saving…";
  status.className = "form-status";
  status.textContent = "Writing local configuration…";
  const payload = {
    openrouter_model: $("settingModel").value,
    autodl_image_uuid: $("settingImageUuid").value,
    autodl_gpu_specs: $("settingGpuSpecs").value,
    github_remote_url: $("settingRemote").value
  };
  if ($("settingOpenRouterKey").value) payload.openrouter_api_key = $("settingOpenRouterKey").value;
  if ($("settingAutoDLToken").value) payload.autodl_token = $("settingAutoDLToken").value;
  try {
    state.settings = await api("/api/settings", { method: "PATCH", body: JSON.stringify(payload) });
    $("settingOpenRouterKey").value = ""; $("settingAutoDLToken").value = "";
    applySettings();
    await refreshAutoDL();
    status.textContent = "Settings saved and applied immediately.";
    toast("Settings saved locally");
    setTimeout(() => { if ($("settingsDialog").open) $("settingsDialog").close(); }, 450);
  } catch (error) {
    status.className = "form-status error";
    status.textContent = error.message;
    toast(error.message, true);
  } finally {
    saveButton.disabled = false;
    saveButton.textContent = "Save Settings";
  }
});

$("toggleExperiment").onclick = () => $("experimentForm").classList.toggle("hidden");

$("checkAutoDL").onclick = () => refreshAutoDL(true);

async function launchAutoDLLogin(action) {
  const button = action === "Setup" ? $("setupAutoDLLogin") : $("openAutoDLLogin");
  button.disabled = true;
  try {
    await api("/api/autodl/browser-login/launch", {
      method: "POST",
      body: JSON.stringify({
        action, browser: "auto", open_page: "console", keep_open: true, confirm_launch: true
      })
    });
    toast(action === "Setup"
      ? "AutoDL encrypted login setup opened in a private console."
      : "AutoDL automatic login opened. Complete verification in the new windows.");
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; }
}

$("setupAutoDLLogin").onclick = () => launchAutoDLLogin("Setup");
$("openAutoDLLogin").onclick = () => launchAutoDLLogin("Login");

$("createInstance").onclick = async () => {
  if (state.instanceCreating || state.creationUncertain) return;
  state.instanceCreating = true;
  $("createInstance").disabled = true;
  let creationRequested = false;
  try {
    const readiness = await refreshAutoDL();
    if (!readiness?.ready) return;
    if (!confirm("AutoDL Pro instances are billed after startup. Create the preferred GPU instance?")) return;
    creationRequested = true;
    const result = await api("/api/autodl/instances", { method: "POST", body: JSON.stringify({ confirm_billable: true }), timeoutMs: 120000 });
    if (typeof result.instance_uuid !== "string" || !result.instance_uuid) {
      const error = new Error("The creation response has no instance UUID. Check the instance list first.");
      error.code = "autodl_create_uncertain";
      throw error;
    }
    $("instanceUuid").value = result.instance_uuid;
    toast(`Instance ${result.instance_uuid} is being created (${result.gpu_spec})`);
  } catch (error) {
    if (creationRequested && ["autodl_create_uncertain", "connection_uncertain"].includes(error.code)) {
      state.creationUncertain = true;
      $("createInstance").textContent = "Creation result unknown; check instances";
      $("autodlReadiness").textContent = "The creation result is unknown and retries have stopped. Check the AutoDL console and confirm that no new instance exists before reloading this page.";
    }
    toast(error.message, true);
  } finally {
    state.instanceCreating = false;
    $("createInstance").disabled = !state.autodl?.ready || state.creationUncertain;
  }
};

$("loadSsh").onclick = async () => {
  const id = $("instanceUuid").value.trim(); if (!id) return toast("Enter an instance UUID first", true);
  try {
    const result = await api(`/api/autodl/instances/${encodeURIComponent(id)}`);
    if (!result.ssh) return toast(`Instance status: ${result.status}. SSH information is not available yet.`, true);
    $("sshHost").value = result.ssh.host || ""; $("sshPort").value = result.ssh.port || 22;
    $("sshUser").value = result.ssh.username || "root"; $("sshPassword").value = result.ssh.password || "";
    toast("SSH information loaded");
  } catch (error) { toast(error.message, true); }
};

$("experimentForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!confirm("The generated code will be uploaded to the remote host and the displayed command will run. Continue?")) return;
  const payload = {
    project_id: state.current.id,
    instance_uuid: $("instanceUuid").value || null,
    connection: { host: $("sshHost").value, port: Number($("sshPort").value), username: $("sshUser").value, password: $("sshPassword").value || null },
    command: $("remoteCommand").value,
    max_iterations: Number($("maxIterations").value),
    allow_release_replacement: $("allowRelease").checked,
    recover_busy_gpu: true,
    confirm_execute: true
  };
  try {
    const result = await api("/api/experiments", { method: "POST", body: JSON.stringify(payload) });
    state.experimentId = result.experiment_id; $("experimentStatus").textContent = "Preparing"; beginPolling(); toast("Experiment started");
  } catch (error) { toast(error.message, true); }
});

async function refreshExperiment() {
  const result = await api(`/api/experiments/${state.experimentId}`);
  const labels = { preparing: "Preparing", uploading: "Uploading", running: `Running · Round ${result.iteration}`, analyzing: "Analyzing", completed: "Completed", failed: "Failed" };
  $("experimentStatus").textContent = labels[result.status] || result.status;
  if (["completed", "failed"].includes(result.status)) {
    if (result.status === "failed" && result.error) toast(result.error, true);
    state.experimentId = null;
    await loadExperiments();
  }
}

bootstrap();
