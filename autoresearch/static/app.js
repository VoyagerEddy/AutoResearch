const state = {
  projects: [], current: null, selectedId: null, colabFirst: false,
  connection: null, colab: null, colabResourceAgent: null,
  colabResourceAgentUnavailable: false, experimentResults: null,
  refreshing: false, lastConfigurationCheck: 0,
};
const phaseLabels = {
  queued: "Queued", planning: "Planning", searching: "Source Search", synthesizing: "Method Design",
  generating: "Code Generation", ready: "Code Saved", experiment: "Remote Experiment", analyzing: "Result Analysis",
  completed: "Completed", failed: "Failed", experiment_failed: "Experiment Failed",
  chatgpt_thinking: "ChatGPT Reasoning", chatgpt_analysis: "ChatGPT Analysis Complete"
};

const $ = (id) => document.getElementById(id);
const projectPath = (id) => `/api/projects/${encodeURIComponent(id)}`;

async function api(path) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(path, { signal: controller.signal, cache: "no-store" });
    const data = await response.json();
    if (!response.ok) {
      const detail = data?.detail;
      const error = new Error(typeof detail === "string" ? detail : (detail?.message || `Request failed (${response.status})`));
      error.status = response.status;
      throw error;
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The local service timed out");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function formatTime(value) {
  const time = Date.parse(value || "");
  return Number.isFinite(time)
    ? new Date(time).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : "";
}

function showEmpty(title, message) {
  $("emptyTitle").textContent = title;
  $("emptyMessage").textContent = message;
  $("emptyView").classList.remove("hidden");
  $("projectView").classList.add("hidden");
}

async function bootstrap() {
  const query = new URLSearchParams(location.search);
  state.selectedId = query.get("project");
  state.colabFirst = query.get("compute") === "colab";
  if (!state.colabFirst) $("computePanels").prepend($("autodlPanel"));
  await refreshDashboard();
  setInterval(refreshDashboard, 5000);
}

async function refreshDashboard() {
  if (state.refreshing) return;
  state.refreshing = true;
  try {
    const refreshConfiguration = !state.lastConfigurationCheck || Date.now() - state.lastConfigurationCheck >= 30000;
    const connection = refreshConnection();
    const configuration = refreshConfiguration ? loadConfiguration() : null;
    await loadProjects();
    if (!state.selectedId) {
      showEmpty("No research projects yet", "Projects created through the AutoResearch plugin in ChatGPT will appear here.");
    } else {
      await refreshProject(state.selectedId, refreshConfiguration);
    }
    await Promise.all([connection, configuration]);
  } catch (error) {
    if (!state.current) showEmpty("Project status unavailable", error.message);
  } finally {
    state.refreshing = false;
  }
}

async function refreshConnection() {
  const [health, connection] = await Promise.allSettled([
    api("/api/health"), api("/api/chatgpt/connection")
  ]);
  const ready = health.status === "fulfilled" && health.value.ok === true;
  $("healthDot").classList.toggle("ok", ready);
  $("healthText").textContent = ready ? "Local service ready" : "Local service unavailable";
  $("localMcpStatus").textContent = ready ? "Local MCP ready" : "Local MCP unavailable";
  state.connection = connection.status === "fulfilled" ? connection.value : null;
  renderConnection();
}

function renderConnection() {
  const c = state.connection || {};
  $("tunnelInstallStatus").textContent = c.tunnel_client_installed ? "tunnel-client installed" : "tunnel-client status unavailable";
  $("tunnelConfigStatus").textContent = c.tunnel_configured ? "Tunnel configured" : "Tunnel configuration unavailable";
  let status = "ChatGPT connection unavailable";
  if (c.tunnel_ready) status = "ChatGPT tunnel ready";
  else if (c.tunnel_process_running) status = "ChatGPT tunnel starting · connection not ready";
  else if (c.tunnel_configured) status = "ChatGPT tunnel stopped";
  else if (state.connection) status = "ChatGPT tunnel setup incomplete";
  $("tunnelRunStatus").textContent = status;
  $("tunnelRunStatus").classList.toggle("ready", Boolean(c.tunnel_ready));
}

async function loadConfiguration() {
  state.lastConfigurationCheck = Date.now();
  const [settings, readiness] = await Promise.allSettled([
    api("/api/settings"), api("/api/autodl/preflight?verify_api=false")
  ]);
  const s = settings.status === "fulfilled" ? settings.value : null;
  $("gpuOrder").textContent = s?.autodl_gpu_specs?.join(" → ") || "—";
  $("monitorPolicy").textContent = s ? `Every ${s.monitor_seconds}s · Up to ${s.max_iterations} rounds` : "—";
  const report = readiness.status === "fulfilled" ? readiness.value : null;
  const messages = report?.ready
    ? ["AutoDL configuration available", ...(report.warnings || []).map((warning) => warning.message)]
    : (report?.blocking_issues || []).map((issue) => issue.message);
  $("autodlReadiness").textContent = messages.join("\n") || "AutoDL readiness unavailable";
  $("autodlReadiness").className = `status-message${report?.ready ? "" : " error"}`;
}

async function loadProjects() {
  const projects = await api("/api/projects");
  if (!state.selectedId && projects.length) state.selectedId = projects[0].id;
  const changed = JSON.stringify(projects) !== JSON.stringify(state.projects);
  state.projects = projects;
  if (changed) renderProjectList();
}

function renderProjectList() {
  const list = $("projectList");
  list.replaceChildren();
  for (const project of state.projects) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `project-item${state.selectedId === project.id ? " active" : ""}`;
    if (state.selectedId === project.id) button.setAttribute("aria-current", "true");
    const title = document.createElement("strong");
    title.textContent = project.topic;
    const meta = document.createElement("span");
    meta.textContent = `${phaseLabels[project.phase] || project.phase} · ${project.progress}%`;
    button.append(title, meta);
    button.onclick = () => selectProject(project.id);
    list.append(button);
  }
}

async function selectProject(id) {
  if (id === state.selectedId) return;
  state.selectedId = id;
  renderProjectList();
  showEmpty("Loading project", "Reading the selected project's status.");
  const query = new URLSearchParams({ project: id });
  if (state.colabFirst) query.set("compute", "colab");
  history.replaceState(null, "", `?${query}`);
  try {
    await refreshProject(id);
  } catch (error) {
    if (state.selectedId === id) showEmpty("Project status unavailable", error.message);
  }
}

async function refreshProject(id, refreshReadiness = false) {
  const project = state.projects.find((item) => item.id === id) || await api(projectPath(id));
  if (state.selectedId !== id) return;
  const changingProject = state.current?.id !== id;
  state.current = project;
  if (changingProject) {
    state.colab = null;
    state.colabResourceAgent = null;
    state.colabResourceAgentUnavailable = false;
    state.experimentResults = null;
    $("eventList").replaceChildren();
    $("sourceList").replaceChildren();
    $("sourceCount").textContent = "0";
    $("experimentResults").replaceChildren();
    $("experimentStatus").textContent = "Checking experiment status";
    $("experimentReadiness").className = "status-message";
    $("experimentReadiness").textContent = "Checking experiment requirements…";
    renderColab();
  }
  renderProject();
  $("emptyView").classList.add("hidden");
  $("projectView").classList.remove("hidden");
  await Promise.allSettled([
    loadEvents(), loadSources(), loadExperiments(), loadColab(),
    changingProject || refreshReadiness ? loadExperimentReadiness() : null
  ]);
}

function renderProject() {
  const p = state.current;
  $("projectTopic").textContent = p.topic;
  $("projectId").textContent = p.id;
  $("projectSummary").textContent = p.error || p.summary || "";
  $("statusBadge").textContent = phaseLabels[p.phase] || p.phase;
  $("statusBadge").className = `status-badge${p.status === "failed" ? " failed" : ""}`;
  $("progressText").textContent = `${p.progress}%`;
  $("progressRing").style.setProperty("--progress", `${p.progress * 3.6}deg`);
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
  const projectId = state.current.id;
  const events = await api(`${projectPath(projectId)}/events`);
  if (state.current?.id !== projectId) return;
  const list = $("eventList");
  list.replaceChildren();
  if (!events.length) {
    list.innerHTML = '<div class="empty">No research activity recorded</div>';
    return;
  }
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
  const projectId = state.current.id;
  const sources = await api(`${projectPath(projectId)}/sources`);
  if (state.current?.id !== projectId) return;
  $("sourceCount").textContent = sources.length;
  const list = $("sourceList");
  list.replaceChildren();
  if (!sources.length) {
    list.innerHTML = '<div class="empty">No sources retrieved</div>';
    return;
  }
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

async function loadExperimentReadiness() {
  const projectId = state.current.id;
  try {
    const report = await api(`${projectPath(projectId)}/readiness`);
    if (state.current?.id !== projectId) return;
    const details = report.blocking_issues.map((item) => `${item.path || item.code} · ${item.message}`);
    $("experimentReadiness").textContent = report.ready
      ? "Experiment files passed static checks"
      : ["Experiment readiness blocked", ...details].join("\n");
    $("experimentReadiness").className = `status-message${report.ready ? "" : " error"}`;
  } catch (error) {
    if (state.current?.id !== projectId) return;
    $("experimentReadiness").textContent = `Experiment readiness unavailable · ${error.message}`;
    $("experimentReadiness").className = "status-message error";
  }
}

async function loadExperiments() {
  const projectId = state.current.id;
  const experiments = await api(`${projectPath(projectId)}/experiments`);
  if (state.current?.id !== projectId) return;
  const serialized = JSON.stringify(experiments);
  if (state.experimentResults === serialized) return;
  state.experimentResults = serialized;
  const list = $("experimentResults");
  const openLogs = new Set(Array.from(list.querySelectorAll("details[open]"), (log) => log.parentElement.dataset.experimentId));
  list.replaceChildren();
  if (!experiments.length) {
    list.innerHTML = '<div class="empty compact">No experiments recorded</div>';
    $("experimentStatus").textContent = "Not started";
    return;
  }
  const newest = experiments[0];
  const statusLabels = { preparing: "Preparing", uploading: "Uploading", running: "Running", analyzing: "Awaiting analysis", completed: "Completed", failed: "Failed" };
  $("experimentStatus").textContent = statusLabels[newest.status] || newest.status;
  for (const experiment of experiments) {
    const card = document.createElement("article"); card.className = `experiment-result ${experiment.status}`;
    card.dataset.experimentId = experiment.id;
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
    if (experiment.error) {
      const error = document.createElement("p"); error.className = "result-error"; error.textContent = experiment.error;
      card.append(error);
    }
    if (result.log_tail) {
      const details = document.createElement("details");
      details.open = openLogs.has(experiment.id);
      const summary = document.createElement("summary"); summary.textContent = "View log tail";
      const log = document.createElement("pre"); log.textContent = String(result.log_tail).slice(-5000);
      details.append(summary, log); card.append(details);
    }
    list.append(card);
  }
}

function renderColab() {
  const job = state.colab || {};
  const labels = {
    not_prepared: "Not prepared / 未准备", idle: "Not prepared / 未准备", prepared: "Prepared / 已准备", queued: "Waiting for browser / 等待浏览器",
    checking: "Requesting GPU / 正在申请", waiting: "Waiting to retry / 等待重试", retry_wait: "Waiting to retry / 等待重试",
    attempting: "Requesting GPU / 正在申请", waiting_for_gpu: "Waiting for GPU retry / 等待 GPU 重试",
    waiting_for_connection: "Connection unavailable / 等待恢复连接", gpu_available: "GPU allocated / 已分配 GPU",
    available: "GPU allocated / 已分配 GPU", running: "Running / 运行中", completed: "Completed / 已完成",
    auth: "Waiting for Google authorization / 等待 Google 授权", awaiting_auth: "Waiting for Google authorization / 等待 Google 授权",
    auth_required: "Browser step pending / 浏览器步骤待完成", disconnected: "Session disconnected / 会话已断开",
    failed: "Failed / 失败", cancelled: "Retry stopped / 已停止重试", blocked: "Preparation blocked / 准备受阻"
  };
  const completed = job.completion || {};
  const completionLabels = {
    campaign_complete: "Experiment completed / 实验已完成",
    acceptance_passed: "Resume acceptance passed / 断线续训验收通过",
    process_exit: "Program finished / 程序运行结束"
  };
  const currentSource = job.source_current !== false;
  $("colabStatus").textContent = currentSource
    ? (job.status === "completed" && completionLabels[completed.completion_kind]) || labels[job.status] || job.status || "Checking status"
    : "Code changed / 准备版本已过期";
  const lines = [];
  if (!currentSource) lines.push("项目代码已更新，已准备的 Notebook 对应此前版本。");
  if (currentSource && job.status === "completed" && completed.completion_kind === "acceptance_passed") lines.push("断线续训功能验收通过，正式实验待完成。");
  if (currentSource && job.status === "completed") {
    for (const [name, value] of Object.entries(completed.metrics || {})) {
      if (Number.isFinite(value)) lines.push(`${name} · ${value}`);
    }
  }
  if (job.status === "auth_required") {
    const authKind = job.last_observation?.details?.auth_kind;
    const authStatus = {
      google_sign_in: "Waiting for Google sign-in / 等待 Google 登录",
      drive_consent: "Waiting for Google Drive authorization / 等待 Google Drive 授权",
      experiment_inputs: "Waiting for experiment inputs / 等待实验输入"
    };
    lines.push(authStatus[authKind] || "Waiting for Colab browser / 等待 Colab 浏览器状态");
  } else {
    if (job.blocked_reason) lines.push(job.blocked_reason);
    if (job.last_observation?.message) lines.push(job.last_observation.message);
  }
  if (job.next_attempt_at) lines.push(`Next attempt / 下次申请 · ${colabResourceTime(job.next_attempt_at)}`);
  if (job.attempt_count) lines.push(`Browser attempts / 浏览器连接次数 · ${job.attempt_count}`);
  if (job.drive_project_dir) lines.push(`Persistent files / 持久保存 · ${job.drive_project_dir}`);
  $("colabDetails").textContent = lines.join("\n") || (job.prepared ? "已准备 Colab 实验。" : state.colab ? "尚无 Colab 实验。" : "正在读取 Colab 状态。");
  $("colabDetails").className = `status-message${job.blocked_reason || job.status === "failed" ? " error" : ""}`;
  $("colabDownloads").classList.toggle("hidden", !job.prepared);
  if (job.prepared && state.current) {
    const base = `${projectPath(state.current.id)}/colab`;
    $("downloadColabNotebook").href = `${base}/notebook.ipynb`;
    $("downloadColabBundle").href = `${base}/bundle.zip`;
    $("downloadColabInputs").href = `${base}/inputs.zip`;
    $("downloadColabInputs").classList.toggle("hidden", !job.input_bundle_url);
  }
  renderColabResourceAgent();
}

function colabResourceView(snapshot, job, now = Date.now()) {
  const heartbeatTime = Date.parse(snapshot?.heartbeat_at || "");
  const heartbeatFresh = snapshot?.agent_heartbeat_fresh !== false && Number.isFinite(heartbeatTime) && now - heartbeatTime <= 45000 && now >= heartbeatTime - 10000;
  const checkedTime = Date.parse(snapshot?.last_checked_at || "");
  const checkFresh = Number.isFinite(checkedTime) && now - checkedTime <= 45000 && now >= checkedTime - 10000;
  const active = ["watching", "checking"].includes(snapshot?.agent_state);
  const allocated = ["gpu_available", "available", "running"];
  const hasGpu = job?.source_current !== false && snapshot?.worker_running === true && job?.worker_running === true
    && allocated.includes(snapshot?.job_status) && allocated.includes(job?.status);
  let availability = "unknown";
  if (heartbeatFresh && active && snapshot?.availability === "available" && hasGpu) availability = "available";
  if (heartbeatFresh && checkFresh && snapshot?.agent_state === "watching" && snapshot?.availability === "unavailable" && job?.status === "waiting_for_gpu") availability = "unavailable";
  const agentLabels = {
    watching: "持续监测中", checking: "正在检查 GPU", quota_wait: "等待额度冷却",
    auth_required: "等待登录或授权", stopped: "已停止", offline: "监测 Agent 离线"
  };
  const resultLabels = {
    available: "当前账号已分配 GPU", unavailable: "当前账号暂不可用", quota: "当前账号不可用（额度限制）",
    network: "连接失败，资源状态未确认", auth: "需要登录或授权，资源状态未确认", unknown: "尚未确认"
  };
  let agentLabel = "监测 Agent 尚未启动";
  if (snapshot?.agent_state === "stopped") agentLabel = agentLabels.stopped;
  else if (heartbeatFresh) agentLabel = agentLabels[snapshot.agent_state] || "监测中";
  else if (snapshot?.heartbeat_at) agentLabel = "监测 Agent 离线 · 状态已过期";
  return {
    availability,
    availabilityLabel: { available: "有 · 当前账号已分配 GPU", unavailable: "暂不可用 · 最近一次申请结果", unknown: "待确认" }[availability],
    agentLabel,
    lastResultLabel: resultLabels[snapshot?.last_result] || resultLabels.unknown,
    nextCheck: job?.next_attempt_at || snapshot?.next_check_at || null,
    heartbeatFresh
  };
}

function colabResourceTime(value) {
  const time = Date.parse(value || "");
  return Number.isFinite(time) ? new Date(time).toLocaleString("zh-CN") : "—";
}

function renderColabResourceAgent() {
  const snapshot = state.colabResourceAgent?.project_id === state.current?.id ? state.colabResourceAgent : null;
  const view = colabResourceView(snapshot, state.colab);
  $("colabResourceAvailability").textContent = view.availabilityLabel;
  $("colabResourceAvailability").className = `resource-availability ${view.availability}`;
  $("colabResourceAgentState").textContent = state.colabResourceAgentUnavailable ? "监测状态暂时不可用" : view.agentLabel;
  $("colabResourceHeartbeat").textContent = colabResourceTime(snapshot?.heartbeat_at);
  $("colabResourceLastResult").textContent = view.lastResultLabel;
  $("colabResourceLastChecked").textContent = colabResourceTime(snapshot?.last_checked_at);
  $("colabResourceNextCheck").textContent = colabResourceTime(view.nextCheck);
  $("colabResourceMessage").textContent = snapshot?.heartbeat_at ? snapshot.message || "" : "";
}

async function loadColabResourceAgent(projectId) {
  try {
    const snapshot = await api(`${projectPath(projectId)}/colab/resources`);
    if (state.current?.id !== projectId) return;
    if (snapshot.project_id !== projectId) throw new Error("Resource status belongs to another project");
    state.colabResourceAgent = snapshot;
    state.colabResourceAgentUnavailable = false;
  } catch (error) {
    if (state.current?.id !== projectId) return;
    state.colabResourceAgent = null;
    state.colabResourceAgentUnavailable = error.status !== 404;
  }
  renderColabResourceAgent();
}

async function loadColab() {
  const projectId = state.current.id;
  const loadJob = async () => {
    try {
      const job = await api(`${projectPath(projectId)}/colab`);
      if (state.current?.id !== projectId) return;
      state.colab = job;
      renderColab();
    } catch (error) {
      if (state.current?.id !== projectId) return;
      state.colab = null;
      renderColab();
      $("colabStatus").textContent = "Status unavailable";
      $("colabDetails").textContent = `Colab status unavailable · ${error.message}`;
      $("colabDetails").className = "status-message error";
    }
  };
  await Promise.all([loadJob(), loadColabResourceAgent(projectId)]);
}

bootstrap();
