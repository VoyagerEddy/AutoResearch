const state = { projects: [], current: null, settings: null, connection: null, poller: null, connectionPoller: null, experimentId: null };
const phaseLabels = {
  queued: "排队中", planning: "问题规划", searching: "资源检索", synthesizing: "算法形成",
  generating: "代码生成", ready: "实验就绪", experiment: "远程实验", analyzing: "结果分析",
  completed: "已完成", failed: "失败", experiment_failed: "实验失败",
  chatgpt_thinking: "ChatGPT 思考中", chatgpt_analysis: "ChatGPT 已分析"
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
    if (!response.ok) throw new Error(data.detail || `请求失败 (${response.status})`);
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("本地服务响应超时，请确认启动窗口仍在运行");
    if (error instanceof TypeError) throw new Error("无法连接本地服务，请重新运行 start.cmd");
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
    $("healthText").textContent = "本地服务与 ChatGPT 工具已就绪";
    $("mcpEndpoint").textContent = health.mcp_endpoint;
    [state.settings, state.connection] = await Promise.all([
      api("/api/settings"), api("/api/chatgpt/connection")
    ]);
    applySettings();
    renderConnection();
    clearInterval(state.connectionPoller);
    state.connectionPoller = setInterval(refreshConnection, 5000);
    await loadProjects();
    const requested = new URLSearchParams(location.search).get("project");
    if (requested && state.projects.some((project) => project.id === requested)) await selectProject(requested);
  } catch (error) {
    $("healthText").textContent = "本地服务不可用";
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
  setStep("tunnelInstallStatus", c.tunnel_client_installed, "tunnel-client 已安装", "tunnel-client 未安装");
  setStep("tunnelConfigStatus", c.tunnel_configured, `Tunnel 已配置 ${c.tunnel_id_hint || ""}`, "Tunnel 待配置");
  setStep("tunnelRunStatus", c.tunnel_running, "Tunnel 正在运行", "Tunnel 未运行");
  $("startChatgpt").disabled = !c.tunnel_configured;
  $("settingsStartChatgpt").disabled = !c.tunnel_configured;
  $("setupChatgpt").textContent = c.tunnel_configured ? "重新配置连接" : "首次配置连接";
  $("settingsTunnelStatus").textContent = c.tunnel_running
    ? "Tunnel 已连接，可在 ChatGPT 中使用"
    : (c.tunnel_configured ? "配置已保存，Tunnel 当前未运行" : "尚未完成首次配置");
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
    toast(action === "Setup" ? "首次配置向导已打开" : "ChatGPT 连接窗口已打开");
    setTimeout(refreshConnection, 1500);
  } catch (error) { toast(error.message, true); }
}

function applySettings() {
  const s = state.settings || {};
  $("gpuOrder").textContent = (s.autodl_gpu_specs || ["v-48g", "5090"]).join(" → ");
  $("monitorPolicy").textContent = `每 ${s.monitor_seconds || 30} 秒 · 最多 ${s.max_iterations || 3} 轮`;
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
    ? "ChatGPT 负责推理；AutoResearch 等待工具调用"
    : (state.settings?.openrouter_configured ? `OpenRouter · ${state.settings.openrouter_model}` : "未配置 Key 时将生成离线基线");
  $("startResearch").innerHTML = collaborative ? "创建协作项目 <b>→</b>" : "开始全自动研究 <b>→</b>";
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
  $("pageTitle").textContent = "研究工作台";
  $("openCode").disabled = false;
  $("syncGit").disabled = state.current.status !== "ready";
  history.replaceState(null, "", `?project=${encodeURIComponent(id)}`);
  renderProject();
  renderProjectList();
  await Promise.all([loadEvents(), loadSources(), loadManifest(), loadExperiments()]);
  beginPolling();
}

function renderProject() {
  const p = state.current;
  if (!p) return;
  $("projectTopic").textContent = p.topic;
  $("projectId").textContent = p.id;
  $("projectSummary").textContent = p.error || p.summary || "研究代理正在工作，产物会持续保存到本地工作区。";
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
  if (!events.length) { list.innerHTML = '<div class="empty">等待代理开始工作…</div>'; return; }
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
  if (!sources.length) { list.innerHTML = '<div class="empty">检索后会在这里显示来源</div>'; return; }
  for (const source of sources) {
    const a = document.createElement("a"); a.className = "source";
    if (/^https:\/\//.test(source.url)) { a.href = source.url; a.target = "_blank"; a.rel = "noreferrer"; }
    const meta = document.createElement("div"); meta.className = "meta";
    meta.textContent = `${source.provider} · ${source.kind === "code" ? "CODE" : (source.year || "PAPER")} · ${source.citation_count || 0} 引用/星标`;
    const title = document.createElement("h4"); title.textContent = source.title;
    const desc = document.createElement("p"); desc.textContent = source.abstract || source.url;
    a.append(meta, title, desc); list.append(a);
  }
}

async function loadManifest() {
  if (!state.current || state.current.status !== "ready") return;
  const manifest = await api(`/api/projects/${state.current.id}/manifest`);
  if (manifest.run_command) $("remoteCommand").value = manifest.run_command;
}

async function loadExperiments() {
  if (!state.current) return;
  const experiments = await api(`/api/projects/${state.current.id}/experiments`);
  const list = $("experimentResults"); list.replaceChildren();
  if (!experiments.length) {
    state.experimentId = null;
    list.innerHTML = '<div class="empty compact">实验状态、指标和日志会显示在这里</div>';
    $("experimentStatus").textContent = "尚未启动";
    return;
  }
  const active = experiments.find((item) => !["completed", "failed"].includes(item.status));
  state.experimentId = active ? active.id : null;
  const newest = experiments[0];
  const statusLabels = { preparing: "准备中", uploading: "上传中", running: "运行中", analyzing: "等待分析", completed: "已完成", failed: "失败" };
  $("experimentStatus").textContent = `${statusLabels[newest.status] || newest.status} · ${newest.id}`;
  for (const experiment of experiments) {
    const card = document.createElement("article"); card.className = `experiment-result ${experiment.status}`;
    const head = document.createElement("div"); head.className = "result-head";
    const title = document.createElement("strong"); title.textContent = `实验 ${experiment.id}`;
    const badge = document.createElement("span"); badge.textContent = statusLabels[experiment.status] || experiment.status;
    head.append(title, badge);
    const meta = document.createElement("p"); meta.textContent = `第 ${experiment.iteration} 轮 · ${experiment.command}`;
    card.append(head, meta);
    const result = experiment.result || {};
    if (Object.keys(result.metrics || {}).length) {
      const metrics = document.createElement("pre"); metrics.textContent = JSON.stringify(result.metrics, null, 2);
      card.append(metrics);
    }
    if (experiment.error) { const error = document.createElement("p"); error.className = "result-error"; error.textContent = experiment.error; card.append(error); }
    if (result.log_tail) {
      const details = document.createElement("details");
      const summary = document.createElement("summary"); summary.textContent = "查看日志末尾";
      const log = document.createElement("pre"); log.textContent = String(result.log_tail).slice(-5000);
      details.append(summary, log); card.append(details);
    }
    if (["completed", "failed"].includes(experiment.status)) {
      const analyze = document.createElement("button"); analyze.type = "button"; analyze.className = "ghost analyze-result";
      analyze.textContent = "复制 ChatGPT 分析请求";
      analyze.onclick = () => copyText(`请调用 AutoResearch 的 get_experiment_result，分析实验 ${experiment.id}，再用 record_chatgpt_analysis 把结论保存到项目 ${experiment.project_id}。`, "分析请求已复制");
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
  const button = event.submitter; button.disabled = true; button.textContent = "正在创建…";
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
    if (collaborative) toast(`协作项目 ${project.id} 已创建；在 ChatGPT 中使用这个项目 ID`);
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; updateWorkflowMode(); }
});

$("newProject").onclick = () => {
  state.current = null; clearInterval(state.poller);
  $("projectView").classList.add("hidden"); $("createView").classList.remove("hidden");
  history.replaceState(null, "", location.pathname);
  $("pageTitle").textContent = "把一个想法，变成可复现实验";
  $("openCode").disabled = true; $("syncGit").disabled = true; renderProjectList();
};

function copyText(value, message) {
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(value).then(() => toast(message)).catch(() => toast("复制失败，请手动选择文本", true));
  } else { toast("当前浏览器不支持自动复制，请手动选择文本", true); }
}

$("workflowMode").addEventListener("change", updateWorkflowMode);
$("copyMcp").onclick = () => copyText($("mcpEndpoint").textContent, "MCP 地址已复制");
$("copyProjectId").onclick = () => copyText($("projectId").textContent, "项目 ID 已复制");
$("setupChatgpt").onclick = () => launchChatgptConnection("Setup");
$("startChatgpt").onclick = () => launchChatgptConnection("Run");
$("settingsSetupChatgpt").onclick = () => launchChatgptConnection("Setup");
$("settingsStartChatgpt").onclick = () => launchChatgptConnection("Run");

$("openCode").onclick = async () => {
  try { await api(`/api/projects/${state.current.id}/open`, { method: "POST" }); toast("已发送到 VS Code"); }
  catch (error) { toast(error.message, true); }
};

$("syncGit").onclick = async () => {
  if (!state.settings?.github_remote_url) { $("settingsDialog").showModal(); return toast("请先配置 GitHub 仓库", true); }
  try {
    const result = await api("/api/git/sync", { method: "POST", body: JSON.stringify({ project_id: state.current.id }) });
    toast(result.pushed ? "实验代码已推送到 GitHub" : "代码已提交到本地 Git");
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
  saveButton.textContent = "保存中…";
  status.className = "form-status";
  status.textContent = "正在写入本机配置…";
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
    status.textContent = "设置已保存并立即生效。";
    toast("设置已保存到本机");
    setTimeout(() => { if ($("settingsDialog").open) $("settingsDialog").close(); }, 450);
  } catch (error) {
    status.className = "form-status error";
    status.textContent = error.message;
    toast(error.message, true);
  } finally {
    saveButton.disabled = false;
    saveButton.textContent = "保存设置";
  }
});

$("toggleExperiment").onclick = () => $("experimentForm").classList.toggle("hidden");

$("createInstance").onclick = async () => {
  if (!confirm("AutoDL Pro 实例启动后将按量计费。确认创建首选 GPU 实例吗？")) return;
  try {
    const result = await api("/api/autodl/instances", { method: "POST", body: JSON.stringify({ confirm_billable: true }) });
    $("instanceUuid").value = result.instance_uuid;
    toast(`实例 ${result.instance_uuid} 正在创建（规格 ${result.gpu_spec}）`);
  } catch (error) { toast(error.message, true); }
};

$("loadSsh").onclick = async () => {
  const id = $("instanceUuid").value.trim(); if (!id) return toast("请先填写实例 UUID", true);
  try {
    const result = await api(`/api/autodl/instances/${encodeURIComponent(id)}`);
    if (!result.ssh) return toast(`实例状态 ${result.status}，暂未取得 SSH 信息`, true);
    $("sshHost").value = result.ssh.host || ""; $("sshPort").value = result.ssh.port || 22;
    $("sshUser").value = result.ssh.username || "root"; $("sshPassword").value = result.ssh.password || "";
    toast("已载入 SSH 信息");
  } catch (error) { toast(error.message, true); }
};

$("experimentForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!confirm("即将把生成代码上传到远程主机并执行所示命令。是否继续？")) return;
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
    state.experimentId = result.experiment_id; $("experimentStatus").textContent = "准备中"; beginPolling(); toast("实验任务已启动");
  } catch (error) { toast(error.message, true); }
});

async function refreshExperiment() {
  const result = await api(`/api/experiments/${state.experimentId}`);
  const labels = { preparing: "准备中", uploading: "上传中", running: `运行中 · 第 ${result.iteration} 轮`, analyzing: "分析中", completed: "已完成", failed: "失败" };
  $("experimentStatus").textContent = labels[result.status] || result.status;
  if (["completed", "failed"].includes(result.status)) {
    if (result.status === "failed" && result.error) toast(result.error, true);
    state.experimentId = null;
    await loadExperiments();
  }
}

bootstrap();
