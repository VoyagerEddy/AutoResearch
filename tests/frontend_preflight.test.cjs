const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const root = path.join(__dirname, '..', 'autoresearch', 'static');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8').replace(/bootstrap\(\);\s*$/, '');

function element() {
  const classes = new Set();
  return {
    textContent: '', children: [], dataset: {}, attributes: {},
    classList: {
      add(name) { classes.add(name); },
      remove(name) { classes.delete(name); },
      contains(name) { return classes.has(name); },
      toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); },
    },
    style: {setProperty() {}},
    setAttribute(name, value) { this.attributes[name] = value; },
    append(...children) { this.children.push(...children); },
    prepend(...children) { this.children = [...children, ...this.children.filter(child => !children.includes(child))]; },
    replaceChildren() { this.children = []; },
    querySelectorAll() { return []; },
  };
}

function ui(fetch, search = '') {
  const ids = [...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]);
  const nodes = new Map(ids.map(id => [id, element()]));
  const panelIds = [...html.matchAll(/<section id="(colabPanel|autodlPanel)"/g)].map(m => m[1]);
  nodes.get('computePanels').append(...panelIds.map(id => nodes.get(id)));
  const timers = [];
  const location = {search};
  const context = vm.createContext({
    document: {
      getElementById(id) { assert.ok(nodes.has(id), "Unknown DOM ID " + id); return nodes.get(id); },
      querySelectorAll() { return []; },
      createElement() { return element(); },
    },
    fetch, AbortController, URLSearchParams, location,
    history: {replaceState(_state, _title, query) { location.search = query; }},
    setTimeout() { return 1; }, clearTimeout() {},
    setInterval(handler) { timers.push(handler); },
  });
  vm.runInContext(source, context);
  return { nodes, context, timers, location };
}

const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const project = (id, phase = 'ready') => ({ id, topic: 'Project ' + id, phase, status: phase, progress: 70 });
const ready = { ready: true, blocking_issues: [], warnings: [] };

function statusResponse(url, projects) {
  if (url === '/api/health') return response({ok: true});
  if (url === '/api/chatgpt/connection') return response({tunnel_ready: true});
  if (url === '/api/settings') return response({autodl_gpu_specs: ['v-48g'], monitor_seconds: 30, max_iterations: 3});
  if (url.includes('/preflight')) return response(ready);
  if (url === '/api/projects') return response(projects);
  if (url.endsWith('/readiness')) return response(ready);
  if (url.endsWith('/colab')) return response({prepared: true, status: 'waiting_for_gpu'});
  if (url.endsWith('/colab/resources')) return response({project_id: url.split('/')[3], agent_state: 'watching'});
  if (/\/(events|sources|experiments)$/.test(url)) return response([]);
  if (url === '/api/projects/earlier') return response(project('earlier'));
  throw new Error('Unexpected status request ' + url);
}

test('dashboard has no forms or operation controls', () => {
  assert.doesNotMatch(html, /<(form|input|select|textarea|button|dialog)(\s|>)/i);
  assert.doesNotMatch(source, /\b(method\s*:|confirm\(|prompt\(|alert\()/);
});

test('latest ChatGPT project and its compute status appear automatically using only status requests', async () => {
  const calls = [];
  let projects = [];
  const app = ui(async (url, options) => {
    calls.push({url, method: options.method || 'GET'});
    return statusResponse(url, projects);
  });
  await vm.runInContext('bootstrap()', app.context);
  assert.match(app.nodes.get('emptyTitle').textContent, /No research projects/);
  assert.equal(app.timers.length, 1);
  projects = [project('latest'), project('earlier')];
  await app.timers[0]();
  assert.equal(app.nodes.get('projectId').textContent, 'latest');
  assert.match(app.nodes.get('colabStatus').textContent, /Waiting for GPU/);
  assert.equal(app.nodes.get('experimentStatus').textContent, 'Not started');
  assert.equal(app.nodes.get('computePanels').children[0], app.nodes.get('autodlPanel'));
  assert.ok(calls.some(call => call.url === '/api/projects/latest/colab/resources'));
  assert.ok(calls.every(call => call.method === 'GET'));
  assert.equal(calls.some(call => /\/(check|launch|start|resume|login|prepare|cancel)$/.test(call.url)), false);
});

test('project navigation changes the status display and uses the existing deep link', async () => {
  const calls = [];
  const app = ui(async (url, options) => {
    calls.push({url, method: options.method || 'GET'});
    return statusResponse(url, [project('latest'), project('earlier')]);
  }, '?project=earlier&compute=colab');
  await vm.runInContext('bootstrap()', app.context);
  assert.equal(app.nodes.get('projectId').textContent, 'earlier');
  assert.equal(app.nodes.get('computePanels').children[0], app.nodes.get('colabPanel'));
  await app.nodes.get('projectList').children[0].onclick();
  assert.equal(app.nodes.get('projectId').textContent, 'latest');
  assert.equal(app.location.search, '?project=latest&compute=colab');
  assert.ok(calls.every(call => call.method === 'GET'));
});

test('a direct link can load an older project outside the list window', async () => {
  const app = ui(async url => statusResponse(url, [project('latest')]), '?project=earlier&compute=colab');
  await vm.runInContext('bootstrap()', app.context);
  assert.equal(app.nodes.get('projectId').textContent, 'earlier');
});

test('running launcher reports a connection that is still starting', () => {
  const app = ui(async () => { throw new Error('No request expected'); });
  vm.runInContext('state.connection = {tunnel_configured: true, tunnel_process_running: true, tunnel_ready: false}; renderConnection()', app.context);
  assert.match(app.nodes.get('tunnelRunStatus').textContent, /not ready/i);
});

test('project input blockers are visible without executing commands', async () => {
  const calls = [];
  const app = ui(async (url, options) => {
    calls.push({url, method: options.method || 'GET'});
    return response({ ready: false, blocking_issues: [{code: 'input', path: '/remote/data', message: 'Remote input remains unverified'}] });
  });
  await vm.runInContext("state.current = {id: 'project'}; loadExperimentReadiness()", app.context);
  assert.deepEqual(calls, [{url: '/api/projects/project/readiness', method: 'GET'}]);
  assert.match(app.nodes.get('experimentReadiness').textContent, /unverified/);
});

test('Colab waiting state displays a retry deadline without claiming GPU allocation', () => {
  const app = ui(async () => { throw new Error('No request expected'); });
  vm.runInContext("state.current = {id: 'sample-project'}; state.colab = {prepared: true, status: 'waiting', next_attempt_at: '2026-10-08T12:00:00Z', attempt_count: 1}; renderColab()", app.context);
  assert.match(app.nodes.get('colabStatus').textContent, /Waiting/);
  assert.doesNotMatch(app.nodes.get('colabStatus').textContent, /allocated|Training/);
  assert.match(app.nodes.get('colabDetails').textContent, /Next attempt/);
  assert.equal(app.nodes.get('downloadColabNotebook').href, '/api/projects/sample-project/colab/notebook.ipynb');
});

test('Colab shows the runtime phase, environment mismatch and release confirmation', () => {
  const app = ui();
  vm.runInContext(`state.current = {id: 'sample-project'};
    state.colab = {prepared: true, status: 'failed', stage: 'runtime_setup', runtime_release: {status: 'released'},
      runtime_requirements: {torch_version: '2.9.1+cu126'},
      diagnostics: {torch_version: '2.14.1+cu130', preflight_error: 'Exact PyTorch version differs'}};
    renderColab();`, app.context);
  const details = app.nodes.get('colabDetails').textContent;
  assert.match(details, /安装依赖和运行时预检/);
  assert.match(details, /2\.14\.1\+cu130 \/ 要求 2\.9\.1\+cu126/);
  assert.match(details, /Exact PyTorch version differs/);
  assert.match(details, /已确认释放 GPU 会话/);
});

test('Colab authorization is displayed as status without a dashboard action', async () => {
  const calls = [];
  const app = ui(async (url, options) => {
    calls.push({url, method: options.method || 'GET'});
    if (url.endsWith('/resources')) return response({project_id: 'sample-project', availability: 'unknown'});
    return response({
      prepared: true,
      status: 'auth_required',
      blocked_reason: 'Complete Google Drive authorization, then click Continue in AutoResearch',
      last_observation: {details: {auth_kind: 'drive_consent'}},
    });
  });
  await vm.runInContext("state.current = {id: 'sample-project'}; loadColab()", app.context);
  assert.match(app.nodes.get('colabStatus').textContent, /Browser step pending/i);
  assert.match(app.nodes.get('colabDetails').textContent, /Waiting for Google Drive authorization/i);
  assert.doesNotMatch(app.nodes.get('colabDetails').textContent, /click Continue/i);
  assert.match(app.nodes.get('colabResourceAvailability').textContent, /待确认/);
  assert.deepEqual(calls, [
    {url: '/api/projects/sample-project/colab', method: 'GET'},
    {url: '/api/projects/sample-project/colab/resources', method: 'GET'},
  ]);
});

test('a slow Colab status response cannot overwrite a different selected project', async () => {
  let resolve;
  const app = ui(async url => {
    if (url.endsWith('/resources')) return response({}, 404);
    return new Promise(done => { resolve = done; });
  });
  const load = vm.runInContext("state.current = {id: 'sample-project'}; loadColab()", app.context);
  vm.runInContext("state.current = {id: 'other'}; state.colab = {status: 'running'}", app.context);
  resolve(response({prepared: true, status: 'waiting'}));
  await load;
  assert.equal(vm.runInContext('state.colab.status', app.context), 'running');
});

test('a superseded Colab run does not display its completed metrics as current', () => {
  const app = ui(async () => { throw new Error('No request expected'); });
  vm.runInContext(`
    state.current = {id: 'sample-project'};
    state.colab = {prepared: true, status: 'completed', source_current: true,
      completion: {completion_kind: 'campaign_complete', metrics: {score: 0.42}}};
    renderColab();
  `, app.context);
  assert.match(app.nodes.get('colabDetails').textContent, /0\.42/);
  vm.runInContext('state.colab.source_current = false; renderColab()', app.context);
  assert.match(app.nodes.get('colabStatus').textContent, /Code changed/);
  assert.doesNotMatch(app.nodes.get('colabDetails').textContent, /0\.42/);
});

const monitorNow = Date.parse('2026-10-09T05:00:00Z');
const monitorSnapshot = (extra = {}) => ({
  project_id: 'sample-project', agent_state: 'watching', heartbeat_at: '2026-10-09T05:00:00Z',
  availability: 'available', last_result: 'available', last_checked_at: '2026-10-09T04:59:50Z',
  job_status: 'running', worker_running: true, ...extra
});
function resourceView(app, snapshot, job = { status: 'running', worker_running: true }) {
  return vm.runInContext(`colabResourceView(${JSON.stringify(snapshot)}, ${JSON.stringify(job)}, ${monitorNow})`, app.context);
}

test('resource availability requires a fresh agent and a live allocated worker', () => {
  const app = ui(async () => { throw new Error('No request expected'); });
  assert.equal(resourceView(app, monitorSnapshot()).availability, 'available');
  assert.equal(resourceView(app, monitorSnapshot({worker_running: false})).availability, 'unknown');
  assert.equal(resourceView(app, monitorSnapshot(), {status: 'waiting_for_gpu', worker_running: false}).availability, 'unknown');
  assert.equal(resourceView(app, monitorSnapshot(), {}).availability, 'unknown');
});

test('an old allocated result remains historical when the monitor heartbeat expires', () => {
  const app = ui(async () => { throw new Error('No request expected'); });
  const view = resourceView(app, monitorSnapshot({heartbeat_at: '2026-10-09T04:58:00Z'}));
  assert.equal(view.availability, 'unknown');
  assert.match(view.agentLabel, /离线/);
  assert.match(view.lastResultLabel, /已分配/);
});

test('recent GPU denial is displayed separately from an older denial or a quota cooldown', () => {
  const app = ui(async () => { throw new Error('No request expected'); });
  const denial = monitorSnapshot({availability: 'unavailable', last_result: 'unavailable', job_status: 'waiting_for_gpu', worker_running: false});
  const waiting = {status: 'waiting_for_gpu', worker_running: false};
  assert.equal(resourceView(app, denial, waiting).availability, 'unavailable');
  assert.equal(resourceView(app, {...denial, last_checked_at: '2026-10-09T04:55:00Z'}, waiting).availability, 'unknown');
  const quota = resourceView(app, {...denial, agent_state: 'quota_wait', last_result: 'quota'}, waiting);
  assert.equal(quota.availability, 'unknown');
  assert.match(quota.lastResultLabel, /不可用（额度限制）/);
});

test('checking does not carry forward an earlier resource denial', () => {
  const app = ui(async () => { throw new Error('No request expected'); });
  const view = resourceView(app, monitorSnapshot({agent_state: 'checking', availability: 'unavailable', last_result: 'unavailable'}));
  assert.equal(view.availability, 'unknown');
  assert.match(view.agentLabel, /正在检查/);
});

test('resource polling only reads uncached status and does not request GPU allocation', async () => {
  const calls = [];
  const app = ui(async (url, options) => {
    calls.push({url, cache: options.cache});
    if (url.endsWith('/colab/resources')) return response(monitorSnapshot({heartbeat_at: new Date().toISOString(), availability: 'unknown', agent_state: 'quota_wait', last_result: 'quota'}));
    return response({prepared: true, status: 'waiting_for_gpu', worker_running: false});
  });
  await vm.runInContext("state.current = {id: 'sample-project'}; loadColab()", app.context);
  assert.equal(calls.length, 2);
  assert.deepEqual(calls.find(call => call.url.endsWith('/colab/resources')), {url: '/api/projects/sample-project/colab/resources', cache: 'no-store'});
  assert.match(app.nodes.get('colabResourceAvailability').textContent, /待确认/);
  assert.match(app.nodes.get('colabResourceLastResult').textContent, /额度限制/);
});

test('a slow resource monitor response cannot overwrite a newly selected project', async () => {
  let resolveStatus;
  const app = ui(() => new Promise(resolve => { resolveStatus = resolve; }));
  const load = vm.runInContext("state.current = {id: 'sample-project'}; loadColabResourceAgent('sample-project')", app.context);
  vm.runInContext("state.current = {id: 'other'}; state.colabResourceAgent = {project_id: 'other', agent_state: 'stopped'}", app.context);
  resolveStatus(response(monitorSnapshot()));
  await load;
  assert.equal(vm.runInContext('state.colabResourceAgent.project_id', app.context), 'other');
});

test('missing monitor status keeps the saved retry deadline without inventing availability', async () => {
  const app = ui(async () => response({}, 404));
  await vm.runInContext("state.current = {id: 'sample-project'}; state.colab = {status: 'waiting_for_gpu', next_attempt_at: '2026-10-09T05:10:00Z'}; loadColabResourceAgent('sample-project')", app.context);
  assert.match(app.nodes.get('colabResourceAvailability').textContent, /待确认/);
  assert.match(app.nodes.get('colabResourceAgentState').textContent, /尚未启动/);
  assert.notEqual(app.nodes.get('colabResourceNextCheck').textContent, '—');
});

test('job API failure suppresses a fresh allocation claimed by the resource monitor', async () => {
  const app = ui(async url => {
    if (url.endsWith('/colab/resources')) return response(monitorSnapshot({heartbeat_at: new Date().toISOString()}));
    return response({detail: 'status unavailable'}, 503);
  });
  await vm.runInContext("state.current = {id: 'sample-project'}; loadColab()", app.context);
  assert.match(app.nodes.get('colabResourceAvailability').textContent, /待确认/);
  assert.match(app.nodes.get('colabDetails').textContent, /unavailable/);
});
