const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// Run the actual UI handlers against controlled HTTP responses. This verifies
// billing request counts and locks; it is not a visual/browser layout test.
function ui(fetch) {
  const root = path.join(__dirname, '..', 'autoresearch', 'static');
  const ids = [...fs.readFileSync(path.join(root, 'index.html'), 'utf8').matchAll(/id="([^"]+)"/g)].map(m => m[1]);
  const nodes = new Map(ids.map(id => [id, {
    textContent: '', value: '', disabled: false,
    classList: { toggle() {}, add() {}, remove() {} },
    listeners: {},
    addEventListener(event, handler) { this.listeners[event] = handler; },
  }]));
  let confirmations = 0;
  const context = vm.createContext({
    document: {
      getElementById(id) { assert.ok(nodes.has(id), `Unknown DOM id: ${id}`); return nodes.get(id); },
      querySelectorAll() { return []; },
    },
    fetch, AbortController, history: {replaceState() {}},
    setTimeout() { return 1; }, clearTimeout() {},
    confirm() { confirmations++; return true; },
  });
  const source = fs.readFileSync(path.join(root, 'app.js'), 'utf8').replace(/bootstrap\(\);\s*$/, '');
  vm.runInContext(source, context);
  return { nodes, context, confirmations: () => confirmations, click: () => nodes.get('createInstance').onclick() };
}

const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const ready = { ready: true, verified: false, blocking_issues: [], warnings: [] };

test('missing configuration prevents confirmation and billable request', async () => {
  const calls = [];
  const app = ui(async url => {
    calls.push(url);
    return response({ ...ready, ready: false, blocking_issues: [{ message: 'missing token and image' }] });
  });
  await app.click();
  assert.equal(app.confirmations(), 0);
  assert.equal(calls.length, 1);
  assert.match(calls[0], /preflight/);
  assert.equal(app.nodes.get('createInstance').disabled, true);
});

test('concurrent clicks send exactly one create request', async () => {
  let creates = 0;
  const app = ui(async url => {
    if (url.includes('preflight')) return response(ready);
    creates++;
    return response({ instance_uuid: 'pro-test', gpu_spec: 'v-48g' });
  });
  await Promise.all([app.click(), app.click()]);
  assert.equal(creates, 1);
  assert.equal(app.confirmations(), 1);
  assert.equal(app.nodes.get('instanceUuid').value, 'pro-test');
});

test('uncertain creation remains locked after preflight refresh and another click', async () => {
  let creates = 0;
  const app = ui(async url => {
    if (url.includes('preflight')) return response(ready);
    creates++;
    return response({ detail: { code: 'autodl_create_uncertain', message: 'Check instances first' } }, 409);
  });
  await app.click();
  await vm.runInContext('refreshAutoDL()', app.context);
  await app.click();
  assert.equal(creates, 1);
  assert.equal(app.nodes.get('createInstance').disabled, true);
  assert.match(app.nodes.get('createInstance').textContent, /result unknown/i);
});

test('running launcher is not rendered as connected', () => {
  const app = ui(async () => { throw new Error('No request expected'); });
  vm.runInContext('state.connection = {tunnel_configured: true, tunnel_process_running: true, tunnel_ready: false}; renderConnection()', app.context);
  assert.match(app.nodes.get('tunnelRunStatus').textContent, /not ready/i);
  assert.doesNotMatch(app.nodes.get('settingsTunnelStatus').textContent, /connected/i);
});

test('project input blockers are visible without executing commands', async () => {
  const calls = [];
  const app = ui(async url => {
    calls.push(url);
    return response({ ready: false, blocking_issues: [{code: 'input', path: '/remote/data', message: 'Remote input remains unverified'}] });
  });
  await vm.runInContext("state.current = {id: 'project'}; loadExperimentReadiness()", app.context);
  assert.deepEqual(calls, ['/api/projects/project/readiness']);
  assert.match(app.nodes.get('experimentReadiness').textContent, /unverified/);
  assert.equal(app.nodes.get('checkExperimentReadiness').disabled, false);
});

test('an absent run command clears a previous project command', async () => {
  const app = ui(async () => response({}));
  app.nodes.get('remoteCommand').value = 'old command';
  await vm.runInContext("state.current = {id: 'new-project', status: 'running'}; loadManifest()", app.context);
  assert.equal(app.nodes.get('remoteCommand').value, '');
});

test('Colab waiting state displays a retry deadline without claiming GPU allocation', () => {
  const app = ui(async () => { throw new Error('No request expected'); });
  vm.runInContext("state.current = {id: 'speaker'}; state.colab = {prepared: true, status: 'waiting', next_attempt_at: '2026-10-08T12:00:00Z', attempt_count: 1}; renderColab()", app.context);
  assert.match(app.nodes.get('colabStatus').textContent, /Waiting/);
  assert.doesNotMatch(app.nodes.get('colabStatus').textContent, /allocated|Training/);
  assert.match(app.nodes.get('colabDetails').textContent, /Next attempt/);
  assert.equal(app.nodes.get('downloadColabNotebook').href, '/api/projects/speaker/colab/notebook.ipynb');
});

test('concurrent Colab launch clicks start exactly one worker', async () => {
  let launches = 0;
  const app = ui(async url => {
    if (url.endsWith('/launch')) { launches++; return response({status: 'launched'}); }
    return response({prepared: true, status: 'auth'});
  });
  vm.runInContext("state.current = {id: 'speaker'}; state.colab = {prepared: true}", app.context);
  await Promise.all([app.nodes.get('startColab').onclick(), app.nodes.get('startColab').onclick()]);
  assert.equal(launches, 1);
  assert.match(app.nodes.get('colabStatus').textContent, /Login/);
});

test('a slow Colab status response cannot overwrite a different selected project', async () => {
  const app = ui(async () => {
    vm.runInContext("state.current = {id: 'other'}; state.colab = {status: 'running'}", app.context);
    return response({prepared: true, status: 'waiting'});
  });
  await vm.runInContext("state.current = {id: 'speaker'}; loadColab()", app.context);
  assert.equal(vm.runInContext('state.colab.status', app.context), 'running');
});

test('a Colab project link opens the resource panel and retains it across refresh', async () => {
  let checks = 0;
  const app = ui(async url => {
    if (url.endsWith('/check')) { checks++; return response({job: {prepared: true, status: 'queued'}}); }
    return response(url === '/api/projects/speaker' ? {id: 'speaker', status: 'ready'} : {});
  });
  app.context.location = {search: '?project=speaker&compute=colab'};
  app.context.URLSearchParams = URLSearchParams;
  app.context.history = {replaceState(_state, _title, query) { app.context.location.search = query; }};
  app.context.setInterval = () => 1;
  app.context.clearInterval = () => {};
  vm.runInContext(`
    applySettings = renderConnection = renderProject = renderProjectList = beginPolling = () => {};
    refreshAutoDL = loadEvents = loadSources = loadManifest = loadExperiments = loadExperimentReadiness = loadColabResourceAgent = async () => {};
    loadColab = async () => { state.colab = {prepared: true, status: 'queued'}; };
    loadProjects = async () => { state.projects = [{id: 'speaker'}]; };
  `, app.context);
  await vm.runInContext('bootstrap()', app.context);
  assert.equal(app.nodes.get('computeProvider').value, 'colab');
  assert.equal(new URLSearchParams(app.context.location.search).get('compute'), 'colab');
  assert.equal(checks, 1);
});

test('choosing Colab checks prepared project once and polling sends no allocation requests', async () => {
  const calls = [];
  const app = ui(async (url, options) => {
    calls.push({url, method: options.method || 'GET'});
    if (url.endsWith('/check')) return response({status: 'retry_scheduled', job: {prepared: true, status: 'waiting_for_gpu'}});
    if (url.includes('/static/')) return response({project_id: 'speaker', availability: 'unknown'});
    return response({prepared: true, status: 'waiting_for_gpu'});
  });
  vm.runInContext("state.current = {id: 'speaker'}; state.colab = {prepared: true, status: 'queued'}", app.context);
  app.nodes.get('computeProvider').value = 'colab';
  const select = app.nodes.get('computeProvider').listeners.change;
  await Promise.all([select(), select()]);
  await select();
  await vm.runInContext('loadColab();', app.context);
  assert.deepEqual(calls.filter(call => call.method === 'POST'), [{url: '/api/projects/speaker/colab/check', method: 'POST'}]);
});

test('Colab selection shows manual action without calling resume or paid resources', async () => {
  const calls = [];
  const app = ui(async (url, options) => {
    calls.push(url);
    if (url.endsWith('/check')) return response({status: 'auth_required', job: {prepared: true, status: 'auth_required'}});
    return response({project_id: 'speaker', agent_state: 'auth_required', availability: 'unknown', heartbeat_at: new Date().toISOString()});
  });
  vm.runInContext("state.current = {id: 'speaker'}; state.colab = {prepared: true, status: 'auth_required'}", app.context);
  app.nodes.get('computeProvider').value = 'colab';
  await app.nodes.get('computeProvider').listeners.change();
  assert.match(app.nodes.get('colabStatus').textContent, /页面操作/);
  assert.match(app.nodes.get('colabResourceAvailability').textContent, /待确认/);
  assert.equal(calls.some(url => /resume|launch|autodl|create/.test(url)), false);
});

test('unprepared or finished project selection does not request Colab compute', async () => {
  const app = ui(async () => { throw new Error('No request expected'); });
  app.nodes.get('computeProvider').value = 'colab';
  for (const status of ['not_prepared', 'completed', 'failed', 'cancelled', 'blocked']) {
    vm.runInContext(`state.current = {id: 'speaker'}; state.colab = {prepared: ${status !== 'not_prepared'}, status: '${status}'}`, app.context);
    await app.nodes.get('computeProvider').listeners.change();
  }
});

test('a delayed selection check cannot replace a newly selected project status', async () => {
  let resolveCheck;
  const app = ui(() => new Promise(resolve => { resolveCheck = resolve; }));
  vm.runInContext("state.current = {id: 'speaker'}; state.colab = {prepared: true, status: 'queued'}", app.context);
  app.nodes.get('computeProvider').value = 'colab';
  const request = app.nodes.get('computeProvider').listeners.change();
  vm.runInContext("state.current = {id: 'other'}; state.colab = {prepared: false, status: 'not_prepared'}", app.context);
  resolveCheck(response({job: {prepared: true, status: 'running'}}));
  await request;
  assert.equal(vm.runInContext('state.colab.status', app.context), 'not_prepared');
});

const monitorNow = Date.parse('2026-10-09T05:00:00Z');
const monitorSnapshot = (extra = {}) => ({
  project_id: 'speaker', agent_state: 'watching', heartbeat_at: '2026-10-09T05:00:00Z',
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
    if (url.includes('/static/')) return response(monitorSnapshot({heartbeat_at: new Date().toISOString(), availability: 'unknown', agent_state: 'quota_wait', last_result: 'quota'}));
    return response({prepared: true, status: 'waiting_for_gpu', worker_running: false});
  });
  await vm.runInContext("state.current = {id: 'speaker'}; loadColab()", app.context);
  assert.equal(calls.length, 2);
  assert.deepEqual(calls.find(call => call.url.includes('/static/')), {url: '/static/colab-resource-status/speaker.json', cache: 'no-store'});
  assert.match(app.nodes.get('colabResourceAvailability').textContent, /待确认/);
  assert.match(app.nodes.get('colabResourceLastResult').textContent, /额度限制/);
});

test('a slow resource monitor response cannot overwrite a newly selected project', async () => {
  let resolveStatus;
  const app = ui(() => new Promise(resolve => { resolveStatus = resolve; }));
  const load = vm.runInContext("state.current = {id: 'speaker'}; loadColabResourceAgent('speaker')", app.context);
  vm.runInContext("state.current = {id: 'other'}; state.colabResourceAgent = {project_id: 'other', agent_state: 'stopped'}", app.context);
  resolveStatus(response(monitorSnapshot()));
  await load;
  assert.equal(vm.runInContext('state.colabResourceAgent.project_id', app.context), 'other');
});

test('missing monitor status keeps the saved retry deadline without inventing availability', async () => {
  const app = ui(async () => response({}, 404));
  await vm.runInContext("state.current = {id: 'speaker'}; state.colab = {status: 'waiting_for_gpu', next_attempt_at: '2026-10-09T05:10:00Z'}; loadColabResourceAgent('speaker')", app.context);
  assert.match(app.nodes.get('colabResourceAvailability').textContent, /待确认/);
  assert.match(app.nodes.get('colabResourceAgentState').textContent, /尚未启动/);
  assert.notEqual(app.nodes.get('colabResourceNextCheck').textContent, '—');
});

test('job API failure suppresses a fresh allocation claimed by the separate monitor file', async () => {
  const app = ui(async url => {
    if (url.includes('/static/')) return response(monitorSnapshot({heartbeat_at: new Date().toISOString()}));
    return response({detail: 'status unavailable'}, 503);
  });
  await vm.runInContext("state.current = {id: 'speaker'}; loadColab()", app.context);
  assert.match(app.nodes.get('colabResourceAvailability').textContent, /待确认/);
  assert.match(app.nodes.get('colabDetails').textContent, /unavailable/);
});
