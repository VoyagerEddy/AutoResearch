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
    addEventListener() {},
  }]));
  let confirmations = 0;
  const context = vm.createContext({
    document: {
      getElementById(id) { assert.ok(nodes.has(id), `Unknown DOM id: ${id}`); return nodes.get(id); },
      querySelectorAll() { return []; },
    },
    fetch, AbortController,
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
