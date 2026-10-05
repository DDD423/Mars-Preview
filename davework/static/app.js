'use strict';
const $ = id => document.getElementById(id);
const fragment = new URLSearchParams(location.hash.slice(1));
if (fragment.get('token')) { sessionStorage.setItem('dave-token', fragment.get('token')); history.replaceState(null, '', '/'); }
const token = sessionStorage.getItem('dave-token') || '';
let tools = [], config = {}, context = null, preview = null, pendingPlan = null, selections = {}, task = null, seq = 0, polling = false, editRevision = 0;
const json = v => JSON.stringify(v, null, 2);
async function api(path, body) {
  const response = await fetch('/api/' + path, {method: body === undefined ? 'GET' : 'POST', headers: {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}, ...(body === undefined ? {} : {body: JSON.stringify(body)})});
  const value = await response.json();
  if (!response.ok) { const e = new Error(value.error?.message || '请求失败'); e.info = value.error; throw e; }
  return value;
}
let toastTimer;
function toast(message) { $('toast').textContent = message; $('toast').classList.remove('hidden'); clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').classList.add('hidden'), 8000); }
function fail(e) { toast((e.info?.code ? e.info.code + ' · ' : '') + e.message + (e.info?.details ? '\n' + json(e.info.details) : '')); }
function tab(name) { document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === name)); for (const key of ['request', 'tools', 'json']) $(key + 'Pane').classList.toggle('hidden', key !== name); }
document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => tab(b.dataset.tab)));
function invalidate() { editRevision++; preview = null; $('previewBox').classList.add('hidden'); $('selectionBox').classList.add('hidden'); }
function option(select, value, label) { const o = document.createElement('option'); o.value = value; o.textContent = label; select.append(o); }
function setConfig(value) {
  config = value.config;
  for (const key of ['workspace', 'checkpoint', 'device', 'translator_checkpoint', 'translator_device']) $(key).value = config[key];
  $('timeout').value = config.terminal_timeout; $('autoScroll').checked = config.auto_scroll; $('excludes').value = config.excludes.join(', ');
  $('recentWorkspaces').replaceChildren(); option($('recentWorkspaces'), '', '选择工作区'); for (const p of config.recent_workspaces) option($('recentWorkspaces'), p, p);
  $('toolSwitches').replaceChildren();
  for (const spec of tools) { const label = document.createElement('label'); label.className = 'check'; const input = document.createElement('input'); input.type = 'checkbox'; input.value = spec.name; input.checked = config.enabled_tools.includes(spec.name); label.append(input, document.createTextNode(spec.label)); $('toolSwitches').append(label); }
  filterStatus(value.filter);
  $('mainModel').textContent = '主模型 · ' + value.main_model;
  $('translatorStatus').textContent = 'translator1.0 · ' + (value.translator?.state === 'error' ? '加载失败' : (value.translator?.message || '尚未加载')) + (value.translator?.device ? ' · ' + value.translator.device.toUpperCase() : '');
  $('translatorStatus').title = value.translator?.message || '';
}
function filterStatus(status) { $('filterStatus').textContent = 'filter1.0 · ' + (status?.message || '尚未加载') + (status?.device ? ' · ' + status.device.toUpperCase() : ''); $('filterStatus').title = status?.message || ''; }
$('recentWorkspaces').addEventListener('change', e => { if (e.target.value) $('workspace').value = e.target.value; });
$('configForm').addEventListener('submit', async e => { e.preventDefault(); try { const value = await api('config', {workspace: $('workspace').value, checkpoint: $('checkpoint').value, device: $('device').value, translator_checkpoint: $('translator_checkpoint').value, translator_device: $('translator_device').value, terminal_timeout: Number($('timeout').value), auto_scroll: $('autoScroll').checked, excludes: $('excludes').value.split(',').map(s => s.trim()).filter(Boolean), enabled_tools: Array.from($('toolSwitches').querySelectorAll('input:checked')).map(i => i.value)}); setConfig(value); $('configMessage').textContent = '已保存。配置将在新请求中生效；已有参数上下文保持原工作区。'; toast('配置已保存'); } catch (e) { fail(e); } });
function showContext(value) {
  context = value; invalidate();
  const f = value.filter; $('extraction').classList.remove('hidden'); $('extractState').textContent = f.status;
  $('contextMeta').textContent = value.context_id + ' · ' + value.workspace; $('original').textContent = f.original;
  $('normalized').replaceChildren();
  // Structured rendering avoids literal {1} collisions and UTF-16 slicing entirely.
  for (const segment of f.segments) { if (segment.kind === 'literal') $('normalized').append(document.createTextNode(segment.text)); else { const span = document.createElement('span'); span.className = 'slot'; span.textContent = segment.id; $('normalized').append(span); } }
  $('bindings').replaceChildren();
  for (const [key, binding] of Object.entries(f.bindings)) { const row = document.createElement('tr'); for (const value of [key, binding.type, binding.value, '[' + binding.start + ', ' + binding.end + ')']) { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); } $('bindings').append(row); }
  if (!Object.keys(f.bindings).length) { const row = document.createElement('tr'); const cell = document.createElement('td'); cell.colSpan = 4; cell.textContent = '没有提取到参数'; row.append(cell); $('bindings').append(row); }
  $('filterReason').textContent = f.reason || ''; $('plannerMessage').textContent = value.planner.kind === 'plan' ? 'translator1.0 已生成计划。请核对下方绑定参数与动作，再点击执行。' : value.planner.message; $('toolContext').textContent = '上下文 · ' + value.workspace; renderFields();
}
$('requestText').addEventListener('input', invalidate);
$('requestForm').addEventListener('submit', async e => { e.preventDefault(); invalidate(); $('extractButton').disabled = true; filterStatus({message: '正在提取与规划'}); try { const value = await api('context', {text: $('requestText').value, extract: true}); showContext(value); $('jsonEditor').value = json(value.planner); setConfig(await api('config')); if (value.planner.kind === 'plan') await previewPlan(value.planner); } catch (e) { fail(e); } finally { $('extractButton').disabled = false; } });
$('useTools').addEventListener('click', () => tab('tools'));
async function emptyContext() { showContext(await api('context', {text: '', extract: false})); }
$('newContext').addEventListener('click', async () => { try { await emptyContext(); fillExample(); toast('已创建独立空上下文'); } catch (e) { fail(e); } });
function renderFields() {
  invalidate(); const spec = tools.find(t => t.name === $('toolSelect').value); if (!spec) return;
  $('toolFields').replaceChildren();
  for (const [key, p] of Object.entries(spec.params)) {
    const label = document.createElement('label'); label.textContent = p.label + (p.required ? ' *' : ''); label.htmlFor = 'arg_' + key;
    const row = document.createElement('div'); row.className = 'field-row';
    let field;
    if (p.type === 'boolean' || p.choices) { field = document.createElement('select'); if (p.type === 'boolean') { option(field, 'false', '否'); option(field, 'true', '是'); } else for (const choice of p.choices) option(field, choice, choice); }
    else if (key === 'text' || key === 'script' || p.type === 'array') { field = document.createElement('textarea'); field.rows = key === 'script' ? 5 : 3; }
    else { field = document.createElement('input'); if (p.type === 'integer') { field.type = 'number'; field.min = 1; field.max = 3600; field.placeholder = '使用配置默认值'; } }
    field.id = 'arg_' + key; field.dataset.key = key; field.dataset.kind = p.type;
    field.value = p.default == null ? '' : p.type === 'array' ? json(p.default) : String(p.default);
    const mode = document.createElement('select'); mode.id = 'mode_' + key; option(mode, '', '直接填写');
    if (['string','source','target','path','name'].includes(p.type) && key !== 'script') for (const [slot, binding] of Object.entries(context?.filter.bindings || {})) option(mode, slot.slice(1,-1), slot + ' · ' + binding.type);
    mode.addEventListener('change', () => { field.disabled = Boolean(mode.value); invalidate(); }); field.addEventListener('input', invalidate); field.addEventListener('change', invalidate);
    row.append(field, mode); $('toolFields').append(label, row);
  }
}
$('toolSelect').addEventListener('change', renderFields);
$('toolForm').addEventListener('submit', async e => { e.preventDefault(); try { const spec = tools.find(t => t.name === $('toolSelect').value); const args = {}; for (const [key, p] of Object.entries(spec.params)) { const mode = $('mode_' + key).value; const text = $('arg_' + key).value; if (mode) args[key] = {slot: Number(mode)}; else if (p.type === 'array') args[key] = JSON.parse(text || '[]'); else if (p.type === 'boolean') args[key] = text === 'true'; else if (p.type === 'integer') args[key] = text ? Number(text) : null; else args[key] = text; } if (!context) await emptyContext(); const plan = {protocol: 'davework/1', kind: 'plan', context_id: context.context_id, steps: [{id: 'step_1', tool: spec.name, args}]}; $('jsonEditor').value = json(plan); await previewPlan(plan); } catch (e) { fail(e); } });
function fillExample() { if (!context) return; $('jsonEditor').value = json({protocol:'davework/1',kind:'plan',context_id:context.context_id,steps:[{id:'list_1',tool:'fs.list',args:{path:'.'}}]}); invalidate(); }
$('jsonExample').addEventListener('click', async () => { try { if (!context) await emptyContext(); fillExample(); } catch (e) { fail(e); } });
$('jsonEditor').addEventListener('input', invalidate);
$('jsonPreview').addEventListener('click', async () => { try { await previewPlan(JSON.parse($('jsonEditor').value)); } catch (e) { fail(e); } });
async function previewPlan(plan, keepSelections = false) {
  invalidate(); const revision = editRevision; pendingPlan = plan; if (!keepSelections) selections = {};
  try { const result = await api('preview', {plan, selections}); if (revision !== editRevision) return; preview = result; $('previewWorkspace').textContent = '工作区 · ' + preview.workspace; $('previewContent').textContent = json({kind:preview.kind,steps:preview.steps,message:preview.message}); $('previewBox').classList.remove('hidden'); $('executeButton').disabled = false; }
  catch (e) { if (revision !== editRevision) return; if (e.info?.code === 'AMBIGUOUS_NAME') { const d = e.info.details; $('pathSelection').replaceChildren(); for (const p of d.candidates) option($('pathSelection'), p, p); $('selectionBox').dataset.key = d.selection_key; $('selectionBox').classList.remove('hidden'); } else throw e; }
}
$('confirmSelection').addEventListener('click', async () => { try { selections[$('selectionBox').dataset.key] = $('pathSelection').value; await previewPlan(pendingPlan, true); } catch (e) { fail(e); } });
$('executeButton').addEventListener('click', async () => { if (!preview) return; $('executeButton').disabled = true; try { const result = await api('execute', {preview_id:preview.preview_id,digest:preview.digest}); preview = null; followTask(result.task_id); } catch (e) { fail(e); $('executeButton').disabled = false; } });
function log(event) {
  $('outputLog').querySelector('.empty')?.remove(); const item = document.createElement('div'); item.className = 'log-entry' + (event.kind === 'error' ? ' error' : '');
  const head = document.createElement('div'); head.className = 'log-head'; const names = {task_started:'任务开始',step_started:'执行步骤',step_completed:'步骤完成',step_skipped:'跳过步骤',task_finished:'任务结束',error:'错误',stdout:'stdout',stderr:'stderr',output_limit:'输出已截断',noop:'不执行',clarification:'需要澄清'};
  head.textContent = '#' + event.seq + ' · ' + (names[event.kind] || event.kind) + (event.data.step ? ' · ' + event.data.step : '');
  const content = document.createElement('pre');
  if (event.kind === 'step_completed' && typeof event.data.result?.text === 'string') { const r = event.data.result; content.textContent = (r.path || '') + '\n\n' + r.text + (r.truncated ? '\n[内容达到 1 MiB，已截断]' : ''); }
  else content.textContent = event.data.text ?? json(event.data);
  item.append(head, content); $('outputLog').append(item);
  if (config.auto_scroll) $('outputLog').scrollTop = $('outputLog').scrollHeight;
}
function followTask(id) { task = id; seq = 0; sessionStorage.setItem('dave-task', id); $('stopButton').disabled = false; $('taskState').textContent = id; poll(); }
async function poll() {
  if (polling || !task) return; polling = true; const id = task;
  try { const value = await api('events?task_id=' + encodeURIComponent(id) + '&after=' + seq); if (id !== task) return; for (const event of value.events) { if (event.seq > seq) { log(event); seq = event.seq; } } $('taskState').textContent = value.status + ' · ' + id; const running = value.status === 'running'; $('stopButton').disabled = !running; if (running) setTimeout(poll, 350); else { sessionStorage.removeItem('dave-task'); await refreshTasks(); } }
  catch (e) { $('taskState').textContent = '连接中断，将重试 · ' + id; setTimeout(poll, 1500); }
  finally { polling = false; if (id !== task) setTimeout(poll, 0); }
}
$('stopButton').addEventListener('click', async () => { try { await api('cancel', {task_id:task}); $('taskState').textContent = '正在停止 · ' + task; } catch (e) { fail(e); } });
$('clearOutput').addEventListener('click', () => $('outputLog').replaceChildren());
async function refreshTasks() { const value = await api('tasks'); $('recentTasks').replaceChildren(); for (const t of value.tasks) { const b = document.createElement('button'); b.className = 'task-link'; b.textContent = t.status + ' · ' + new Date(t.created_at*1000).toLocaleTimeString(); b.title = t.workspace + '\n' + t.id; b.addEventListener('click', () => { $('outputLog').replaceChildren(); followTask(t.id); }); $('recentTasks').append(b); } }
$('refreshTasks').addEventListener('click', () => refreshTasks().catch(fail));
(async () => { try { tools = (await api('tools')).tools; for (const spec of tools) option($('toolSelect'), spec.name, spec.label + ' · ' + spec.name); const value = await api('config'); setConfig(value); renderFields(); $('connection').textContent = '本机连接 · davework/1'; await refreshTasks(); const old = sessionStorage.getItem('dave-task') || value.active_task; if (old) followTask(old); } catch (e) { $('connection').textContent = '无法连接'; fail(e); } })();
