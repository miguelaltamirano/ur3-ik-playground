// Python runs off the UI thread. Stopping terminates this disposable worker;
// it does not need SharedArrayBuffer or special GitHub Pages headers.
const VERSION = '0.29.3';
const CDN = `https://cdn.jsdelivr.net/pyodide/v${VERSION}/full/`;
let runtime, kernel, model, outputSize = 0, currentId = null;
const send = (type, detail = {}) => self.postMessage({type, id: currentId, ...detail});
function output(text) {
  if (outputSize >= 16000) return;
  const remaining = 16000 - outputSize;
  const chunk = String(text).slice(0, remaining);
  outputSize += chunk.length + 1;
  send('output', {text: chunk + (outputSize >= 16000 ? '\n[Output truncated at 16 KB.]' : '')});
}
async function initialise() {
  send('progress', {text: 'Loading Python in your browser…'});
  const {loadPyodide} = await import(CDN + 'pyodide.mjs');
  runtime = await loadPyodide({indexURL: CDN, stdout: output, stderr: output});
  send('progress', {text: 'Loading NumPy…'});
  await runtime.loadPackage('numpy');
  const response = await fetch(new URL('./python/kernel.py', import.meta.url));
  if (!response.ok) throw new Error('Could not load the UR3 Python helpers.');
  kernel = await response.text();
  runtime.runPython(kernel);
  send('ready', {version: runtime.version});
}
self.onmessage = async ({data}) => {
  try {
    if (data.type === 'init') { model = data.model; await initialise(); return; }
    if (data.type !== 'run' || !runtime) return;
    currentId = data.id; outputSize = 0;
    send('started');
    runtime.globals.set('_student_source', data.source);
    runtime.globals.set('_config_json', JSON.stringify(data.config));
    runtime.globals.set('_model_json', JSON.stringify(model));
    // Refresh helper definitions between runs; student code uses a separate namespace.
    runtime.runPython(kernel);
    const result = await runtime.runPythonAsync('json.dumps(run_student(_student_source, json.loads(_config_json), json.loads(_model_json)), allow_nan=False)');
    send('result', {result: JSON.parse(result)});
  } catch (error) { send('error', {message: String(error.message || error).slice(-12000)}); }
};
