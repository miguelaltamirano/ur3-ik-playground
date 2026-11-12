

export class PythonRunner {
  constructor(model, onStatus, onOutput) {
    this.model = model;
    this.onStatus = onStatus; this.onOutput = onOutput;
    this.worker = null; this.ready = false; this.pending = null; this.nextId = 0;
  }
  run(source, config) {
    if (this.pending) return Promise.reject(new Error('A run is already active.'));
    return new Promise((resolve, reject) => {
      this.pending = {id: ++this.nextId, source, config: structuredClone(config), resolve, reject};
      if (this.ready) { this.submit(); return; }
      this.onStatus('Loading Python and NumPy. The first run needs an internet connection…');
      this.worker = new Worker(new URL('./python-worker.js', import.meta.url), {type: 'module'});
      this.worker.onmessage = ({data}) => this.receive(data);
      this.worker.onerror = e => { e.preventDefault(); this.fail('Python could not start. Check your connection and try Run code again.'); };
      this.timer = setTimeout(() => this.fail('Python loading timed out. Check your connection, then try again.'), 90000);
      this.worker.postMessage({type: 'init', model: this.model});
    });
  }
  submit() {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.fail('Stopped after 10 seconds. Check for an infinite loop or reduce the computation.'), 10000);
    this.worker.postMessage({type: 'run', ...this.pending, resolve: undefined, reject: undefined});
  }
  receive(data) {
    if (!this.pending) return;
    if (data.type === 'progress') { this.onStatus(data.text); return; }
    if (data.type === 'ready') { this.ready = true; this.submit(); return; }
    if (data.id !== null && data.id !== this.pending.id) return;
    if (data.type === 'started') this.onStatus('Running your Python solver…');
    if (data.type === 'output') this.onOutput(data.text);
    if (data.type === 'error') {
      // A bad student program can be corrected and run again in the same runtime.
      const pending = this.pending; this.pending = null; clearTimeout(this.timer);
      if (!this.ready) { this.worker.terminate(); this.worker = null; }
      pending.reject(new Error(data.message));
    }
    if (data.type === 'result') {
      const pending = this.pending; this.pending = null; clearTimeout(this.timer);
      pending.resolve(data.result);
    }
  }
  fail(message) {
    clearTimeout(this.timer);
    this.worker?.terminate(); this.worker = null; this.ready = false;
    const pending = this.pending; this.pending = null;
    pending?.reject(new Error(message));
  }
  stop() { this.fail('Stopped. Your code is kept; Run code starts a fresh Python runtime.'); }
}
