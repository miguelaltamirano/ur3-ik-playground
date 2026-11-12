"""Local browser QA through Chrome DevTools, using an isolated Chrome profile."""
import base64,json,time,urllib.request
from pathlib import Path
import websocket
ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:8771'
targets=json.load(urllib.request.urlopen('http://127.0.0.1:9340/json'))
target=next(x for x in targets if x['type']=='page')
ws=websocket.create_connection(target['webSocketDebuggerUrl'],origin='http://localhost:9340',timeout=60)
next_id=0; errors=[]
def call(method,params=None):
 global next_id
 next_id+=1;ident=next_id;ws.send(json.dumps({'id':ident,'method':method,'params':params or {}}))
 while True:
  msg=json.loads(ws.recv())
  if msg.get('method')=='Runtime.exceptionThrown':errors.append(msg['params'])
  if msg.get('id')==ident:
   if 'error' in msg:raise RuntimeError(msg['error'])
   return msg.get('result',{})
def js(expression):
 r=call('Runtime.evaluate',{'expression':expression,'returnByValue':True,'awaitPromise':True,'userGesture':True})
 if 'exceptionDetails' in r:raise RuntimeError(r['exceptionDetails'])
 return r.get('result',{}).get('value')
def wait(expression,timeout=90):
 until=time.monotonic()+timeout
 while time.monotonic()<until:
  if js(expression):return
  time.sleep(.1)
 raise AssertionError('Timed out: '+expression)
def shot(name):
 data=call('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})['data']
 (ROOT/'qa'/name).write_bytes(base64.b64decode(data))

call('Page.enable');call('Runtime.enable')
call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1100,'deviceScaleFactor':1,'mobile':False})
call('Page.navigate',{'url':BASE})
wait("document.querySelector('#code-editor') && !document.querySelector('#code-editor').disabled")
wait("document.querySelector('#robot-canvas').dataset.ready==='true'")
assert js('document.documentElement.scrollWidth<=innerWidth')
assert '2F-85' in js("document.querySelector('.stage-caption').textContent")
assert js("fetch('assets/robot/robot.json',{cache:'no-cache'}).then(r=>r.json()).then(m=>m.gripper.model)")=='Robotiq 2F-85'

def set_code(code):
 js('document.querySelector("#code-editor").value='+json.dumps(code)+';document.querySelector("#code-editor").dispatchEvent(new Event("input"))')
def run():
 js("document.querySelector('#run-code').click()")
 wait("!document.querySelector('#run-code').disabled",120)
def reached():return js("document.querySelector('#solver-status').textContent")=='Target reached'
def preset(name):js("document.querySelector('#pose-preset').value="+json.dumps(name)+";document.querySelector('#pose-preset').dispatchEvent(new Event('change'))")
def example(name):
 js("document.querySelector('#example-select').value="+json.dumps(name)+";document.querySelector('#load-example').click()")
 if js("document.querySelector('#replace-dialog').open"):js("document.querySelector('#confirm-replace').click()")
example('dls');default=js("document.querySelector('#code-editor').value")
# Numeric targets need not fall on an HTML spinbutton's step grid.
js("document.querySelector('#target-x').value=-.27341;document.querySelector('#target-x').dispatchEvent(new Event('change'))")
assert js("document.querySelector('#target-x').value")=='-0.27341'
preset('comfortable')

shot('desktop.png')
run();assert reached(),js("document.querySelector('#python-output').textContent")
assert js("document.querySelector('#iteration-value').textContent")=='3'
assert float(js("document.querySelector('#position-error').textContent"))<.1
assert float(js("document.querySelector('#orientation-error').textContent"))<.1
print('PASS: browser NumPy DLS reaches full pose in 3 updates.',flush=True)
js("document.querySelector('#playback').value=0;document.querySelector('#playback').dispatchEvent(new Event('input'))")
assert js("document.querySelector('#iteration-value').textContent")=='0'
js("document.querySelector('#step-motion').click()")
assert js("document.querySelector('#iteration-value').textContent")=='1'
js("document.querySelector('#play-motion').click()")
wait("document.querySelector('#iteration-value').textContent==='3'")
preset('wrist');assert js("document.querySelector('#rank-value').textContent")=='Rank 5 / 6'
assert js("document.querySelector('#position-rank').textContent")=='3 / 3'
assert js("document.querySelector('#pinv-gain').textContent")=='Discarded'
js("document.querySelector('#show-ellipsoid').click();document.querySelector('#show-frames').click();document.querySelector('#singularities').scrollIntoView()")
time.sleep(.3);shot('singularities.png')
js("document.querySelector('#task-mode').value='position';document.querySelector('#task-mode').dispatchEvent(new Event('change'))")
assert js("document.querySelector('#rank-value').textContent")=='Rank 3 / 3'
assert js("document.querySelector('#target-roll').disabled")
run();assert reached()
js("document.querySelector('#task-mode').value='pose';document.querySelector('#task-mode').dispatchEvent(new Event('change'))")
preset('wrist_near')
before=js("document.querySelector('#dls-gain').textContent")
js("document.querySelector('#damping').value=.1;document.querySelector('#damping').dispatchEvent(new Event('input'))")
assert before!=js("document.querySelector('#dls-gain').textContent")
run();assert reached()
js("document.querySelector('#damping').value=.03;document.querySelector('#damping').dispatchEvent(new Event('input'))")
print('PASS: replay, wrist rank loss, position rank, damping gain and singular starts.',flush=True)
preset('comfortable')
set_code('def solve_ik(q, target)\n    return q');run()
assert 'SyntaxError' in js("document.querySelector('#python-output').textContent")
set_code('def solve_ik(q, target):\n    return np.full(6,np.nan)');run()
assert 'finite' in js("document.querySelector('#python-output').textContent")
set_code('def solve_ik(q, target):\n    while True:\n        pass')
js("document.querySelector('#run-code').click()")
wait("document.querySelector('#runtime-status').textContent==='Running your Python solver…'")
js("document.querySelector('#stop-code').click()")
wait("!document.querySelector('#run-code').disabled")
assert js("document.querySelector('#solver-status').textContent")=='Stopped'
set_code(default);run();assert reached()
for name in ['pseudoinverse','transpose']:
 example(name);run();assert reached(),name
example('dls');js("document.querySelector('#far-target').click()");run();assert not reached()
print('PASS: syntax/NaN errors, infinite-loop Stop, restart, all solver examples, unreachable target.',flush=True)
preset('comfortable')
set_code(default+'\n# My UR3 classroom draft\n')
call('Page.reload');wait("document.querySelector('#code-editor') && !document.querySelector('#code-editor').disabled")
assert 'My UR3 classroom draft' in js("document.querySelector('#code-editor').value")
call('Browser.setDownloadBehavior',{'behavior':'deny'})
js("window.__createURL=URL.createObjectURL;URL.createObjectURL=function(blob){window.__downloadBlob=blob;return window.__createURL(blob)};document.querySelector('#download-code').click()")
wait('!!window.__downloadBlob')
export=js('window.__downloadBlob.text()');compile(export,'exported_student_solver.py','exec');exec(export,{'__name__':'__main__'})
call('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':1,'mobile':True})
js("window.scrollTo(0,0)");time.sleep(.3)
assert js('document.documentElement.scrollWidth<=innerWidth')
shot('mobile-robot.png')
js("document.querySelector('#editor-title').scrollIntoView()");time.sleep(.3);shot('mobile-editor.png')
assert not errors,errors
print('PASS: saved draft, self-contained Python export, mobile layout, no unhandled exceptions.',flush=True)
ws.close()
