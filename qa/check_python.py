"""Runtime-independent classroom checks; NumPy only, no Pinocchio install needed."""
import importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('kernel',ROOT/'dist/python/kernel.py')
k=importlib.util.module_from_spec(spec);spec.loader.exec_module(k)
model=json.loads((ROOT/'dist/assets/robot/kinematics.json').read_text())
refs=json.loads((ROOT/'qa/reference.json').read_text())
fk=k.make_kinematics(model)
for sample in refs['samples']:
 p,R,J=fk(sample['q'])
 for actual,key in [(p,'p'),(R,'R'),(J,'J')]:np.testing.assert_allclose(actual,sample[key],atol=1e-12)
 source=(ROOT/'dist/python/dls.py').read_text()
for sample in refs['samples'][:4]:
 q=np.array(sample['q']);p,R,_=fk(q+np.deg2rad([4,-4,5,3,-4,3]));target={'p':p.tolist(),'R':R.tolist()}
 for mode in ['pose','position']:
  config={'start':q.tolist(),'target':target,'mode':mode,'damping':.03}
  result=k.run_student(source,config,model)
  assert result['converged'],(sample['name'],mode,result['error'],result['angle'])
  assert result['updates']<50
  task=k.make_task(fk,mode);e,J=task(q,target);h=1e-6
  fd=np.column_stack([(task(q-np.eye(6)[j]*h,target)[0]-task(q+np.eye(6)[j]*h,target)[0])/(2*h) for j in range(6)])
  np.testing.assert_allclose(J,fd,atol=2e-8)
q=np.array(refs['samples'][0]['q']);p,R,_=fk(q+np.deg2rad([4,-4,5,3,-4,3]))
config={'start':q.tolist(),'target':{'p':p.tolist(),'R':R.tolist()},'mode':'pose','damping':.03}
for name in ['pseudoinverse','transpose']:
 result=k.run_student((ROOT/f'dist/python/{name}.py').read_text(),config,model)
 assert result['converged'],(name,result['error'],result['angle'])
for body in ['return np.full(6,np.nan)','return np.full(6,100.)','return np.zeros(7)']:
 try:k.run_student('def solve_ik(q,target):\n    '+body,config,model)
 except ValueError:pass
 else:raise AssertionError(body)
far={**config,'target':{'p':[.95,.95,.9],'R':R.tolist()}}
assert not k.run_student(source,far,model)['converged']
# A returned start is measured, not assumed successful.
assert not k.run_student('def solve_ik(q,target):\n    return q',config,model)['converged']
print('PASS: native reference FK/J, DLS in both tasks at four presets, task finite differences, other templates, invalid output and unreachable target.')
