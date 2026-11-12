export const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
export const norm=a=>Math.sqrt(dot(a,a));
export const add=(a,b)=>a.map((x,i)=>x+b[i]);
export const sub=(a,b)=>a.map((x,i)=>x-b[i]);
export const scale=(a,s)=>a.map(x=>x*s);
export const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const eye=n=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>+(i===j)));
export const transpose=A=>A[0].map((_,i)=>A.map(row=>row[i]));
export const mv=(A,v)=>A.map(row=>dot(row,v));
export const mm=(A,B)=>A.map(row=>transpose(B).map(col=>dot(row,col)));
export function rotation(axis,t){const [x,y,z]=axis,c=Math.cos(t),s=Math.sin(t),C=1-c;return [[c+x*x*C,x*y*C-z*s,x*z*C+y*s],[y*x*C+z*s,c+y*y*C,y*z*C-x*s],[z*x*C-y*s,z*y*C+x*s,c+z*z*C]];}
export function rpyMatrix(rpy){return mm(mm(rotation([0,0,1],rpy[2]),rotation([0,1,0],rpy[1])),rotation([1,0,0],rpy[0]));}
export function matrixRpy(R){const pitch=Math.asin(Math.max(-1,Math.min(1,-R[2][0])));if(Math.abs(Math.cos(pitch))<1e-8)return [0,pitch,Math.atan2(-R[0][1],R[1][1])];return [Math.atan2(R[2][1],R[2][2]),pitch,Math.atan2(R[1][0],R[0][0])];}
export function rotationLog(R){
  const c=Math.max(-1,Math.min(1,(R[0][0]+R[1][1]+R[2][2]-1)/2)),theta=Math.acos(c);
  const v=[R[2][1]-R[1][2],R[0][2]-R[2][0],R[1][0]-R[0][1]];
  if(theta<1e-7)return scale(v,.5);
  if(Math.PI-theta<1e-5){
    const i=[0,1,2].reduce((best,k)=>R[k][k]>R[best][best]?k:best,0),a=[0,0,0];
    a[i]=Math.sqrt(Math.max(0,(R[i][i]+1)/2));
    for(let j=0;j<3;j++)if(j!==i)a[j]=(R[i][j]+R[j][i])/(4*Math.max(a[i],1e-12));
    if(dot(a,v)<0)for(let j=0;j<3;j++)a[j]*=-1;
    return scale(a,theta/Math.max(norm(a),1e-12));
  }
  return scale(v,theta/(2*Math.sin(theta)));
}
// One-sided Jacobi SVD avoids squaring the condition number in J^T J.
// Returns min(m,n) singular values in descending order and left singular vectors.
export function svd(A){
  const B=A.map(r=>[...r]),m=B.length,n=B[0].length,V=eye(n);
  for(let sweep=0;sweep<100;sweep++){
    let changed=false;
    for(let p=0;p<n-1;p++)for(let q=p+1;q<n;q++){
      let a=0,b=0,g=0;for(let i=0;i<m;i++){a+=B[i][p]**2;b+=B[i][q]**2;g+=B[i][p]*B[i][q];}
      if(a===0||b===0||Math.abs(g)<=1e-14*Math.sqrt(a*b))continue;
      const tau=(b-a)/(2*g),t=(tau>=0?1:-1)/(Math.abs(tau)+Math.hypot(1,tau)),c=1/Math.hypot(1,t),s=c*t;
      for(const M of [B,V])for(const row of M){const x=row[p],y=row[q];row[p]=c*x-s*y;row[q]=s*x+c*y;}
      changed=true;
    }
    if(!changed)break;
  }
  const columns=Array.from({length:n},(_,j)=>{const b=B.map(row=>row[j]),s=norm(b);return {s,u:s>1e-15?scale(b,1/s):Array(m).fill(0),v:V.map(row=>row[j])};}).sort((a,b)=>b.s-a.s).slice(0,Math.min(m,n));
  return {s:columns.map(c=>c.s),U:transpose(columns.map(c=>c.u)),V:transpose(columns.map(c=>c.v))};
}
export const LENGTH=.4;
export function makeKinematics(model){
  return q=>{
    let p=[0,0,0],R=eye(3);const origins=[],axes=[];
    model.joints.forEach((j,i)=>{p=add(p,mv(R,j.xyz));R=mm(R,j.rotation);origins.push([...p]);axes.push(mv(R,j.axis));R=mm(R,rotation(j.axis,q[i]));});
    p=add(p,mv(R,model.tool.xyz));R=mm(R,model.tool.rotation);
    const Jv=transpose(axes.map((axis,i)=>cross(axis,sub(p,origins[i])))),Jw=transpose(axes);
    return {p,R,Jv,Jw,J:[...Jv,...Jw],origins,axes};
  };
}
export function diagnostics(f,mode='pose'){
  const J=mode==='position'?f.Jv.map(r=>scale(r,1/LENGTH)):[...f.Jv.map(r=>scale(r,1/LENGTH)),...f.Jw];
  const decomposition=svd(J),s=decomposition.s,min=s.at(-1),max=s[0],rank=s.filter(v=>v>max*1e-7).length;
  return {...decomposition,rank,condition:min>max*1e-7?max/min:Infinity,ratio:min/max};
}
