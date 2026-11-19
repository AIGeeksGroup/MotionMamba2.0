import * as THREE from 'three';

// Software-WebGL display fallback only. Clustering changes visible STL detail
// by at most sqrt(3)*cell per vertex; policy/physics/reference data stay intact.
export function clusterStlGeometry(geometry, cell) {
  const position=geometry.getAttribute('position');
  if (!position || position.count<12000 || geometry.getAttribute('uv') || geometry.getAttribute('color')) return null;
  const cells=new Map(), remap=new Uint32Array(position.count), sums=[], counts=[];
  for(let i=0;i<position.count;i++) {
    const x=position.getX(i),y=position.getY(i),z=position.getZ(i);
    const key=`${Math.floor(x/cell)},${Math.floor(y/cell)},${Math.floor(z/cell)}`;
    let group=cells.get(key);
    if(group===undefined) { group=counts.length;cells.set(key,group);counts.push(0);sums.push(0,0,0); }
    remap[i]=group;counts[group]++;
    sums[group*3]+=x;sums[group*3+1]+=y;sums[group*3+2]+=z;
  }
  const indices=[],seen=new Set(),source=geometry.index;
  const count=source?source.count:position.count;
  for(let i=0;i<count;i+=3) {
    const a=remap[source?source.getX(i):i],b=remap[source?source.getX(i+1):i+1],c=remap[source?source.getX(i+2):i+2];
    if(a===b || b===c || c===a)continue;
    const key=[a,b,c].sort((x,y)=>x-y).join(',');
    if(seen.has(key))continue;
    seen.add(key);indices.push(a,b,c);
  }
  if(indices.length>=count*.8 || !indices.length)return null;
  for(let i=0;i<counts.length;i++)for(let j=0;j<3;j++)sums[i*3+j]/=counts[i];
  const result=new THREE.BufferGeometry();
  result.setAttribute('position',new THREE.Float32BufferAttribute(sums,3));
  result.setIndex(indices);result.computeVertexNormals();result.computeBoundingBox();result.computeBoundingSphere();
  result.userData.sonicVisualLod={originalTriangles:count/3,triangles:indices.length/3,cell};
  return result;
}
