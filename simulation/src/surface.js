// Smooth only coincident STL vertices. A coarse spatial bucket can blend
// opposite sides of thin G1 shells and create dark patches in reflections.
export function smoothSurfaceNormals(geometry,creaseAngle=Math.PI/3) {
  const result = geometry.index ? geometry.toNonIndexed() : geometry;
  const positions = result.attributes.position;
  const normals = result.attributes.normal.clone();
  const faces = new Float32Array(positions.count);
  const groups = new Map();
  const keys = new Array(positions.count);
  const key = i => `${Math.round(positions.getX(i)*1e5)},${Math.round(positions.getY(i)*1e5)},${Math.round(positions.getZ(i)*1e5)}`;
  for(let i=0;i<positions.count;i+=3) {
    const ax=positions.getX(i),ay=positions.getY(i),az=positions.getZ(i);
    const ux=positions.getX(i+1)-ax,uy=positions.getY(i+1)-ay,uz=positions.getZ(i+1)-az;
    const vx=positions.getX(i+2)-ax,vy=positions.getY(i+2)-ay,vz=positions.getZ(i+2)-az;
    let nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
    const length=Math.hypot(nx,ny,nz)||1;nx/=length;ny/=length;nz/=length;
    faces[i]=nx;faces[i+1]=ny;faces[i+2]=nz;
    for(let j=i;j<i+3;j++) {
      const hash=key(j);keys[j]=hash;
      if(!groups.has(hash))groups.set(hash,[]);
      groups.get(hash).push(i);
    }
  }
  const threshold=Math.cos(creaseAngle);
  for(let i=0;i<positions.count;i++) {
    const face=i-i%3;let nx=0,ny=0,nz=0;
    for(const other of groups.get(keys[i])) {
      const dot=faces[face]*faces[other]+faces[face+1]*faces[other+1]+faces[face+2]*faces[other+2];
      if(dot>threshold){nx+=faces[other];ny+=faces[other+1];nz+=faces[other+2];}
    }
    const length=Math.hypot(nx,ny,nz)||1;
    normals.setXYZ(i,nx/length,ny/length,nz/length);
  }
  result.setAttribute('normal',normals);
  return result;
}
