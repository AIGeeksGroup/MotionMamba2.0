// Presentation only: never changes policy, physics, source frames or controls.
export class PlayoutClock {
  constructor(delayMs = 40) { this.delayMs = delayMs; this.reset(); }
  reset() {
    this.offset = null; this.latest = null; this.arrival = null;
    this.presented = null; this.gaps = []; this.bufferMs = this.delayMs;
  }
  observe(seconds, arrival) {
    if (!Number.isFinite(seconds)) return;
    if (this.latest != null && seconds < this.latest - 0.1) this.reset();
    if (this.latest != null && seconds <= this.latest) return;
    const gap = this.arrival == null ? 0 : arrival - this.arrival;
    const observed = arrival - seconds * 1000;
    if (this.offset == null || gap > 500) {
      // Resume after a pause without dragging the old wall/simulation offset.
      this.offset = observed; this.gaps.length = 0;
    } else {
      this.offset += (observed - this.offset) * 0.08;
      this.gaps.push(gap);
      if (this.gaps.length > 80) this.gaps.shift();
    }
    if (this.gaps.length >= 10) {
      const sorted = [...this.gaps].sort((a,b)=>a-b);
      const jitterBudget = sorted[Math.floor((sorted.length-1)*0.95)] * 1.5;
      this.bufferMs = Math.max(this.delayMs, Math.min(120, jitterBudget));
    }
    this.latest = seconds; this.arrival = arrival;
  }
  sample(now) {
    if (this.latest == null) return null;
    const target = Math.min(this.latest, (now - this.offset - this.bufferMs) / 1000);
    // Changing jitter estimates must never replay an older rendered pose.
    this.presented = Math.max(this.presented ?? target, target);
    return this.presented;
  }
}
