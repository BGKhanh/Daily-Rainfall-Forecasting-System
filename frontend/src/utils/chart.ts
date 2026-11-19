/** Shared SVG chart utility helpers. */

export function computeTicksMultiplesOf5(maxVal: number): number[] {
  const ceil5 = Math.max(15, Math.ceil(maxVal / 5) * 5);
  const step = ceil5 <= 25 ? 5 : Math.ceil(ceil5 / 5 / 5) * 5;
  const ticks: number[] = [];
  for (let v = 0; v <= ceil5; v += step) {
    ticks.push(v);
  }
  if (ticks[ticks.length - 1] < ceil5) {
    ticks.push(ceil5);
  }
  return ticks;
}
