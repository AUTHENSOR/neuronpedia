import { gzipSync } from 'node:zlib';
import { afterEach, describe, expect, it, vi } from 'vitest';
import fixture from './__fixtures__/neuron-dashboard-qwen3-32b-31-12345.json';
import {
  applyNeuronDashboard,
  fetchNeuronDashboard,
  gunzipJson,
  layerFileStem,
  NEURON_DASHBOARD_BOTTOM_DATA_SOURCE,
  neuronByteRange,
  NeuronDashboardError,
  NeuronDashboardRecord,
  recordToNeuronDashboard,
} from './neuron-dashboards';

const record = fixture.record as NeuronDashboardRecord;
const vocabArray: string[] = [];
Object.entries(fixture.vocab).forEach(([id, token]) => {
  vocabArray[Number(id)] = token;
});
const ids = { modelId: 'qwen3-32b', layer: '31', index: '12345' };

function uint64Offsets(offsets: number[]) {
  const view = new DataView(new ArrayBuffer(8 * offsets.length));
  offsets.forEach((offset, i) => view.setBigUint64(8 * i, BigInt(offset), true));
  return new Uint8Array(view.buffer);
}

describe('neuronByteRange', () => {
  it('reads two little-endian uint64 offsets', () => {
    const bytes = uint64Offsets([396275988, 396309573]);
    expect(neuronByteRange(bytes.buffer)).toEqual([396275988, 396309573]);
  });

  it('refuses a short read and an empty range', () => {
    expect(() => neuronByteRange(new ArrayBuffer(8))).toThrow(NeuronDashboardError);
    expect(() => neuronByteRange(uint64Offsets([5, 5]).buffer)).toThrow(NeuronDashboardError);
  });
});

describe('layerFileStem', () => {
  it('pads to two digits like the Python writer', () => {
    expect(layerFileStem(3)).toBe('neurons/layer_03');
    expect(layerFileStem(31)).toBe('neurons/layer_31');
    expect(layerFileStem(120)).toBe('neurons/layer_120');
  });
});

describe('recordToNeuronDashboard', () => {
  const dashboard = recordToNeuronDashboard(record, vocabArray, ids);

  it('decodes tokens and puts the peak at maxValueTokenIndex', () => {
    const top = dashboard.activations[0];
    const example = record.top[0];
    expect(top.tokens?.join('')).toContain('EHEC infection');
    expect(top.maxValueTokenIndex).toBe(example.pos - example.start);
    expect(top.values?.[top.maxValueTokenIndex ?? -1]).toBeCloseTo(example.act, 1);
    expect(top.maxValue).toBe(example.act);
    expect(top.tokens?.length).toBe(top.values?.length);
  });

  it('gives top examples binMin -1 and each band its interval', () => {
    const tops = dashboard.activations.filter((a) => a.id?.includes('-top-'));
    expect(tops).toHaveLength(record.top.length);
    tops.forEach((a) => {
      expect(a.binMin).toBe(-1);
      expect(a.binMax).toBe(record.stats.max);
    });
    record.bands.forEach((band, b) => {
      const rows = dashboard.activations.filter((a) => a.id?.includes(`-band${b}-`));
      expect(rows).toHaveLength(band.examples.length);
      rows.forEach((a) => {
        expect(a.binMin).toBe(band.lo);
        expect(a.binMax).toBe(band.hi);
        expect(a.binContains).toBeCloseTo(band.q_hi - band.q_lo);
        // A row whose maxValue reaches binMax would be moved to another interval by ActivationsList.
        expect(a.maxValue).toBeLessThan(band.hi);
      });
    });
  });

  it('marks negative-peak examples as bottom and puts the bottom list below every band', () => {
    const bottoms = dashboard.activations.filter((a) => a.id?.includes('-bottom-'));
    expect(bottoms).toHaveLength(record.bottom.length);
    const lowestEdge = Math.min(...record.bands.map((b) => b.lo));
    bottoms.forEach((a) => {
      expect(a.dataSource).toBe(NEURON_DASHBOARD_BOTTOM_DATA_SOURCE);
      expect(a.binMin).toBe(record.stats.min);
      expect(a.binMax).toBe(lowestEdge);
      expect(a.binContains).toBeNull();
      expect(a.minValue).toBeLessThanOrEqual(a.maxValue ?? 0);
    });
    dashboard.activations
      .filter((a) => (a.maxValue ?? 0) < 0)
      .forEach((a) => expect(a.dataSource).toBe(NEURON_DASHBOARD_BOTTOM_DATA_SOURCE));
  });

  it('fills the Neuron stats fields', () => {
    expect(dashboard.neuron.maxActApprox).toBe(record.stats.max);
    expect(dashboard.neuron.pos_str).toEqual(record.logits.pos.map((l) => l[1]));
    expect(dashboard.neuron.neg_values).toEqual(record.logits.neg.map((l) => l[2]));
    expect(dashboard.neuron.frac_nonzero).toBeCloseTo(0.4719, 3);
  });

  it('gives every row a unique id', () => {
    const rowIds = dashboard.activations.map((a) => a.id);
    expect(new Set(rowIds).size).toBe(rowIds.length);
  });

  it('refuses an unknown format version', () => {
    expect(() => recordToNeuronDashboard({ ...record, v: 2 }, vocabArray, ids)).toThrow(NeuronDashboardError);
  });

  it('keeps database fields when applied over a neuron', () => {
    const neuron = { modelId: 'qwen3-32b', layer: '31', index: '12345', explanations: [{ id: 'e1' }] };
    const applied = applyNeuronDashboard(neuron as never, dashboard);
    expect(applied.explanations).toEqual([{ id: 'e1' }]);
    expect(applied.activations).toBe(dashboard.activations);
    expect(applied.maxActApprox).toBe(record.stats.max);
  });
});

describe('fetchNeuronDashboard', () => {
  const base = 'https://dashboards.test/v1/qwen3-32b/';
  const other = { ...record, neuron: 12344 };
  const members = [gzipSync(JSON.stringify(other)), gzipSync(JSON.stringify(record))];
  const shard = Buffer.concat(members);
  // A file whose neurons 12344 and 12345 are its two members.
  const offsets = new Array<number>(12345).fill(0);
  offsets.push(members[0].length, shard.length);
  const idx = uint64Offsets(offsets);
  const files: Record<string, Uint8Array> = {
    'https://dashboards.test/v1/qwen3-32b/neurons/layer_31.idx': idx,
    'https://dashboards.test/v1/qwen3-32b/neurons/layer_31.json.gz': shard,
    'https://dashboards.test/v1/qwen3-32b/corpus/vocab.json': new TextEncoder().encode(JSON.stringify(vocabArray)),
  };

  function serve(honorRange: boolean, overrides: Record<string, Uint8Array> = {}) {
    const ranges: string[] = [];
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      const file = overrides[url] ?? files[url];
      const body = file ? Uint8Array.from(file) : undefined;
      if (!body) {
        return new Response('not found', { status: 404 });
      }
      const range = new Headers(init?.headers).get('Range');
      if (!range || !honorRange) {
        return new Response(body, { status: 200 });
      }
      ranges.push(`${url.split('/').pop()} ${range}`);
      const [, a, b] = /bytes=(\d+)-(\d+)/.exec(range) ?? [];
      return new Response(body.slice(Number(a), Number(b) + 1), { status: 206 });
    });
    return ranges;
  }

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('reads one neuron with two Range requests', async () => {
    const ranges = serve(true);
    const dashboard = await fetchNeuronDashboard(base, ids);
    expect(dashboard.record.neuron).toBe(12345);
    expect(dashboard.activations[0].tokens?.join('')).toContain('EHEC');
    expect(ranges).toEqual([
      `layer_31.idx bytes=${8 * 12345}-${8 * 12345 + 15}`,
      `layer_31.json.gz bytes=${members[0].length}-${shard.length - 1}`,
    ]);
  });

  it('still works when the server ignores Range', async () => {
    serve(false);
    const dashboard = await fetchNeuronDashboard(base, ids);
    expect(dashboard.record.neuron).toBe(12345);
  });

  it('reads the neighbouring member for the neighbouring neuron', async () => {
    serve(true);
    const dashboard = await fetchNeuronDashboard(base, { ...ids, index: '12344' });
    expect(dashboard.record.neuron).toBe(12344);
  });

  it('refuses a record for a different neuron', async () => {
    const swapped = Buffer.concat([members[1], members[0]]);
    const swappedIdx = uint64Offsets([...offsets.slice(0, 12345), members[1].length, swapped.length]);
    serve(true, {
      'https://dashboards.test/v1/qwen3-32b/neurons/layer_31.idx': swappedIdx,
      'https://dashboards.test/v1/qwen3-32b/neurons/layer_31.json.gz': swapped,
    });
    await expect(fetchNeuronDashboard(base, ids)).rejects.toThrow(/got 31\/12344/);
  });

  it('refuses ids that are not numbers', async () => {
    serve(true);
    await expect(fetchNeuronDashboard(base, { ...ids, layer: '31-res-jb' })).rejects.toThrow(NeuronDashboardError);
  });
});

describe('gunzipJson', () => {
  it('round-trips a gzip member', async () => {
    const bytes = gzipSync(JSON.stringify({ a: [1, 2] }));
    expect(await gunzipJson(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length))).toEqual({
      a: [1, 2],
    });
  });
});
