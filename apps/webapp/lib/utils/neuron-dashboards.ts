// Neuron dashboards stored as files, not database rows. A source set with externalDashboardsUrl serves:
//   neurons/layer_XX.idx      N + 1 byte offsets, little-endian uint64
//   neurons/layer_XX.json.gz  one gzip member per neuron, at bytes offsets[n] .. offsets[n+1]-1
//   corpus/vocab.json         decoded string for each token id
// The browser reads one neuron with two Range requests. utils/neuron-dashboards-worker serves the files.

import { linearHistogram } from '@/lib/utils/neuron-dashboard-histogram';
import { ActivationPartialWithRelations, NeuronWithPartialRelations } from '@/prisma/generated/zod';

export const NEURON_DASHBOARD_FORMAT_VERSION = 1;

// Same value as the private constant in components/activation-item.tsx, which draws these in red.
export const NEURON_DASHBOARD_BOTTOM_DATA_SOURCE = 'bottom';
export const NEURON_DASHBOARD_TOP_DATA_SOURCE = 'top';

export type NeuronDashboardExample = {
  doc: number;
  pos: number;
  act: number;
  gate: number;
  up: number;
  start: number;
  tokens: number[];
  acts: (number | null)[];
  // one char per window token, '1' means the gate is closed (gate <= 0)
  gate_closed: string;
};

export type NeuronDashboardBand = {
  q_lo: number;
  q_hi: number;
  lo: number;
  hi: number;
  examples: NeuronDashboardExample[];
};

export type NeuronDashboardLogit = [tokenId: number, token: string, value: number];

export type NeuronDashboardRecord = {
  v: number;
  model: string;
  layer: number;
  neuron: number;
  stats: {
    max: number;
    min: number;
    mean: number;
    std: number;
    sigma_calib: number;
    frac_gate_open: number;
    echo_abs_share: number;
    n_hist_tokens: number;
  };
  quantiles: { q: number[]; v: number[] };
  hist: {
    sigma: number;
    open_pos: number[];
    open_neg: number[];
    closed_pos: number[];
    closed_neg: number[];
  };
  top: NeuronDashboardExample[];
  bottom: NeuronDashboardExample[];
  echo_top: NeuronDashboardExample[];
  echo_bottom: NeuronDashboardExample[];
  bands: NeuronDashboardBand[];
  logits: { pos: NeuronDashboardLogit[]; neg: NeuronDashboardLogit[] };
};

export type NeuronDashboard = {
  // Fields of the Neuron row that the dashboard components read.
  neuron: Pick<
    NeuronWithPartialRelations,
    | 'maxActApprox'
    | 'pos_str'
    | 'pos_values'
    | 'neg_str'
    | 'neg_values'
    | 'frac_nonzero'
    | 'freq_hist_data_bar_values'
    | 'freq_hist_data_bar_heights'
  >;
  activations: ActivationPartialWithRelations[];
  // The full record, for panels the Neuron row has no fields for (gate and up, echoes, histogram).
  record: NeuronDashboardRecord;
};

export class NeuronDashboardError extends Error {}

export function layerFileStem(layer: number) {
  return `neurons/layer_${String(layer).padStart(2, '0')}`;
}

// Returns [start, end) of neuron n's gzip member from the 16 bytes at 8n .. 8n+15 of the .idx file.
export function neuronByteRange(idxBytes: ArrayBuffer): [number, number] {
  if (idxBytes.byteLength !== 16) {
    throw new NeuronDashboardError(`expected 16 index bytes, got ${idxBytes.byteLength}`);
  }
  const view = new DataView(idxBytes);
  const start = Number(view.getBigUint64(0, true));
  const end = Number(view.getBigUint64(8, true));
  if (!(end > start)) {
    throw new NeuronDashboardError(`bad index range ${start}..${end}`);
  }
  return [start, end];
}

export async function gunzipJson<T>(bytes: ArrayBuffer): Promise<T> {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return (await new Response(stream).json()) as T;
}

function exampleToActivation(
  example: NeuronDashboardExample,
  vocab: string[],
  id: string,
  ids: { modelId: string; layer: string; index: string },
  bin: { binMin: number; binMax: number; binContains: number | null },
): ActivationPartialWithRelations {
  const values = example.acts.map((v) => v ?? 0);
  return {
    id,
    modelId: ids.modelId,
    layer: ids.layer,
    index: ids.index,
    tokens: example.tokens.map((t) => vocab[t] ?? `<${t}>`),
    values,
    // The peak token's value, not the window max: ActivationsList re-bins rows whose maxValue passes binMax.
    maxValue: example.act,
    maxValueTokenIndex: example.pos - example.start,
    // Window values have 3 significant digits and the peak has 4, so the peak can be the lower one.
    minValue: Math.min(...values, example.act),
    dataIndex: String(example.doc),
    dataSource: example.act < 0 ? NEURON_DASHBOARD_BOTTOM_DATA_SOURCE : NEURON_DASHBOARD_TOP_DATA_SOURCE,
    binMin: bin.binMin,
    binMax: bin.binMax,
    binContains: bin.binContains,
    dfaValues: [],
    dfaTargetIndex: null,
    dfaMaxValue: null,
    zQIndices: [],
    zKIndices: [],
    zValues: [],
    lossValues: [],
  };
}

// Maps a record to the Neuron fields and Activation rows that FeatureStats and ActivationsList read.
// dataSource puts a row in the positive or negative column. Bands carry binContains; the top and bottom
// lists do not, so their headers read "Top" and "Bottom". Top rows keep binMin -1 for the one-column view.
export function recordToNeuronDashboard(
  record: NeuronDashboardRecord,
  vocab: string[],
  ids: { modelId: string; layer: string; index: string },
): NeuronDashboard {
  if (record.v !== NEURON_DASHBOARD_FORMAT_VERSION) {
    throw new NeuronDashboardError(`unsupported dashboard format v${record.v}`);
  }
  const prefix = `${ids.modelId}-${ids.layer}-${ids.index}`;
  const activations: ActivationPartialWithRelations[] = [];
  record.top.forEach((example, i) => {
    activations.push(
      exampleToActivation(example, vocab, `${prefix}-top-${i}`, ids, {
        binMin: -1,
        binMax: record.stats.max,
        binContains: null,
      }),
    );
  });
  record.bands.forEach((band, b) => {
    band.examples.forEach((example, i) => {
      activations.push(
        exampleToActivation(example, vocab, `${prefix}-band${b}-${i}`, ids, {
          binMin: band.lo,
          binMax: band.hi,
          binContains: band.q_hi - band.q_lo,
        }),
      );
    });
  });
  const lowestBand = record.bands.reduce<NeuronDashboardBand | undefined>(
    (low, band) => (low === undefined || band.q_lo < low.q_lo ? band : low),
    undefined,
  );
  record.bottom.forEach((example, i) => {
    activations.push(
      exampleToActivation(example, vocab, `${prefix}-bottom-${i}`, ids, {
        binMin: record.stats.min,
        binMax: lowestBand ? lowestBand.lo : example.act,
        binContains: null,
      }),
    );
  });

  const { hist } = record;
  const total = record.stats.n_hist_tokens;
  const positive = [...hist.open_pos, ...hist.closed_pos].reduce((sum, count) => sum + count, 0);
  const linear = linearHistogram(record);

  return {
    neuron: {
      maxActApprox: record.stats.max,
      pos_str: record.logits.pos.map(([, token]) => token),
      pos_values: record.logits.pos.map(([, , value]) => value),
      neg_str: record.logits.neg.map(([, token]) => token),
      neg_values: record.logits.neg.map(([, , value]) => value),
      // FeatureStats labels this as the share of tokens with activation > 0.
      frac_nonzero: total > 0 ? positive / total : 0,
      // approximate; FeatureStats can also draw the exact counts with signedLogHistogram
      freq_hist_data_bar_values: linear.values,
      freq_hist_data_bar_heights: linear.heights,
    },
    activations,
    record,
  };
}

// Puts dashboard data over a neuron. Database-only fields (explanations, comments, lists) stay.
export function applyNeuronDashboard(
  neuron: NeuronWithPartialRelations,
  dashboard: NeuronDashboard,
): NeuronWithPartialRelations {
  return { ...neuron, ...dashboard.neuron, activations: dashboard.activations };
}

function trimSlash(url: string) {
  return url.replace(/\/+$/, '');
}

async function fetchRange(url: string, start: number, endExclusive: number, signal?: AbortSignal) {
  const response = await fetch(url, { headers: { Range: `bytes=${start}-${endExclusive - 1}` }, signal });
  if (response.status === 206) {
    return response.arrayBuffer();
  }
  if (response.status === 200) {
    // The server ignored the Range header and sent the whole file.
    return (await response.arrayBuffer()).slice(start, endExclusive);
  }
  if (response.status === 416) {
    throw new NeuronDashboardError(`no dashboard at ${url} bytes ${start}-${endExclusive - 1}`);
  }
  throw new NeuronDashboardError(`${url}: HTTP ${response.status}`);
}

const vocabCache = new Map<string, Promise<string[]>>();

export function fetchNeuronDashboardVocab(baseUrl: string): Promise<string[]> {
  const url = `${trimSlash(baseUrl)}/corpus/vocab.json`;
  let vocab = vocabCache.get(url);
  if (!vocab) {
    vocab = fetch(url).then(async (response) => {
      if (!response.ok) {
        throw new NeuronDashboardError(`${url}: HTTP ${response.status}`);
      }
      return (await response.json()) as string[];
    });
    // A failed fetch is not cached, so the next feature tries again.
    vocab.catch(() => vocabCache.delete(url));
    vocabCache.set(url, vocab);
  }
  return vocab;
}

export async function fetchNeuronDashboardRecord(
  baseUrl: string,
  layer: number,
  index: number,
  signal?: AbortSignal,
): Promise<NeuronDashboardRecord> {
  if (!Number.isInteger(layer) || layer < 0 || !Number.isInteger(index) || index < 0) {
    throw new NeuronDashboardError(`bad neuron ${layer}/${index}`);
  }
  const stem = `${trimSlash(baseUrl)}/${layerFileStem(layer)}`;
  const [start, end] = neuronByteRange(await fetchRange(`${stem}.idx`, 8 * index, 8 * index + 16, signal));
  const record = await gunzipJson<NeuronDashboardRecord>(await fetchRange(`${stem}.json.gz`, start, end, signal));
  if (record.layer !== layer || record.neuron !== index) {
    throw new NeuronDashboardError(`asked for ${layer}/${index}, got ${record.layer}/${record.neuron}`);
  }
  return record;
}

export async function fetchNeuronDashboard(
  baseUrl: string,
  ids: { modelId: string; layer: string; index: string },
  signal?: AbortSignal,
): Promise<NeuronDashboard> {
  if (!/^\d+$/.test(ids.layer) || !/^\d+$/.test(ids.index)) {
    throw new NeuronDashboardError(`bad neuron ${ids.layer}/${ids.index}`);
  }
  const [record, vocab] = await Promise.all([
    fetchNeuronDashboardRecord(baseUrl, Number(ids.layer), Number(ids.index), signal),
    fetchNeuronDashboardVocab(baseUrl),
  ]);
  return recordToNeuronDashboard(record, vocab, ids);
}
