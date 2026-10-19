import { beforeEach, describe, expect, it, vi } from 'vitest';
import fixture from '../utils/__fixtures__/neuron-dashboard-qwen3-32b-31-12345.json';
import { NeuronDashboardRecord, recordToNeuronDashboard } from '../utils/neuron-dashboards';
import { ensureNeuronRowForDashboard, getActivationsForFeature } from './neuron-dashboard';

// vi.mock is hoisted above the imports, so its factory can only close over
// values created by vi.hoisted.
const { sourceSetFindUnique, activationFindMany, neuronUpsert, fetchNeuronDashboard } = vi.hoisted(() => ({
  sourceSetFindUnique: vi.fn(),
  activationFindMany: vi.fn(),
  neuronUpsert: vi.fn(),
  fetchNeuronDashboard: vi.fn(),
}));

vi.mock('@/lib/db', () => ({
  prisma: {
    sourceSet: { findUnique: sourceSetFindUnique },
    activation: { findMany: activationFindMany },
    neuron: { upsert: neuronUpsert },
  },
}));
vi.mock('@/lib/utils/neuron-dashboards', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  fetchNeuronDashboard,
}));

const ids = { modelId: 'qwen3-32b', layer: '31', index: '12345' };
const vocab: string[] = [];
Object.entries(fixture.vocab).forEach(([id, token]) => {
  vocab[Number(id)] = token;
});
const dashboard = recordToNeuronDashboard(fixture.record as NeuronDashboardRecord, vocab, ids);

beforeEach(() => {
  vi.clearAllMocks();
});

describe('getActivationsForFeature', () => {
  it('reads from the dashboard files when the source set has them', async () => {
    sourceSetFindUnique.mockResolvedValue({ externalDashboardsUrl: 'https://dashboards.example/v1/qwen3-32b' });
    fetchNeuronDashboard.mockResolvedValue(dashboard);

    const { activations, dashboard: loaded } = await getActivationsForFeature(ids);

    expect(sourceSetFindUnique.mock.calls[0][0].where).toEqual({
      modelId_name: { modelId: 'qwen3-32b', name: 'neurons' },
    });
    expect(fetchNeuronDashboard).toHaveBeenCalledWith('https://dashboards.example/v1/qwen3-32b', ids);
    expect(activationFindMany).not.toHaveBeenCalled();
    expect(loaded).toBe(dashboard);
    expect(activations).toHaveLength(dashboard.activations.length);
    // the explainers read tokens and values; the route ranks by maxValue
    const top = activations.toSorted((a, b) => b.maxValue - a.maxValue)[0];
    expect(top.maxValue).toBe(fixture.record.stats.max);
    expect(top.tokens.length).toBe(top.values.length);
    expect(top).toMatchObject({ modelId: 'qwen3-32b', layer: '31', index: '12345' });
  });

  it('reads Activation rows when the source set has no dashboard files', async () => {
    sourceSetFindUnique.mockResolvedValue({ externalDashboardsUrl: null });
    activationFindMany.mockResolvedValue([{ id: 'a' }]);

    const { activations, dashboard: loaded } = await getActivationsForFeature({ ...ids, layer: '6-res-jb' });

    expect(sourceSetFindUnique.mock.calls[0][0].where.modelId_name.name).toBe('res-jb');
    expect(fetchNeuronDashboard).not.toHaveBeenCalled();
    expect(loaded).toBeNull();
    expect(activations).toEqual([{ id: 'a' }]);
  });
});

describe('ensureNeuronRowForDashboard', () => {
  it('creates a bare row only if missing, and returns it with the dashboard stats and logits', async () => {
    neuronUpsert.mockResolvedValue({ ...ids, sourceSetName: 'neurons', maxActApprox: 0, pos_str: [] });

    const feature = await ensureNeuronRowForDashboard(ids, dashboard);

    const args = neuronUpsert.mock.calls[0][0];
    expect(args.where).toEqual({ modelId_layer_index: ids });
    expect(args.update).toEqual({});
    expect(args.create).toEqual({ ...ids, sourceSetName: 'neurons', maxActApprox: fixture.record.stats.max });
    expect(feature.pos_str).toEqual(dashboard.neuron.pos_str);
    expect(feature.maxActApprox).toBe(fixture.record.stats.max);
  });
});
