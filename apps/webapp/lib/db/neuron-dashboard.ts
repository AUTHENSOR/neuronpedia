// Server reads of neuron dashboards stored as files (SourceSet.externalDashboardsUrl).
// Such neurons have no Activation rows, and no Neuron row until something is written for them.

import { prisma } from '@/lib/db';
import { fetchNeuronDashboard, NeuronDashboard } from '@/lib/utils/neuron-dashboards';
import { getSourceSetNameFromSource } from '@/lib/utils/source';
import { ActivationPartialWithRelations } from '@/prisma/generated/zod';
import { Activation, Neuron } from '@prisma/client';

type FeatureIds = { modelId: string; layer: string; index: string };

// The dashboard from files, or null when the feature's source set keeps its dashboards in the database.
export async function getExternalNeuronDashboard(ids: FeatureIds): Promise<NeuronDashboard | null> {
  const sourceSet = await prisma.sourceSet.findUnique({
    where: { modelId_name: { modelId: ids.modelId, name: getSourceSetNameFromSource(ids.layer) } },
    select: { externalDashboardsUrl: true },
  });
  if (!sourceSet?.externalDashboardsUrl) {
    return null;
  }
  return fetchNeuronDashboard(sourceSet.externalDashboardsUrl, ids);
}

function toActivationRow(activation: ActivationPartialWithRelations, ids: FeatureIds): Activation {
  return {
    id: activation.id || '',
    tokens: activation.tokens || [],
    dataIndex: activation.dataIndex ?? null,
    index: ids.index,
    layer: ids.layer,
    modelId: ids.modelId,
    dataSource: activation.dataSource ?? null,
    maxValue: activation.maxValue ?? 0,
    maxValueTokenIndex: activation.maxValueTokenIndex ?? 0,
    minValue: activation.minValue ?? 0,
    values: activation.values || [],
    dfaValues: [],
    dfaTargetIndex: null,
    dfaMaxValue: null,
    zQIndices: [],
    zKIndices: [],
    zValues: [],
    creatorId: '',
    createdAt: new Date(),
    lossValues: [],
    logitContributions: null,
    binMin: activation.binMin ?? null,
    binMax: activation.binMax ?? null,
    binContains: activation.binContains ?? null,
    qualifyingTokenIndex: null,
  };
}

// A feature's activations, from its dashboard files when it has them, otherwise from the database.
export async function getActivationsForFeature(
  ids: FeatureIds,
): Promise<{ activations: Activation[]; dashboard: NeuronDashboard | null }> {
  const dashboard = await getExternalNeuronDashboard(ids);
  if (dashboard) {
    return { activations: dashboard.activations.map((a) => toActivationRow(a, ids)), dashboard };
  }
  const activations = await prisma.activation.findMany({
    where: { neuron: { modelId: ids.modelId, layer: ids.layer, index: ids.index } },
  });
  return { activations, dashboard: null };
}

// Rows that reference a feature (explanations, for example) need its Neuron row. Creates a bare one
// if it is missing and returns it with the dashboard's stats and logits on top.
export async function ensureNeuronRowForDashboard(ids: FeatureIds, dashboard: NeuronDashboard): Promise<Neuron> {
  const row = await prisma.neuron.upsert({
    where: { modelId_layer_index: ids },
    create: {
      ...ids,
      sourceSetName: getSourceSetNameFromSource(ids.layer),
      maxActApprox: dashboard.neuron.maxActApprox,
    },
    update: {},
  });
  return { ...row, ...dashboard.neuron, maxActApprox: dashboard.neuron.maxActApprox ?? row.maxActApprox };
}
