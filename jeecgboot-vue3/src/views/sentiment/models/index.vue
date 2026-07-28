<template>
  <div class="sentiment-models p-4">
    <AiStatusBanner />

    <PipelineOverview />

    <a-card title="三模型情感对比链" :bordered="false" :loading="loading">
      <template #extra>
        <a-space>
          <SourceBadge v-if="data" :source="data.source" />
          <a-button size="small" @click="loadData">
            <template #icon><ReloadOutlined /></template>
            刷新
          </a-button>
        </a-space>
      </template>

      <a-alert
        v-if="data"
        type="info"
        show-icon
        class="mb-4"
        :message="`当前最优：${data.best_model} · BiLSTM 相对 Baseline F1 提升 ${(data.bilstm_vs_baseline_f1_gain * 100).toFixed(1)}%`"
      />

      <a-row :gutter="16">
        <a-col :xs="24" :lg="14">
          <div class="mb-2 text-gray-500">Accuracy 对比</div>
          <BarMulti :chartData="accChartData" height="300px" :seriesColor="seriesColors" />
        </a-col>
        <a-col :xs="24" :lg="10">
          <div class="mb-2 text-gray-500">F1-Score 对比</div>
          <BarMulti :chartData="f1ChartData" height="300px" :seriesColor="seriesColors" />
        </a-col>
      </a-row>

      <a-table
        class="mt-4"
        :columns="columns"
        :data-source="data?.metrics || []"
        :pagination="false"
        row-key="model"
        size="middle"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.key === 'model'">
            <a-tag v-if="record.model === data?.best_model" color="blue">{{ record.model }}</a-tag>
            <span v-else>{{ record.model }}</span>
          </template>
          <template v-else-if="column.key === 'acc'">
            {{ (record.acc * 100).toFixed(1) }}%
          </template>
          <template v-else-if="column.key === 'f1'">
            {{ (record.f1 * 100).toFixed(1) }}%
          </template>
        </template>
      </a-table>

      <div class="mt-4 text-gray-500 text-sm">
        对比链：Baseline（TF-IDF + 传统 ML）→ BiLSTM+Attention → BERT 微调；联调后指标由 sentiment-ai 实时返回。
      </div>
    </a-card>
  </div>
</template>

<script lang="ts" setup>
  import { computed, onMounted, ref } from 'vue';
  import { ReloadOutlined } from '@ant-design/icons-vue';
  import BarMulti from '/@/components/chart/BarMulti.vue';
  import AiStatusBanner from '../components/AiStatusBanner.vue';
  import PipelineOverview from '../components/PipelineOverview.vue';
  import SourceBadge from '../components/SourceBadge.vue';
  import { getModelCompare, type ModelCompareResult } from '../review.api';

  const loading = ref(false);
  const data = ref<ModelCompareResult | null>(null);

  const seriesColors = [
    { type: 'Accuracy', color: '#1890ff' },
    { type: 'F1-Score', color: '#52c41a' },
  ];

  const columns = [
    { title: '模型', dataIndex: 'model', key: 'model' },
    { title: 'Accuracy', dataIndex: 'acc', key: 'acc' },
    { title: 'F1-Score', dataIndex: 'f1', key: 'f1' },
    { title: '负责人', dataIndex: 'owner', key: 'owner' },
  ];

  const accChartData = computed(() =>
    (data.value?.metrics || []).flatMap((item) => [
      { name: shortName(item.model), type: 'Accuracy', value: Math.round(item.acc * 1000) / 10 },
    ])
  );

  const f1ChartData = computed(() =>
    (data.value?.metrics || []).flatMap((item) => [
      { name: shortName(item.model), type: 'F1-Score', value: Math.round(item.f1 * 1000) / 10 },
    ])
  );

  function shortName(model: string) {
    return model.replace(' Fine-tune', '').replace(' (TF-IDF+RF)', '');
  }

  async function loadData() {
    loading.value = true;
    try {
      data.value = await getModelCompare();
    } finally {
      loading.value = false;
    }
  }

  onMounted(loadData);
</script>
