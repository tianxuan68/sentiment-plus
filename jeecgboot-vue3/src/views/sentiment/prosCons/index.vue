<template>
  <div class="sentiment-pros-cons p-4">
    <AiStatusBanner />

    <a-card title="商品优缺点洞察" :bordered="false" :loading="loading">
      <template #extra>
        <a-space>
          <SourceBadge v-if="data" :source="data.source" />
          <a-button size="small" @click="loadData">
            <template #icon><ReloadOutlined /></template>
            刷新
          </a-button>
        </a-space>
      </template>

      <a-row :gutter="16">
        <a-col :xs="24" :lg="12">
          <div class="mb-3 font-medium text-green-600">优点 Top10</div>
          <Bar :chartData="prosChartData" height="280px" seriesColor="#52c41a" />
          <a-table
            class="mt-4"
            :columns="columns"
            :data-source="data?.pros || []"
            :pagination="false"
            row-key="word"
            size="small"
          />
        </a-col>
        <a-col :xs="24" :lg="12">
          <div class="mb-3 font-medium text-red-600">缺点 Top10</div>
          <Bar :chartData="consChartData" height="280px" seriesColor="#ff4d4f" />
          <a-table
            class="mt-4"
            :columns="columns"
            :data-source="data?.cons || []"
            :pagination="false"
            row-key="word"
            size="small"
          />
        </a-col>
      </a-row>

      <a-divider />

      <div class="text-gray-500 text-sm">
        数据来源：KeyBERT 正/负向评论桶聚合 Top10；联调后由 sentiment-ai 实时推理替换 Mock。
      </div>
    </a-card>
  </div>
</template>

<script lang="ts" setup>
  import { computed, onMounted, ref } from 'vue';
  import { ReloadOutlined } from '@ant-design/icons-vue';
  import Bar from '/@/components/chart/Bar.vue';
  import AiStatusBanner from '../components/AiStatusBanner.vue';
  import SourceBadge from '../components/SourceBadge.vue';
  import { getProsCons, type ProsConsResult } from '../review.api';

  const loading = ref(false);
  const data = ref<ProsConsResult | null>(null);

  const columns = [
    { title: '关键词', dataIndex: 'word', key: 'word' },
    {
      title: '相关度',
      dataIndex: 'score',
      key: 'score',
      customRender: ({ text }: { text: number }) => text.toFixed(2),
    },
  ];

  const prosChartData = computed(() =>
    [...(data.value?.pros || [])]
      .sort((a, b) => a.score - b.score)
      .map((item) => ({ name: item.word, value: item.score }))
  );

  const consChartData = computed(() =>
    [...(data.value?.cons || [])]
      .sort((a, b) => a.score - b.score)
      .map((item) => ({ name: item.word, value: item.score }))
  );

  async function loadData() {
    loading.value = true;
    try {
      data.value = await getProsCons();
    } finally {
      loading.value = false;
    }
  }

  onMounted(loadData);
</script>
