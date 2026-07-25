<template>
  <div class="p-4">
    <a-alert
      type="info"
      show-icon
      class="mb-4"
      message="Sentiment-Plus：基于 Jeecg 系统壳的评论情感分析。概览为演示统计。"
    />
    <a-row :gutter="16">
      <a-col :span="6" v-for="card in statCards" :key="card.title">
        <a-card :bordered="false">
          <div class="stat-title">{{ card.title }}</div>
          <div class="stat-value" :style="{ color: card.color }">{{ card.value }}</div>
        </a-card>
      </a-col>
    </a-row>
    <a-row :gutter="16" class="mt-4">
      <a-col :span="12"><a-card title="情感比例" :bordered="false" :loading="loading"><div ref="pieRef" style="height: 300px"></div></a-card></a-col>
      <a-col :span="12"><a-card title="正负向趋势" :bordered="false" :loading="loading"><div ref="trendRef" style="height: 300px"></div></a-card></a-col>
    </a-row>
    <a-row :gutter="16" class="mt-4">
      <a-col :span="12"><a-card title="模型对比" :bordered="false" :loading="loading"><div ref="modelRef" style="height: 300px"></div></a-card></a-col>
      <a-col :span="12"><a-card title="属性覆盖" :bordered="false" :loading="loading"><div ref="aspectRef" style="height: 300px"></div></a-card></a-col>
    </a-row>
  </div>
</template>

<script lang="ts" name="sentiment-dashboard" setup>
  import { computed, nextTick, onMounted, ref, type Ref } from 'vue';
  import { useECharts } from '/@/hooks/web/useECharts';
  import { getSentimentOverview } from '../sentiment.api';

  const loading = ref(true);
  const overview = ref<Recordable>({});
  const pieRef = ref<HTMLDivElement | null>(null);
  const trendRef = ref<HTMLDivElement | null>(null);
  const modelRef = ref<HTMLDivElement | null>(null);
  const aspectRef = ref<HTMLDivElement | null>(null);
  const { setOptions: setPie } = useECharts(pieRef as Ref<HTMLDivElement>);
  const { setOptions: setTrend } = useECharts(trendRef as Ref<HTMLDivElement>);
  const { setOptions: setModel } = useECharts(modelRef as Ref<HTMLDivElement>);
  const { setOptions: setAspect } = useECharts(aspectRef as Ref<HTMLDivElement>);
  const pct = (v?: number) => `${(((v || 0) as number) * 100).toFixed(1)}%`;
  const statCards = computed(() => [
    { title: '评论总量', value: overview.value.totalReviews ?? '-', color: '#1890ff' },
    { title: '正向占比', value: pct(overview.value.positiveRate), color: '#52c41a' },
    { title: '负向占比', value: pct(overview.value.negativeRate), color: '#ff4d4f' },
    { title: 'BERT Acc', value: pct(overview.value.modelAccuracy?.bert), color: '#722ed1' },
  ]);

  function renderCharts() {
    const o = overview.value;
    setPie({
      tooltip: { trigger: 'item' },
      series: [{
        type: 'pie', radius: ['40%', '68%'],
        data: [
          { name: '正向', value: Math.round((o.positiveRate || 0) * 1000) },
          { name: '负向', value: Math.round((o.negativeRate || 0) * 1000) },
          { name: '中性', value: Math.round((o.neutralRate || 0) * 1000) },
        ],
      }],
    });
    const trend = o.trend || [];
    setTrend({
      tooltip: { trigger: 'axis' },
      legend: { data: ['正向', '负向'] },
      xAxis: { type: 'category', data: trend.map((t: any) => t.month) },
      yAxis: { type: 'value', min: 0, max: 1 },
      series: [
        { name: '正向', type: 'line', data: trend.map((t: any) => t.positive), smooth: true },
        { name: '负向', type: 'line', data: trend.map((t: any) => t.negative), smooth: true },
      ],
    });
    const acc = o.modelAccuracy || {};
    const names = Object.keys(acc);
    setModel({
      xAxis: { type: 'category', data: names },
      yAxis: { type: 'value', min: 0.8, max: 1 },
      series: [{ type: 'bar', data: names.map((n) => acc[n]) }],
    });
    const coverage = o.aspectCoverage || [];
    setAspect({
      xAxis: { type: 'category', data: coverage.map((c: any) => c.aspect) },
      yAxis: { type: 'value' },
      series: [{ type: 'bar', data: coverage.map((c: any) => c.count) }],
    });
  }

  onMounted(async () => {
    try {
      overview.value = (await getSentimentOverview()) || {};
      await nextTick();
      renderCharts();
    } finally {
      loading.value = false;
    }
  });
</script>

<style scoped>
  .stat-title { color: rgba(0, 0, 0, 0.45); }
  .stat-value { margin-top: 8px; font-size: 28px; font-weight: 600; }
  .mb-4 { margin-bottom: 16px; }
  .mt-4 { margin-top: 16px; }
</style>
