<template>

  <div class="sentiment-dashboard p-4">

    <AiStatusBanner ref="statusBannerRef" />



    <div class="mb-4 flex justify-end">

      <a-button :loading="loading" @click="loadStats">

        <template #icon><ReloadOutlined /></template>

        刷新数据

      </a-button>

    </div>



    <a-row :gutter="16" class="mb-4">

      <a-col :xs="24" :sm="12" :lg="6">

        <a-card :bordered="false" :loading="loading">

          <a-statistic title="综合满意度" :value="summary?.satisfaction_score ?? 0" suffix="分" :precision="1" />

        </a-card>

      </a-col>

      <a-col :xs="24" :sm="12" :lg="6">

        <a-card :bordered="false" :loading="loading">

          <a-statistic title="平均评分" :value="rating?.avg_rating ?? summary?.avg_rating ?? 0" suffix="/ 5" :precision="2" />

        </a-card>

      </a-col>

      <a-col :xs="24" :sm="12" :lg="6">

        <a-card :bordered="false" :loading="loading">

          <a-statistic title="评论总量" :value="polarity?.total ?? 0" />

        </a-card>

      </a-col>

      <a-col :xs="24" :sm="12" :lg="6">

        <a-card :bordered="false" :loading="loading">

          <a-statistic

            title="正向占比"

            :value="(polarity?.positive_ratio ?? 0) * 100"

            :precision="1"

            suffix="%"

            :value-style="{ color: '#3f8600' }"

          >

            <template #prefix>

              <ArrowUpOutlined v-if="summary?.trend_direction === 'up'" />

              <ArrowDownOutlined v-else-if="summary?.trend_direction === 'down'" />

              <MinusOutlined v-else />

            </template>

          </a-statistic>

        </a-card>

      </a-col>

    </a-row>



    <a-row :gutter="16" class="mb-4">

      <a-col :xs="24" :sm="8">

        <a-card :bordered="false" :loading="loading">

          <a-statistic

            title="正向评论"

            :value="polarity?.positive ?? 0"

            :value-style="{ color: '#3f8600' }"

            :suffix="polarity ? `（${(polarity.positive_ratio * 100).toFixed(1)}%）` : ''"

          />

        </a-card>

      </a-col>

      <a-col :xs="24" :sm="8">

        <a-card :bordered="false" :loading="loading">

          <a-statistic

            title="负向评论"

            :value="polarity?.negative ?? 0"

            :value-style="{ color: '#cf1322' }"

            :suffix="polarity ? `（${(polarity.negative_ratio * 100).toFixed(1)}%）` : ''"

          />

        </a-card>

      </a-col>

      <a-col :xs="24" :sm="8">

        <a-card :bordered="false" :loading="loading">

          <template #extra>

            <SourceBadge v-if="summary" :source="summary.source" />

          </template>

          <Gauge

            :chartData="{ name: '满意度', value: summary?.satisfaction_score ?? 0 }"

            height="160px"

            seriesColor="#52c41a"

          />

        </a-card>

      </a-col>

    </a-row>



    <a-row :gutter="16" class="mb-4">

      <a-col :xs="24" :lg="10">

        <a-card title="正负比例" :bordered="false" :loading="loading">

          <Pie :chartData="pieData" height="320px" />

        </a-card>

      </a-col>

      <a-col :xs="24" :lg="14">

        <a-card title="评分分布（1-5 星）" :bordered="false" :loading="loading">

          <template #extra>

            <SourceBadge v-if="rating" :source="rating.source" />

          </template>

          <Bar :chartData="ratingChartData" height="320px" seriesColor="#faad14" />

        </a-card>

      </a-col>

    </a-row>



    <a-row :gutter="16">

      <a-col :xs="24" :lg="14">

        <a-card title="评论量趋势（总量）" :bordered="false" :loading="loading">

          <SingleLine :chartData="trendTotalData" height="150px" seriesColor="#1890ff" />

        </a-card>

        <a-card title="正/负向趋势对比" :bordered="false" :loading="loading" class="mt-4">

          <LineMulti :chartData="trendSplitData" height="260px" type="line" />

        </a-card>

      </a-col>

      <a-col :xs="24" :lg="10">
        <PipelineOverview />
      </a-col>
    </a-row>

  </div>

</template>



<script lang="ts" setup>

  import { computed, onMounted, ref } from 'vue';

  import { ArrowDownOutlined, ArrowUpOutlined, MinusOutlined, ReloadOutlined } from '@ant-design/icons-vue';

  import Bar from '/@/components/chart/Bar.vue';

  import Pie from '/@/components/chart/Pie.vue';

  import SingleLine from '/@/components/chart/SingleLine.vue';

  import LineMulti from '/@/components/chart/LineMulti.vue';

  import Gauge from '/@/components/chart/Gauge.vue';

  import AiStatusBanner from '../components/AiStatusBanner.vue';

  import PipelineOverview from '../components/PipelineOverview.vue';

  import SourceBadge from '../components/SourceBadge.vue';

  import {

    getDashboardSummary,

    getPolarityStats,

    getRatingDistribution,

    getTrendStats,

    type DashboardSummary,

    type PolarityStat,

    type RatingDistribution,

    type TrendPoint,

  } from '../review.api';



  const loading = ref(false);

  const statusBannerRef = ref<{ refresh?: () => void } | null>(null);

  const polarity = ref<PolarityStat | null>(null);

  const summary = ref<DashboardSummary | null>(null);

  const rating = ref<RatingDistribution | null>(null);

  const pieData = ref<{ name: string; value: number }[]>([]);

  const trendTotalData = ref<TrendPoint[]>([]);

  const trendSplitData = ref<{ name: string; type: string; value: number; color?: string }[]>([]);



  const ratingChartData = computed(() =>

    (rating.value?.buckets || []).map((item) => ({

      name: `${item.rating} 星`,

      value: item.count,

    }))

  );



  async function loadStats() {

    loading.value = true;

    try {

      const [p, t, s, r] = await Promise.all([

        getPolarityStats(),

        getTrendStats(),

        getDashboardSummary(),

        getRatingDistribution(),

      ]);

      polarity.value = p;

      summary.value = s;

      rating.value = r;

      pieData.value = [

        { name: '正向', value: p.positive },

        { name: '负向', value: p.negative },

      ];

      trendTotalData.value = t;

      trendSplitData.value = t.flatMap((point) => [

        { name: point.name, type: '正向', value: point.positive ?? 0, color: '#3f8600' },

        { name: point.name, type: '负向', value: point.negative ?? 0, color: '#cf1322' },

      ]);

      statusBannerRef.value?.refresh?.();

    } finally {

      loading.value = false;

    }

  }



  onMounted(loadStats);

</script>

