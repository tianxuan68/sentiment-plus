<template>
  <div class="sentiment-keywords p-4">
    <AiStatusBanner />

    <a-card title="KeyBERT vs TF-IDF 关键词对比" :bordered="false">
      <a-textarea
        v-model:value="text"
        :rows="4"
        :maxlength="1000"
        show-count
        placeholder="输入评论文本，对比语义向量 KeyBERT 与统计基线 TF-IDF 的提取差异"
        @keydown.ctrl.enter="handleCompare"
      />
      <div class="mt-4 flex flex-wrap gap-3">
        <a-button type="primary" :loading="loading" @click="handleCompare">对比提取</a-button>
        <a-button @click="fillSample">填入示例</a-button>
        <span class="self-center text-gray-400 text-sm">Ctrl + Enter 快捷对比</span>
      </div>
    </a-card>

    <a-card v-if="result" class="mt-4" :bordered="false" :loading="loading">
      <template #extra>
        <a-space>
          <SourceBadge :source="result.source" />
          <a-tag color="purple">KeyBERT 独有 {{ result.keybert_unique_count }} 个</a-tag>
        </a-space>
      </template>

      <a-row :gutter="16">
        <a-col :xs="24" :lg="12">
          <div class="mb-2 font-medium text-blue-600">KeyBERT（语义向量）</div>
          <Bar :chartData="keybertChartData" height="260px" seriesColor="#722ed1" />
          <a-table
            class="mt-4"
            :columns="wordColumns"
            :data-source="result.keybert"
            :pagination="false"
            row-key="word"
            size="small"
          />
        </a-col>
        <a-col :xs="24" :lg="12">
          <div class="mb-2 font-medium text-orange-600">TF-IDF（统计基线）</div>
          <Bar :chartData="tfidfChartData" height="260px" seriesColor="#fa8c16" />
          <a-table
            class="mt-4"
            :columns="wordColumns"
            :data-source="result.tfidf"
            :pagination="false"
            row-key="word"
            size="small"
          />
        </a-col>
      </a-row>

      <a-divider />

      <div class="mb-2 text-gray-500">两种方法共同识别的关键词</div>
      <a-table
        v-if="result.overlap.length"
        :columns="overlapColumns"
        :data-source="result.overlap"
        :pagination="false"
        row-key="word"
        size="small"
      />
      <a-empty v-else description="暂无重叠关键词" />
    </a-card>
  </div>
</template>

<script lang="ts" setup>
  import { computed, ref } from 'vue';
  import { message } from 'ant-design-vue';
  import Bar from '/@/components/chart/Bar.vue';
  import AiStatusBanner from '../components/AiStatusBanner.vue';
  import SourceBadge from '../components/SourceBadge.vue';
  import { compareKeywords, type KeywordCompareResult } from '../review.api';

  const SAMPLE = '物流很快，包装有点简陋，但性价比还可以，客服回复也很耐心';

  const text = ref('');
  const loading = ref(false);
  const result = ref<KeywordCompareResult | null>(null);

  const wordColumns = [
    { title: '关键词', dataIndex: 'word', key: 'word' },
    {
      title: '得分',
      dataIndex: 'score',
      key: 'score',
      customRender: ({ text: v }: { text: number }) => v.toFixed(2),
    },
  ];

  const overlapColumns = [
    { title: '关键词', dataIndex: 'word', key: 'word' },
    {
      title: 'KeyBERT',
      dataIndex: 'keybert_score',
      key: 'keybert_score',
      customRender: ({ text: v }: { text: number }) => v.toFixed(2),
    },
    {
      title: 'TF-IDF',
      dataIndex: 'tfidf_score',
      key: 'tfidf_score',
      customRender: ({ text: v }: { text: number }) => v.toFixed(2),
    },
  ];

  const keybertChartData = computed(() =>
    [...(result.value?.keybert || [])].map((item) => ({ name: item.word, value: item.score }))
  );

  const tfidfChartData = computed(() =>
    [...(result.value?.tfidf || [])].map((item) => ({ name: item.word, value: item.score }))
  );

  async function handleCompare() {
    const value = text.value.trim();
    if (!value) {
      message.warning('请先输入评论文本');
      return;
    }
    loading.value = true;
    try {
      result.value = await compareKeywords({ text: value, top_n: 8 });
    } finally {
      loading.value = false;
    }
  }

  function fillSample() {
    text.value = SAMPLE;
  }
</script>
