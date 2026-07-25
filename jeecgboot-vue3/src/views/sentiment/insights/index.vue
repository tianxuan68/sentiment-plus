<template>
  <div class="p-4">
    <a-row :gutter="16">
      <a-col :span="12">
        <a-card title="优点 Top10" :bordered="false" :loading="loading">
          <a-table size="small" :pagination="false" :columns="columns" :data-source="pros" row-key="word" />
        </a-card>
      </a-col>
      <a-col :span="12">
        <a-card title="缺点 Top10" :bordered="false" :loading="loading">
          <a-table size="small" :pagination="false" :columns="columns" :data-source="cons" row-key="word" />
        </a-card>
      </a-col>
    </a-row>
    <a-card title="批量评论快检" class="mt-4" :bordered="false">
      <a-textarea v-model:value="batchText" :rows="6" placeholder="每行一条评论" />
      <a-button type="primary" class="mt-3" :loading="batchLoading" @click="runBatch">批量分析</a-button>
      <a-descriptions v-if="batchResult" bordered size="small" class="mt-4" :column="4">
        <a-descriptions-item label="总数">{{ batchResult.total }}</a-descriptions-item>
        <a-descriptions-item label="正向">{{ batchResult.positive }}</a-descriptions-item>
        <a-descriptions-item label="负向">{{ batchResult.negative }}</a-descriptions-item>
        <a-descriptions-item label="中性">{{ batchResult.neutral }}</a-descriptions-item>
      </a-descriptions>
    </a-card>
  </div>
</template>

<script lang="ts" name="sentiment-insights" setup>
  import { onMounted, ref } from 'vue';
  import { message } from 'ant-design-vue';
  import { getSentimentOverview, predictSentimentBatch } from '../sentiment.api';

  const loading = ref(true);
  const batchLoading = ref(false);
  const pros = ref<any[]>([]);
  const cons = ref<any[]>([]);
  const batchText = ref('');
  const batchResult = ref<Recordable | null>(null);
  const columns = [
    { title: '排名', customRender: ({ index }: any) => index + 1, width: 70 },
    { title: '关键词', dataIndex: 'word' },
    { title: '得分', dataIndex: 'score', width: 100 },
  ];

  onMounted(async () => {
    try {
      const data = (await getSentimentOverview()) || {};
      pros.value = data.prosTop10 || [];
      cons.value = data.consTop10 || [];
    } finally {
      loading.value = false;
    }
  });

  async function runBatch() {
    const texts = batchText.value.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
    if (!texts.length) return message.warning('请输入评论');
    batchLoading.value = true;
    try {
      batchResult.value = (await predictSentimentBatch({ texts })) || null;
    } finally {
      batchLoading.value = false;
    }
  }
</script>

<style scoped>
  .mt-3 { margin-top: 12px; }
  .mt-4 { margin-top: 16px; }
</style>
