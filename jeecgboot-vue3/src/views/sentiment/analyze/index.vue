<template>
  <div class="p-4">
    <a-card title="单条评论分析" :bordered="false">
      <a-textarea v-model:value="text" :rows="5" placeholder="输入商品评论…" allow-clear />
      <div class="mt-3">
        <a-radio-group v-model:value="model" class="mr-3">
          <a-radio-button value="baseline">Baseline</a-radio-button>
          <a-radio-button value="bert">BERT（预留）</a-radio-button>
        </a-radio-group>
        <a-button class="mr-2" @click="fillSample">样例</a-button>
        <a-button type="primary" :loading="loading" @click="runAnalyze">开始分析</a-button>
      </div>
    </a-card>

    <a-row v-if="result" :gutter="16" class="mt-4">
      <a-col :span="8">
        <a-card title="情感分类" :bordered="false">
          <a-result :status="resultStatus" :title="result.labelText" :sub-title="`score=${result.score}`" />
        </a-card>
      </a-col>
      <a-col :span="8">
        <a-card title="关键词" :bordered="false">
          <a-tag v-for="k in keywords" :key="k.word" color="blue" class="mb-2">{{ k.word }}</a-tag>
        </a-card>
      </a-col>
      <a-col :span="8">
        <a-card title="属性情感" :bordered="false">
          <a-table size="small" :pagination="false" :columns="cols" :data-source="aspects" row-key="aspect" />
        </a-card>
      </a-col>
    </a-row>
  </div>
</template>

<script lang="ts" name="sentiment-analyze" setup>
  import { computed, ref } from 'vue';
  import { message } from 'ant-design-vue';
  import { analyzeAspects, extractKeywords, predictSentiment } from '../sentiment.api';

  const text = ref('');
  const model = ref('baseline');
  const loading = ref(false);
  const result = ref<Recordable | null>(null);
  const keywords = ref<any[]>([]);
  const aspects = ref<any[]>([]);
  const cols = [
    { title: '属性', dataIndex: 'aspect' },
    { title: '极性', dataIndex: 'polarity' },
    { title: '触发词', dataIndex: 'triggers' },
  ];
  const resultStatus = computed(() => {
    if (!result.value) return 'info';
    if (result.value.label === 1) return 'success';
    if (result.value.label === 0) return 'error';
    return 'warning';
  });
  function fillSample() {
    text.value = '物流很快包装也完好，质量不错好用，就是价格有点贵，客服态度还行。';
  }
  async function runAnalyze() {
    if (!text.value.trim()) return message.warning('请输入评论');
    loading.value = true;
    try {
      const [pred, kw, asp] = await Promise.all([
        predictSentiment({ text: text.value, model: model.value }),
        extractKeywords({ text: text.value, topN: 10 }),
        analyzeAspects({ text: text.value }),
      ]);
      result.value = pred || null;
      keywords.value = kw?.keywords || [];
      aspects.value = (asp?.aspects || []).map((a: any) => ({
        ...a,
        triggers: (a.triggers || []).join('、'),
      }));
    } finally {
      loading.value = false;
    }
  }
</script>

<style scoped>
  .mt-3 { margin-top: 12px; }
  .mt-4 { margin-top: 16px; }
  .mr-2 { margin-right: 8px; }
  .mr-3 { margin-right: 12px; }
  .mb-2 { margin-bottom: 8px; }
</style>
