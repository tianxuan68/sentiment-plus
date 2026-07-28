<template>
  <a-alert v-if="status" :type="alertType" show-icon class="mb-4">
    <template #message>
      <span>{{ title }}</span>
      <span v-if="status.ai_latency_ms != null" class="ml-2 text-gray-500">
        探测 {{ status.ai_latency_ms }} ms
      </span>
    </template>
    <template #description>
      {{ description }}
    </template>
  </a-alert>
</template>

<script lang="ts" setup>
  import { computed, onMounted, ref } from 'vue';
  import { getAiStatus, type AiStatusResult } from '../review.api';

  const status = ref<AiStatusResult | null>(null);

  const alertType = computed(() => {
    if (!status.value) return 'info';
    if (status.value.mode === 'live') return 'success';
    if (status.value.mode === 'degraded') return 'warning';
    return 'info';
  });

  const title = computed(() => {
    if (!status.value) return '加载服务状态…';
    if (status.value.mode === 'live') return 'sentiment-ai 在线';
    if (status.value.mode === 'degraded') return 'sentiment-ai 不可达，已降级 Mock';
    return '当前为 Mock 演示模式';
  });

  const description = computed(() => {
    if (!status.value) return '';
    if (status.value.mock_enabled) {
      return '关闭 SENTIMENT_AI_MOCK 并启动 sentiment-ai 后，将自动切换真实推理。';
    }
    return `服务地址：${status.value.ai_base_url}`;
  });

  async function loadStatus() {
    status.value = await getAiStatus();
  }

  onMounted(loadStatus);

  defineExpose({ refresh: loadStatus });
</script>
