<template>
  <a-tag :color="color">{{ label }}</a-tag>
</template>

<script lang="ts" setup>
  import { computed } from 'vue';

  const props = defineProps<{ source: string }>();

  const meta = computed(() => {
    switch (props.source) {
      case 'ai':
        return { label: 'AI 推理', color: 'success' };
      case 'mock_fallback':
        return { label: 'AI 降级', color: 'warning' };
      case 'mock':
      default:
        return { label: 'Mock 演示', color: 'default' };
    }
  });

  const label = computed(() => meta.value.label);
  const color = computed(() => meta.value.color);
</script>
