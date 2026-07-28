<template>
  <div class="aspect-highlight">
    <span v-for="(part, index) in parts" :key="index">
      <a-tooltip v-if="part.entity" :title="`${part.entity.aspect} · ${part.entity.polarity_text}`">
        <mark :class="part.entity.polarity === 1 ? 'entity-pos' : 'entity-neg'">{{ part.text }}</mark>
      </a-tooltip>
      <template v-else>{{ part.text }}</template>
    </span>
  </div>
</template>

<script lang="ts" setup>
  import { computed } from 'vue';
  import type { AspectEntity } from '../review.api';

  const props = defineProps<{
    text: string;
    entities: AspectEntity[];
  }>();

  interface TextPart {
    text: string;
    entity?: AspectEntity;
  }

  const parts = computed(() => {
    const text = props.text || '';
    const entities = [...(props.entities || [])].sort((a, b) => a.start - b.start);
    if (!text || !entities.length) {
      return [{ text }] as TextPart[];
    }

    const result: TextPart[] = [];
    let cursor = 0;
    for (const entity of entities) {
      if (entity.start < cursor || entity.end > text.length) {
        continue;
      }
      if (entity.start > cursor) {
        result.push({ text: text.slice(cursor, entity.start) });
      }
      result.push({ text: text.slice(entity.start, entity.end), entity });
      cursor = entity.end;
    }
    if (cursor < text.length) {
      result.push({ text: text.slice(cursor) });
    }
    return result.length ? result : [{ text }];
  });
</script>

<style scoped lang="less">
  .aspect-highlight {
    line-height: 1.8;
    font-size: 14px;
    color: rgba(0, 0, 0, 0.85);
    word-break: break-all;
  }

  mark {
    padding: 0 2px;
    border-radius: 2px;
    background: transparent;
  }

  .entity-pos {
    background: rgba(82, 196, 26, 0.18);
    border-bottom: 2px solid #52c41a;
  }

  .entity-neg {
    background: rgba(255, 77, 79, 0.15);
    border-bottom: 2px solid #ff4d4f;
  }
</style>
