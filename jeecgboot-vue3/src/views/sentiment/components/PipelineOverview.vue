<template>
  <a-card :bordered="false" class="pipeline-overview mb-4">
    <div class="pipeline-title">NLP 分析链路</div>
    <div class="pipeline-steps">
      <div v-for="(step, index) in steps" :key="step.key" class="pipeline-step">
        <div class="step-icon" :style="{ background: step.color }">
          <Icon :icon="step.icon" :size="18" color="#fff" />
        </div>
        <div class="step-body">
          <div class="step-name">{{ step.title }}</div>
          <div class="step-desc">{{ step.desc }}</div>
        </div>
        <Icon v-if="index < steps.length - 1" icon="ant-design:arrow-right-outlined" class="step-arrow" />
      </div>
    </div>
  </a-card>
</template>

<script lang="ts" setup>
  import Icon from '/@/components/Icon';

  const steps = [
    { key: 'sentiment', title: '情感分类', desc: 'Baseline→BiLSTM→BERT', icon: 'ant-design:fund-outlined', color: '#1890ff' },
    { key: 'keyword', title: '关键词', desc: 'KeyBERT vs TF-IDF', icon: 'ant-design:tags-outlined', color: '#722ed1' },
    { key: 'aspect', title: '属性情感', desc: 'BIO NER + 极性', icon: 'ant-design:cluster-outlined', color: '#13c2c2' },
    { key: 'proscons', title: '优缺洞察', desc: '正/负桶 Top10', icon: 'ant-design:like-outlined', color: '#52c41a' },
  ];
</script>

<style scoped lang="less">
  .pipeline-overview {
    background: linear-gradient(135deg, #fafafa 0%, #fff 100%);
  }

  .pipeline-title {
    margin-bottom: 12px;
    font-size: 14px;
    font-weight: 500;
    color: rgba(0, 0, 0, 0.65);
  }

  .pipeline-steps {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 4px;
    align-items: center;
  }

  .pipeline-step {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 180px;
    flex: 1;
  }

  .step-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 8px;
    flex-shrink: 0;
  }

  .step-body {
    min-width: 0;
  }

  .step-name {
    font-size: 13px;
    font-weight: 500;
    color: rgba(0, 0, 0, 0.85);
  }

  .step-desc {
    font-size: 12px;
    color: rgba(0, 0, 0, 0.45);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .step-arrow {
    margin: 0 4px;
    color: rgba(0, 0, 0, 0.25);
    flex-shrink: 0;
  }

  @media (max-width: 768px) {
    .step-arrow {
      display: none;
    }

    .pipeline-step {
      min-width: calc(50% - 8px);
    }
  }
</style>
