"""
LLM-Enhanced Explainability Demo

Demonstration of the LLM-enhanced fault diagnosis system with
natural language explanations and interactive conversations.
"""

import torch
import numpy as np
from typing import Dict, Any

# Import the enhanced model
from model.TSPN_LLM_Enhanced import TSPN_LLM_Enhanced
from explainability.llm.llm_interface import create_llm_interface


def create_sample_vibration_data() -> torch.Tensor:
    """Create sample vibration data for demonstration."""
    # Generate synthetic vibration signal with bearing fault characteristics
    t = np.linspace(0, 1, 4096)
    sampling_rate = 4096

    # Simulate bearing fault frequencies
    shaft_freq = 30  # Hz
    bpfi = 3.05 * shaft_freq  # Ball pass frequency inner race

    # Create signal with fault characteristics
    signal = (
        0.5 * np.sin(2 * np.pi * shaft_freq * t) +  # Shaft frequency
        0.2 * np.sin(2 * np.pi * bpfi * t) +        # Inner race fault
        0.1 * np.sin(2 * np.pi * 2 * bpfi * t) +    # Harmonics
        0.05 * np.random.randn(len(t))              # Noise
    )

    # Convert to tensor with batch dimension
    tensor_signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    return tensor_signal


def demo_basic_prediction_with_explanation():
    """Demonstrate basic prediction with explanation."""
    print("=" * 60)
    print("演示1: 基础预测与解释")
    print("=" * 60)

    # Model configuration
    config = {
        'layer1': 'FFT',
        'layer2': 'WF',
        'layer3': 'HT',
        'layer4': 'I',
        'in_channels': 1,
        'out_channels': 1,
        'in_dim': 4096,
        'out_dim': 10,
        'scale': 4,
        'skip_connection': True,
        'num_classes': 10
    }

    # LLM configuration (without actual API keys for demo)
    llm_config = {
        'providers': {
            'mock': {
                'type': 'local',
                'model_path': 'mock_model'
            }
        }
    }

    try:
        # Create model (will work without actual LLM API)
        model = TSPN_LLM_Enhanced(config, llm_config)

        # Create sample data
        input_data = create_sample_vibration_data()

        # Get prediction with explanation
        result = model.predict_with_explanation(
            input_data,
            user_query="请详细解释这个故障的机理和维修建议",
            include_llm=False  # Set to False for demo without API
        )

        # Display results
        print(f"预测结果: {result['prediction']['class_name']}")
        print(f"置信度: {result['prediction']['confidence']:.1%}")
        print(f"概率分布: {result['prediction']['probabilities'][:3]}")  # First 3

        print(f"\n知识增强洞察:")
        insights = result['knowledge_insights']
        if 'fault_characteristics' in insights:
            print(f"  故障特征: {len(insights['fault_characteristics'].get('symptoms', []))} 种症状")
        if 'related_faults' in insights:
            print(f"  相关故障: {len(insights['related_faults'])} 种")

        print(f"\n推荐措施:")
        for rec in result['recommendations'][:3]:
            print(f"  • {rec['recommendation']} (优先级: {rec['priority']})")

    except Exception as e:
        print(f"演示1出现错误 (这是正常的，因为缺少完整依赖): {e}")


def demo_interactive_conversation():
    """Demonstrate interactive conversation capability."""
    print("\n" + "=" * 60)
    print("演示2: 交互式诊断对话")
    print("=" * 60)

    config = {
        'layer1': 'FFT',
        'layer2': 'WF',
        'layer3': 'HT',
        'layer4': 'I',
        'in_channels': 1,
        'out_channels': 1,
        'in_dim': 4096,
        'out_dim': 10,
        'scale': 4,
        'skip_connection': True,
        'num_classes': 10
    }

    try:
        # Create model
        model = TSPN_LLM_Enhanced(config)

        # Create sample data and device info
        input_data = create_sample_vibration_data()
        device_info = {
            "device_type": "电机",
            "operating_speed": 1800,
            "criticality_level": "high",
            "operating_hours": 15000
        }

        # Start conversation
        greeting = model.start_diagnostic_conversation(input_data, device_info)
        print("助手:", greeting)

        # Simulate conversation turns
        test_queries = [
            "这个故障的主要原因是什么？",
            "应该如何维修这个故障？",
            "故障严重程度如何评估？"
        ]

        session_id = list(model._conversation_sessions.keys())[0] if model._conversation_sessions else None

        for i, query in enumerate(test_queries, 1):
            if session_id:
                print(f"\n用户 ({i}): {query}")
                response = model.continue_conversation(session_id, query)
                print("助手:", response)
            else:
                print(f"\n会话未启动，跳过查询 {i}")

        # End conversation
        if session_id:
            conclusion = model.end_conversation(session_id)
            print(f"\n助手: {conclusion}")

    except Exception as e:
        print(f"演示2出现错误 (这是正常的): {e}")


def demo_llm_interface():
    """Demonstrate LLM interface capabilities."""
    print("\n" + "=" * 60)
    print("演示3: LLM接口功能")
    print("=" * 60)

    # Create mock LLM interface
    try:
        llm_interface = create_llm_interface(
            provider="mock",
            api_key="mock_key"
        )

        print("LLM接口状态:", "可用" if llm_interface.is_available() else "不可用")
        print("支持的提供商:", llm_interface.get_available_providers())

        # Test prompt building (without actual LLM call)
        from explainability.llm.prompt_manager import PromptManager
        prompt_manager = PromptManager()

        sample_explanation = "检测到内圈故障，频谱分析显示3.05倍频特征明显"
        sample_summary = "置信度85%，建议立即检查轴承状态"
        user_query = "请解释这个故障的维修方法"

        prompt = prompt_manager.build_prompt(
            sample_explanation,
            sample_summary,
            user_query,
            language="zh"
        )

        print(f"\n生成的提示长度: {len(prompt)} 字符")
        print("提示包含关键部分:", "角色设定" in prompt, "技术分析" in prompt, "建议" in prompt)

    except Exception as e:
        print(f"演示3出现错误: {e}")


def demo_knowledge_enhancement():
    """Demonstrate knowledge enhancement capabilities."""
    print("\n" + "=" * 60)
    print("演示4: 知识增强功能")
    print("=" * 60)

    try:
        from explainability.knowledge.fault_knowledge_graph import FaultKnowledgeGraph, FaultType
        from explainability.knowledge.terminology_mapper import TerminologyMapper
        from explainability.knowledge.context_processor import ContextProcessor

        # Knowledge graph demo
        knowledge_graph = FaultKnowledgeGraph()
        print("知识图谱包含故障类型:", len(knowledge_graph.fault_patterns))

        # Get fault pattern for inner race fault
        fault_pattern = knowledge_graph.get_fault_pattern(FaultType.INNER_RACE_FAULT)
        if fault_pattern:
            print(f"内圈故障症状数量: {len(fault_pattern.symptoms)}")
            print(f"常见原因数量: {len(fault_pattern.common_causes)}")

        # Characteristic frequencies
        char_freqs = knowledge_graph.get_characteristic_frequencies(
            FaultType.INNER_RACE_FAULT, 1800  # 1800 RPM
        )
        print(f"特征频率数量: {len(char_freqs)}")

        # Terminology mapper demo
        terminology_mapper = TerminologyMapper()
        test_terms = ["内圈故障", "IF", "不对中", "1X"]
        print("\n术语标准化:")
        for term in test_terms:
            standard = terminology_mapper.get_standard_term(term)
            print(f"  {term} -> {standard or '未识别'}")

        # Context processor demo
        context_processor = ContextProcessor()
        device_info = {
            "device_type": "电机",
            "operating_speed": 1800,
            "criticality_level": "high"
        }
        diagnostic_data = {
            "fault_type": "内圈故障",
            "confidence": 0.85,
            "severity": "high"
        }

        context = context_processor.process_diagnostic_context(
            device_info, diagnostic_data
        )
        print(f"\n上下文处理结果:")
        print(f"  运行上下文: {type(context.get('operational')).__name__}")
        print(f"  诊断上下文: {type(context.get('diagnostic')).__name__}")

    except Exception as e:
        print(f"演示4出现错误: {e}")


def demo_feedback_system():
    """Demonstrate feedback collection system."""
    print("\n" + "=" * 60)
    print("演示5: 反馈收集系统")
    print("=" * 60)

    try:
        from explainability.conversation.feedback_collector import FeedbackCollector, FeedbackType

        feedback_collector = FeedbackCollector()

        # Simulate feedback collection
        ratings = {
            "response_quality": 4,
            "diagnosis_accuracy": 5,
            "conversation_flow": 4,
            "user_satisfaction": 4
        }

        comments = {
            "response_quality": "解释很详细，技术内容准确",
            "diagnosis_accuracy": "诊断结果符合实际情况",
            "conversation_flow": "对话流畅，能够理解我的问题",
            "user_satisfaction": "总体满意，很有帮助"
        }

        feedback_id = feedback_collector.collect_session_feedback(
            session_id="demo_session_001",
            ratings=ratings,
            comments=comments,
            context={"device_type": "电机", "fault_severity": "high"}
        )

        print(f"反馈收集完成，反馈ID: {feedback_id}")
        print(f"收集的反馈项数量: {len(feedback_collector.feedback_items)}")

        # Get feedback summary
        summary = feedback_collector.get_feedback_summary()
        if "average_rating" in summary:
            print(f"平均评分: {summary['average_rating']:.1f}/5")
            print(f"反馈类型: {summary.get('feedback_types', [])}")

    except Exception as e:
        print(f"演示5出现错误: {e}")


def main():
    """Run all demonstrations."""
    print("LLM增强故障诊断系统演示")
    print("注意: 这是一个演示版本，某些功能可能需要完整的依赖配置")
    print()

    # Run demonstrations
    demo_basic_prediction_with_explanation()
    demo_interactive_conversation()
    demo_llm_interface()
    demo_knowledge_enhancement()
    demo_feedback_system()

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
    print("\n主要特性总结:")
    print("✓ 信号处理路径可视化")
    print("✓ 自然语言解释生成")
    print("✓ 交互式诊断对话")
    print("✓ 领域知识增强")
    print("✓ 上下文感知推荐")
    print("✓ 术语标准化")
    print("✓ 反馈收集与分析")
    print("✓ 多种LLM提供商支持")
    print("\n要使用完整功能，请:")
    print("1. 配置LLM API密钥")
    print("2. 安装完整依赖")
    print("3. 准备实际故障诊断数据")


if __name__ == "__main__":
    main()