"""
LLM-Enhanced Transparent Signal Processing Network

Advanced TSPN model with integrated LLM capabilities for natural language
explanations, interactive conversations, and enhanced diagnostic insights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime

# Import base classes
from .llm_explainable_base import LLMExplainableMixin
from .TSPN_explainable import (
    ExplainableSignalProcessingLayer,
    FeatureExtractorlayer,
    Classifier
)

# Import explainability modules
from explainability.llm.llm_explainer import LLMExplainer
from explainability.knowledge.fault_knowledge_graph import FaultKnowledgeGraph, FaultType, SeverityLevel
from explainability.knowledge.terminology_mapper import TerminologyMapper
from explainability.knowledge.context_processor import ContextProcessor


class TSPN_LLM_Enhanced(nn.Module, LLMExplainableMixin):
    """
    LLM-Enhanced Transparent Signal Processing Network.

    This model extends the explainable TSPN with LLM integration for:
    - Natural language explanations
    - Interactive diagnostic conversations
    - Context-aware recommendations
    - Knowledge-enhanced diagnostics
    """

    def __init__(self,
                 config: Dict[str, Any],
                 llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM-Enhanced TSPN model.

        Args:
            config: Model configuration dictionary
            llm_config: LLM configuration dictionary
        """
        nn.Module.__init__(self)
        LLMExplainableMixin.__init__(self, llm_config or {})

        # Store configuration
        self.config = config
        self.model_name = "TSPN_LLM_Enhanced"

        # Initialize network layers
        self._build_network_layers(config)

        # Initialize knowledge components
        self._initialize_knowledge_components()

        # Initialize LLM components
        self._initialize_llm_components(llm_config or {})

        # Tracking for enhanced explanations
        self._diagnostic_history = []
        self._conversation_sessions = {}

    def _build_network_layers(self, config: Dict[str, Any]) -> None:
        """Build network layers from configuration."""
        # Signal processing layers
        self.signal_layers = nn.ModuleList()
        layer_configs = [
            config.get('layer1', 'I'),
            config.get('layer2', 'I'),
            config.get('layer3', 'I'),
            config.get('layer4', 'I')
        ]

        in_channels = config.get('in_channels', 1)
        out_channels = config.get('out_channels', 1)
        scale = config.get('scale', 4)

        for i, layer_config in enumerate(layer_configs):
            layer = ExplainableSignalProcessingLayer(
                signal_processing_modules=layer_config,
                input_channels=in_channels,
                output_channels=out_channels,
                skip_connection=config.get('skip_connection', True),
                layer_index=i
            )
            self.signal_layers.append(layer)

        # Feature extraction
        self.feature_extractor = FeatureExtractorlayer(
            in_dim=config.get('in_dim', 4096),
            out_dim=config.get('out_dim', 10),
            scale=scale
        )

        # Classification layer
        self.classifier = Classifier(
            input_dim=scale * out_channels,
            num_classes=config.get('num_classes', 10)
        )

    def _initialize_knowledge_components(self) -> None:
        """Initialize knowledge enhancement components."""
        self.fault_knowledge = FaultKnowledgeGraph()
        self.terminology_mapper = TerminologyMapper()
        self.context_processor = ContextProcessor()

    def _initialize_llm_components(self, llm_config: Dict[str, Any]) -> None:
        """Initialize LLM components."""
        try:
            self.llm_explainer = LLMExplainer(self, llm_config)
            self._llm_enabled = True
        except Exception as e:
            print(f"Warning: LLM initialization failed: {e}")
            self._llm_enabled = False
            self.llm_explainer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor [batch_size, sequence_length, channels]

        Returns:
            Model predictions
        """
        # Signal processing layers
        for layer in self.signal_layers:
            x = layer(x)

        # Feature extraction
        features = self.feature_extractor(x)

        # Classification
        output = self.classifier(features)

        return output

    def predict_with_explanation(self,
                               input_data: torch.Tensor,
                               user_query: Optional[str] = None,
                               include_llm: bool = True) -> Dict[str, Any]:
        """
        Make prediction with comprehensive explanation.

        Args:
            input_data: Input tensor
            user_query: Optional user query
            include_llm: Whether to include LLM-enhanced explanation

        Returns:
            Comprehensive prediction and explanation
        """
        # Get basic prediction
        with torch.no_grad():
            prediction = self.forward(input_data)
            probabilities = F.softmax(prediction, dim=-1)
            confidence, predicted_class = torch.max(probabilities, dim=-1)

        # Generate technical explanation
        signal_path = self.get_signal_path(input_data)

        # Generate LLM-enhanced explanation if requested
        llm_explanation = None
        if include_llm and self._llm_enabled and user_query:
            try:
                llm_result = self.explain_with_llm(input_data, user_query)
                llm_explanation = llm_result.get('llm_enhanced_explanation')
            except Exception as e:
                print(f"Warning: LLM explanation failed: {e}")

        # Generate knowledge-enhanced insights
        knowledge_insights = self._generate_knowledge_insights(
            predicted_class.item(), confidence.item(), input_data
        )

        # Store diagnostic history
        diagnostic_record = {
            "timestamp": datetime.now().isoformat(),
            "predicted_class": predicted_class.item(),
            "confidence": confidence.item(),
            "probabilities": probabilities.tolist()[0],
            "signal_path_summary": self._summarize_signal_path(signal_path),
            "knowledge_insights": knowledge_insights,
            "user_query": user_query
        }
        self._diagnostic_history.append(diagnostic_record)

        return {
            "prediction": {
                "class_id": predicted_class.item(),
                "class_name": self._get_class_name(predicted_class.item()),
                "confidence": confidence.item(),
                "probabilities": probabilities.tolist()[0]
            },
            "signal_path": signal_path,
            "knowledge_insights": knowledge_insights,
            "llm_explanation": llm_explanation,
            "recommendations": self._generate_recommendations(
                predicted_class.item(), confidence.item()
            )
        }

    def start_diagnostic_conversation(self,
                                    input_data: torch.Tensor,
                                    device_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Start an interactive diagnostic conversation.

        Args:
            input_data: Input tensor for diagnosis
            device_info: Device information

        Returns:
            Initial greeting and diagnosis summary
        """
        # Generate initial diagnosis
        prediction_result = self.predict_with_explanation(input_data, include_llm=False)

        # Create session ID
        session_id = f"session_{len(self._conversation_sessions)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize conversation session
        session_data = {
            "session_id": session_id,
            "start_time": datetime.now(),
            "input_data": input_data,
            "device_info": device_info or {},
            "initial_diagnosis": prediction_result,
            "conversation_history": []
        }
        self._conversation_sessions[session_id] = session_data

        # Generate greeting
        fault_type = prediction_result["prediction"]["class_name"]
        confidence = prediction_result["prediction"]["confidence"]

        greeting = f"""您好！我是TSPN智能诊断助手。

根据您的设备振动数据分析，我检测到可能存在 **{fault_type** 故障，诊断置信度为 {confidence:.1%}。

我可以为您提供以下帮助：
• 详细的故障机理分析和技术解释
• 具体的维修方案和操作指导
• 风险评估和紧急程度判断
• 预防性维护建议

请告诉我您希望了解哪个方面，或者您有其他相关问题吗？"""

        return greeting

    def continue_conversation(self,
                            session_id: str,
                            user_message: str) -> str:
        """
        Continue an ongoing diagnostic conversation.

        Args:
            session_id: Conversation session ID
            user_message: User's message

        Returns:
            Assistant's response
        """
        if session_id not in self._conversation_sessions:
            return "抱歉，对话会话已过期。请重新开始诊断。"

        session = self._conversation_sessions[session_id]

        # Add user message to history
        session["conversation_history"].append({
            "speaker": "user",
            "message": user_message,
            "timestamp": datetime.now()
        })

        # Generate response
        if self._llm_enabled and self.llm_explainer:
            try:
                # Prepare context
                context = {
                    "session_info": {
                        "session_id": session_id,
                        "duration": (datetime.now() - session["start_time"]).total_seconds(),
                        "turn_count": len(session["conversation_history"])
                    },
                    "initial_diagnosis": session["initial_diagnosis"],
                    "device_info": session["device_info"],
                    "conversation_history": session["conversation_history"][-5:]  # Last 5 turns
                }

                # Generate LLM response
                response = self.llm_explainer.generate_conversation_response(
                    session["initial_diagnosis"],
                    user_message,
                    context
                )

            except Exception as e:
                print(f"Warning: LLM conversation failed: {e}")
                response = self._generate_rule_based_response(user_message, session)
        else:
            response = self._generate_rule_based_response(user_message, session)

        # Add assistant response to history
        session["conversation_history"].append({
            "speaker": "assistant",
            "message": response,
            "timestamp": datetime.now()
        })

        return response

    def _generate_knowledge_insights(self,
                                   predicted_class: int,
                                   confidence: float,
                                   input_data: torch.Tensor) -> Dict[str, Any]:
        """Generate knowledge-enhanced diagnostic insights."""
        fault_type = self._get_class_name(predicted_class)
        fault_type_enum = self._map_to_fault_type(fault_type)

        insights = {
            "fault_characteristics": {},
            "diagnostic_evidence": {},
            "related_faults": [],
            "maintenance_context": {}
        }

        if fault_type_enum:
            # Get fault pattern information
            fault_pattern = self.fault_knowledge.get_fault_pattern(fault_type_enum)
            if fault_pattern:
                insights["fault_characteristics"] = {
                    "symptoms": fault_pattern.symptoms,
                    "common_causes": fault_pattern.common_causes,
                    "typical_amplitudes": fault_pattern.typical_amplitudes
                }

                # Get characteristic frequencies (assuming typical RPM)
                typical_rpm = 1800  # Default assumption
                char_freqs = self.fault_knowledge.get_characteristic_frequencies(
                    fault_type_enum, typical_rpm
                )
                insights["diagnostic_evidence"]["expected_frequencies"] = char_freqs

            # Get related faults
            related_faults = self.fault_knowledge.get_related_faults(fault_type_enum)
            insights["related_faults"] = [
                {
                    "fault_type": rel.target_fault.value,
                    "relationship": rel.relationship_type,
                    "probability": rel.probability
                }
                for rel in related_faults
            ]

            # Get maintenance context based on severity
            severity = self._assess_severity(confidence)
            maintenance_actions = self.fault_knowledge.get_maintenance_actions(
                fault_type_enum, severity
            )
            insights["maintenance_context"]["recommended_actions"] = [
                {
                    "action": action.description,
                    "priority": action.priority,
                    "time_estimate": action.time_estimate
                }
                for action in maintenance_actions
            ]

        return insights

    def _generate_recommendations(self,
                                predicted_class: int,
                                confidence: float) -> List[Dict[str, Any]]:
        """Generate diagnostic and maintenance recommendations."""
        fault_type = self._get_class_name(predicted_class)
        fault_type_enum = self._map_to_fault_type(fault_type)
        severity = self._assess_severity(confidence)

        recommendations = []

        # Urgency-based recommendations
        if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            recommendations.append({
                "category": "immediate_action",
                "priority": "urgent",
                "recommendation": "立即停机检查，安排紧急维修",
                "reason": "检测到高风险故障，需要立即处理"
            })

        # Technical recommendations
        if fault_type_enum:
            recommendations.append({
                "category": "technical_investigation",
                "priority": "high",
                "recommendation": f"详细检查{fault_type}相关部件",
                "reason": "需要确认故障具体位置和严重程度"
            })

        # Monitoring recommendations
        recommendations.append({
            "category": "monitoring",
            "priority": "medium",
            "recommendation": "增加振动监测频率，跟踪故障发展趋势",
            "reason": "持续监控设备状态变化"
        })

        # Preventive recommendations
        if fault_type_enum:
            related_faults = self.fault_knowledge.get_related_faults(fault_type_enum)
            if related_faults:
                recommendations.append({
                    "category": "prevention",
                    "priority": "medium",
                    "recommendation": "检查可能相关的其他故障模式",
                    "reason": f"发现{len(related_faults)}种相关故障风险"
                })

        return recommendations

    def _generate_rule_based_response(self,
                                    user_message: str,
                                    session: Dict[str, Any]) -> str:
        """Generate rule-based response when LLM is unavailable."""
        initial_diagnosis = session["initial_diagnosis"]
        fault_type = initial_diagnosis["prediction"]["class_name"]
        confidence = initial_diagnosis["prediction"]["confidence"]

        message_lower = user_message.lower()

        if any(word in message_lower for word in ["原因", "为什么", "why"]):
            return f"""关于 **{fault_type}** 的原因分析：

这种故障的常见原因包括：
1. 正常磨损和材料疲劳
2. 润滑不良或污染
3. 过载运行或冲击载荷
4. 安装不当或对中不良

根据当前的置信度 {confidence:.1%}，建议首先检查设备的运行历史和维护记录，以确定最可能的原因。"""

        elif any(word in message_lower for word in ["维修", "维护", "处理"]):
            return f"""针对 **{fault_type}** 的维修建议：

**立即措施：**
• 监控设备运行状态
• 准备必要的维修备件
• 安排合适的维修窗口

**维修步骤：**
1. 详细检查故障部件
2. 更换损坏的零件
3. 检查相关部件状态
4. 重新安装和调试
5. 进行验证测试

**注意事项：**
• 遵循安全操作规程
• 使用合适的工具和设备
• 记录维修过程和结果"""

        elif any(word in message_lower for word in ["严重", "程度", "风险"]):
            severity_text = "高" if confidence > 0.8 else "中等" if confidence > 0.6 else "低"
            return f"""关于 **{fault_type}** 的严重程度评估：

**当前评估：** {severity_text}风险
**诊断置信度：** {confidence:.1%}

**风险分析：**
• 故障可能性：{severity_text}
• 影响程度：可能导致设备性能下降
• 紧急程度：{severity_text}

**建议措施：**
{self._get_severity_based_recommendations(confidence)}"""

        else:
            return f"""关于您的设备问题（**{fault_type}**），我可以为您提供以下方面的帮助：

• 故障机理的详细技术解释
• 具体的维修方案和操作指导
• 风险评估和紧急程度判断
• 预防性维护建议和监测策略

请告诉我您希望了解哪个具体方面？"""

    def _get_severity_based_recommendations(self, confidence: float) -> str:
        """Get recommendations based on confidence/severity."""
        if confidence > 0.8:
            return "立即安排专业检查，准备维修资源"
        elif confidence > 0.6:
            return "短期内安排详细检查，制定维修计划"
        else:
            return "加强监测，定期检查，按计划维护"

    def _assess_severity(self, confidence: float) -> SeverityLevel:
        """Assess fault severity based on confidence."""
        if confidence > 0.8:
            return SeverityLevel.HIGH
        elif confidence > 0.6:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    def _map_to_fault_type(self, class_name: str) -> Optional[FaultType]:
        """Map class name to fault type enum."""
        mapping = {
            "内圈故障": FaultType.INNER_RACE_FAULT,
            "外圈故障": FaultType.OUTER_RACE_FAULT,
            "滚动体故障": FaultType.BALL_DEFECT,
            "不对中": FaultType.MISALIGNMENT,
            "不平衡": FaultType.IMBALANCE
        }
        return mapping.get(class_name)

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        # This would typically come from the dataset configuration
        class_names = [
            "正常", "内圈故障", "外圈故障", "滚动体故障",
            "保持架故障", "不对中", "不平衡", "松动",
            "齿轮故障", "其他故障"
        ]
        return class_names[class_id] if class_id < len(class_names) else "未知"

    def _summarize_signal_path(self, signal_path: Dict[str, Any]) -> str:
        """Summarize signal path for storage."""
        if not signal_path or "data" not in signal_path:
            return "信号路径分析不可用"

        path_data = signal_path["data"]
        if "signal_path" in path_data:
            num_stages = len(path_data["signal_path"])
            return f"经过{num_stages}个信号处理阶段"

        return "信号路径分析完成"

    def get_diagnostic_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent diagnostic history."""
        return self._diagnostic_history[-limit:]

    def get_conversation_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation session data."""
        return self._conversation_sessions.get(session_id)

    def end_conversation(self, session_id: str) -> str:
        """End conversation session and provide summary."""
        if session_id not in self._conversation_sessions:
            return "对话会话不存在。"

        session = self._conversation_sessions[session_id]
        duration = datetime.now() - session["start_time"]
        num_turns = len(session["conversation_history"])

        conclusion = f"""感谢您的咨询！我们的对话持续了 {duration.total_seconds():.0f} 秒。

## 对话总结
• 交流轮次：{num_turns} 次
• 主要问题：{session['initial_diagnosis']['prediction']['class_name']}
• 诊断置信度：{session['initial_diagnosis']['prediction']['confidence']:.1%}

## 后续建议
1. 根据讨论结果制定具体维修计划
2. 加强设备状态监测
3. 定期进行预防性维护
4. 建立故障诊断档案

如果您还需要进一步帮助，可以随时开始新的对话。"""

        # Remove session
        del self._conversation_sessions[session_id]

        return conclusion

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "llm_enabled": self._llm_enabled,
            "config": self.config,
            "knowledge_components": {
                "fault_knowledge": True,
                "terminology_mapper": True,
                "context_processor": True
            },
            "capabilities": [
                "signal_processing_explanation",
                "llm_enhanced_explanations",
                "interactive_conversations",
                "knowledge_enhanced_diagnostics",
                "context_aware_recommendations"
            ]
        }