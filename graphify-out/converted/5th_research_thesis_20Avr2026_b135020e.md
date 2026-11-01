<!-- converted from 5th_research_thesis_20Avr2026.docx -->

## 1. Kaiwu: A Multimodal Manipulation Dataset and Framework for Robot Learning and Human-Robot Interaction
Overview
This paper introduces Kaiwu, a large-scale multimodal dataset and collection framework designed to support robot learning, dexterous manipulation, and human–robot collaboration research. It targets real-world tasks with complex human interactions—a scenario where robots must infer intent and learn control policies from demonstrations.
Dataset & Data Modalities
20 human subjects interacting with 30 objects, producing 11,664 multimodal action segments.
Modalities include:
Hand motion and operation forces, capturing dynamics of manipulation.
Multi-view RGB and depth video, enabling visual observation of tasks.
High-precision motion capture and skeletal tracking, essential for modeling human motion.
Eye gaze (first-person video) and electromyography (EMG), providing signals related to intent and muscle activity.
Audio recordings of assembly processes.
All data streams are temporally synchronized with fine-grained annotations and segmentation labels to support multimodal learning and cross-modal fusion.
## 2. PoseTron: Enabling Close-Proximity Human-Robot Collaboration Through Multi-human Motion Prediction
Overview
This work proposes PoseTron, a transformer-based model for predicting human motion in close-proximity human–robot collaboration (HRC). It also introduces a new multimodal dataset (INTERACT) specifically focused on joint human–robot and human–human collaborative dynamics.
INTERACT Dataset (Context)
PoseTron is evaluated on INTERACT, a multimodal dataset that includes:
3D skeletal motion of multiple agents.
RGB and depth (RGB+D) video streams.
Gaze and robot joint state data.
This dataset supports studying rich agent dynamics in collaborative settings where robots and humans operate in the same workspace.
Model & Approach
PoseTron uses a conditional attention mechanism in its transformer encoder to integrate motion patterns from all agents, effectively capturing interactive dynamics.
Its decoder employs a multimodal attention mechanism to weight information from different modalities when predicting future motion trajectories.
PoseTron achieves lower motion prediction error compared to baseline approaches widely used in HRC tasks.
## 3. Learning to Share Autonomy from Repeated Human-Robot Interaction
Overview
This work introduces SARI (Shared Autonomy across Repeated Interaction), an algorithm that enables assistive robots to learn shared autonomy policies from repeated human–robot interactions, without requiring a prior task definitions.
Core Contributions
Task Recognition from Interaction Data – Embeds human behavior patterns to recognize the current task based on past interactions.
Learning to Replicate Demonstrations – From repeated task demonstrations, the robot learns policies that mimic human guidance.
Autonomy Arbitration – The system continually assesses confidence in its learned assistance, returning control to the human if uncertain.
Theoretical Guarantees and Empirical Evaluation – Provides bounds on assistance error and shows robustness across various tasks through simulation and user studies.