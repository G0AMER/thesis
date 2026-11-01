<!-- converted from Kaiwu_HRC_Executive_Summary_Detailed.docx -->

Target System
A personalized multimodal shared-autonomy system that:
1. Understands user intent from multimodal streams.
2. Predicts/assists manipulation actions in real time.
3. Adapts to new users incrementally without forgetting previous ones.
Proposed Architecture
1. Data Layer (Kaiwu Interface)
- Inputs: multi-view video, hand motion, operation pressure/force-like signals, audio, motion capture, gaze + egocentric video, EMG.
- Core services: timestamp alignment, resampling, missing-sensor masking, per-modality normalization.
- Output: synchronized temporal windows (for example 1-2 s rolling windows).
2. Perception Layer (Modality Encoders)
- Video encoder: spatial-temporal encoder for multi-view + first-person.
- Kinematic encoder: hand trajectory encoder (temporal transformer or TCN).
- Bio-signal encoder: EMG temporal encoder.
- Interaction encoder: pressure + audio encoder.
- Gaze encoder: gaze trajectory and fixation features.
- Output: one latent embedding per modality at each time step.
3. Fusion + Understanding Layer
- Multimodal fusion transformer with cross-attention and modality masks.
- Shared latent state split into three semantics:
1. Intent state (what user wants to do next).
2. Phase/state-of-task (assembly step progress).
3. Control confidence/risk (how safe/reliable assistance is now).
- Auxiliary heads:
1. Intent classification/regression.
2. Fine-grained action/phase prediction.
3. Confidence and anomaly detection.
4. Decision & Adaptation Layer (Shared Autonomy + Continual Learning)
- Assistance policy head: proposes robot assistance/control command.
- Arbitration module: blends user command and autonomous command.
- Safety governor: rule/constraint filter before final command.
- Incremental adaptation module:
1. Shared backbone (generalizable knowledge).
2. User-specific lightweight adapters (personalization).
3. Replay memory (small episodic buffer).
4. Importance regularization (protect previously important parameters)
Training Strategy
1. Stage A: Representation Pretraining
- Self-supervised multimodal objectives (contrastive alignment, masked reconstruction, temporal order consistency).
2. Stage B: Supervised Task Learning
- Train intent, phase, and action heads with Kaiwu annotations.
3. Stage C: Shared-Autonomy Policy Learning
- Behavioral cloning first, then optional RL fine-tuning in simulation.
4. Stage D: Continual Personalization
- Sequential user adaptation with replay + importance constraints + adapters.
Real-Time Inference Loop
1. Read synchronized sensor window.
2. Encode each modality.
3. Fuse and estimate intent/phase/confidence.
4. Generate assistance command.
5. Blend with user control via arbitration.
6. Apply safety constraints.
7. Execute and log for incremental update.
What To Build First (Practical Start)
1. Build the Data Layer and synchronization pipeline for Kaiwu.
2. Implement two-modality baseline first: kinematics + video.
3. Add intent and phase heads before full control policy.
4. Add arbitration and safety gate.
5. Add incremental adaptation block (adapters + replay + regularization).
6. Expand to full modalities (EMG, gaze, audio, pressure).

