# Human Input Sensors for Shared Autonomy and HRC

**Scope:** A shorter, decision-oriented review of the main sensor modalities for human input in shared autonomy. The lists below are sorted by approximate price/performance value inside each category, from best value to premium.

**How to read the shopping lists:**

- Value rank is qualitative and combines cost, robustness, latency, and ease of deployment.
- Lower cost-per-useful-performance is better.
- Prices are typical academic/commercial street prices in 2024–2026 and can vary by vendor and bundle.
- For the final recommendation ranking, I weight **40% performance/integration**, **40% innovation**, and **20% price**.

## 1) Best Overall Recommendation

If you want the strongest practical stack for shared autonomy research, start with:

- **Hand pose** for continuous user intent and fine manipulation
- **Eye gaze** for attention and target disambiguation
- **Force / pressure** for contact and manipulation state
- **Voice** for high-level commands and safety overrides
- **Optional EMG** only if you specifically need fatigue, effort, or muscle-activation cues

This ordering reflects the weighted score above: strong real-time performance and integration, high novelty for shared autonomy, and still acceptable cost.

## 2) EMG: Muscle Activity

**Best use:** effort estimation, tremor/fatigue detection, grasp intent, assistive control research.

| Value rank | Product                                                                            | Typical price | Why it is in the list                      |
| ---------- | ---------------------------------------------------------------------------------- | ------------: | ------------------------------------------ |
| 1          | [MyoWare / low-cost sEMG kits](https://www.sparkfun.com/myoware-2-muscle-sensor.html) |        $42.95 | Cheapest way to prototype EMG signals      |
| 2          | [OpenBCI EMG-based setups](https://shop.openbci.com/)                                 |     $200–800 | Flexible and good for research experiments |
| 3          | [Delsys Trigno Wireless](https://delsys.com/shop/)                                    |    $40/sensor | Best overall research-grade choice         |

**Short recommendation:**

- Choose **MyoWare** if you want a cheap prototype.
- Choose **Delsys Trigno** if you want reliable research results.
- EMG is powerful but setup-heavy, so it is usually not the first sensor I would buy unless muscle intent is central to the thesis.

## 3) Eye Gaze: Attention and Target Selection

**Best use:** target inference, object selection, shared autonomy arbitration.

| Value rank | Product                                                                                | Typical price | Why it is in the list                         |
| ---------- | -------------------------------------------------------------------------------------- | ------------: | --------------------------------------------- |
| 1          | [GazePoint GP3](https://www.gazept.com/product/gazepoint-gp3-eye-tracker/)                |       $985.00 | Lowest-cost usable research eye tracker       |
| 2          | [Pupil Core](https://pupil-labs.com/products/core)                                        | $3,000–3,500 | Strong value for wearable and lab work        |
| 3          | [Tobii Pro Nano](https://www.tobii.com/products/eye-trackers/screen-based/tobii-pro-nano) | $3,500–4,500 | Best balance of accuracy, ease, and stability |

**Short recommendation:**

- **Tobii Pro Nano** is the best default choice for shared autonomy.
- **Pupil Core** is the better budget-research option if you want head-mounted flexibility.
- Eye gaze is one of the most valuable modalities because it directly captures attention before motion happens.

## 4) Motion Capture / Skeletal Tracking

**Best use:** full-body pose, arm motion, gesture tracking, teleoperation, training data.

| Value rank | Product                                                                                                                                                                                                                                                                                                                                                                | Typical price | Why it is in the list                       |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------: | ------------------------------------------- |
| 1          | [MediaPipe Pose / Hands + webcam](https://ai.google.dev/edge/mediapipe/solutions/vision)                                                                                                                                                                                                                                                                                  |       $0–200 | Best cost/performance for quick prototyping |
| 2          | [Intel RealSense](https://www.amazon.com/Intel-RealSense-Depth-Camera-D415/dp/B07JVGRQZT/ref=pd_sbs_d_sccl_1_2/133-6211871-2568903?pd_rd_w=X5QZo&content-id=amzn1.sym.aa738fbd-ad05-4d11-aae2-04b598db6305&pf_rd_p=aa738fbd-ad05-4d11-aae2-04b598db6305&pf_rd_r=S5NKK74E084ZK21RF99P&pd_rd_wg=0YyQ0&pd_rd_r=bdf02acd-0996-4245-90cf-ab168a230219&pd_rd_i=B07JVGRQZT&th=1) |          $400 | Low-cost depth-based skeletal tracking      |

**Short recommendation:**

- Start with **MediaPipe** if you need a fast and cheap baseline.
- Use **Azure Kinect** or **RealSense** if you need depth-based body pose.
- Use **Vicon** only when accuracy is more important than budget.

## 5) BCI / EEG

**Best use:** special populations, research on non-muscular control, niche assistive applications.

| Value rank | Product                                                                            | Typical price | Why it is in the list                     |
| ---------- | ---------------------------------------------------------------------------------- | ------------: | ----------------------------------------- |
| 1          | [OpenBCI Cyton](https://shop.openbci.com/products/cyton-biosensing-board-8-channel)  |         $1250 | Lowest-cost serious EEG research platform |
| 2          | [Emotiv EPOC X](https://www.emotiv.com/epoc-x/)                                       |        $1,199 | Easy to buy and quick to start            |

**Short recommendation:**

- EEG is innovative, but the cost/performance is usually worse than gaze, hand pose, or force for shared autonomy.
- I would only choose EEG if you need a strong novelty angle or must work with users who cannot provide conventional motor input.

## 6) Voice / Speech

**Best use:** high-level commands, corrections, emergency stop, low-bandwidth context.

| Value rank | Product                                                                                                                                 | Typical price | Why it is in the list                                |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------- | :-----------: | ---------------------------------------------------- |
| 1          | [Vosk](https://alphacephei.com/vosk/) / [Silero](https://github.com/snakers4/silero-models) / [Whisper local](https://github.com/openai/whisper) |  open source  | Cheapest and privacy-friendly if you can run locally |
| 2          | [Azure Speech](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/)                                      | pay as you go | Good accuracy and easy integration                   |
| 3          | [Google Speech-to-Text](https://cloud.google.com/speech-to-text/pricing)                                                                   | pay as you go | Strong accuracy, cloud dependence                    |
| 4          | [AWS Transcribe / premium cloud stacks](https://aws.amazon.com/transcribe/pricing/)                                                        | pay as you go | Strong but usually not needed first                  |

**Short recommendation:**

- Local speech models are the best value if you can tolerate some tuning.
- Cloud speech is better when you want fast deployment and don’t mind network dependence.

## 7) Hand Pose / Hand Tracking

**Best use:** grasp intention, finger gestures, direct manipulation cues, shared autonomy.

| Value rank | Product                                                                                                                                                   | Typical price | Why it is in the list                         |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------: | --------------------------------------------- |
| 1          | [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)                                                                     |   $0 + webcam | Best value by far for real-time hand tracking |
| 2          | [Leap Motion Controller 2 / Ultraleap](https://eu.robotshop.com/fr/products/camera-suivi-des-mains-stereo-ultraleap-3di?qd=84276e36107f6ff9c95ba5185abda147) |          $285 | Very good precision and easy setup            |
| 3          | [Rokoko smart gloves](https://www.rokoko.com/products/smartgloves)                                                                                           |         $1795 | Premium, specialized, and expensive           |

**Short recommendation:**

- **MediaPipe Hands** is the best overall value for shared autonomy.
- **Leap Motion / Ultraleap** is the best low-cost hardware option if you want dedicated hand sensing.
- This category is one of the strongest for your thesis because it is cheap, real-time, and very expressive.
