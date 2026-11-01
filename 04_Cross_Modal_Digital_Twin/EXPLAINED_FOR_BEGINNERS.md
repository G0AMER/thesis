# The work explained in simple terms

## The one-sentence version

This study tests whether a computer model trained on people moving in a virtual
factory workstation can recognize the same kinds of actions when real workers
perform them, and how much real-world labelled data helps when direct transfer
is not good enough.

## Why do this work?

An industrial robot may need to know what a nearby person is doing. For
example, recognizing that a worker is picking up a component could help the
robot choose when or how to assist.

To train a recognition system, researchers need examples of workers performing
each action. Recording and labelling many workers in a real factory can take
time, cost money, and interrupt work. A digital twin, which is a virtual copy
of a workstation, can provide a place to record examples without doing every
experiment in the physical workcell.

There is a catch: people do not necessarily move the same way in a virtual
workstation as they do around real equipment. Their posture, timing, reach, and
interaction with objects can change. The study measures how much this
virtual-to-physical difference matters.

## What are the two datasets?

The study uses two related datasets:

| Dataset | What it contains | Its role in the experiment |
|---|---|---|
| **InHARD-DT** | 4,696 motion clips from 12 people in 26 sessions, recorded in a virtual-reality copy of the workstation | The virtual **source** data used to train the starting model |
| **InHARD** | Approximately 4,803 motion clips from 16 workers at the physical workstation | The real **target** data on which transfer is tested |

Both datasets describe movement using the same type of motion-capture suit and
the same original 59-joint skeleton structure. They also use the same 13 action
categories. The main difference is therefore the setting and people, not a
change from one kind of sensor to another.

The model does not use all 59 joints. The processing step selects a
19-joint body representation, converts movement into fixed-length clips, and
normalizes the skeleton position and orientation. This helps the model focus
on movement patterns rather than, for example, whether a person stood slightly
closer to the motion-capture origin.

## What does the model see?

The input is a short sequence of three-dimensional body-joint positions. You
can picture a moving stick figure: each joint is a point, and the connections
between joints describe the body's structure. A clip contains multiple frames,
so the model can use both **where the body parts are** and **how they move
over time**.

The study uses **CTR-GCN** as its recognition model. In plain terms, this is a
neural network designed to learn patterns from connected body joints and their
movement through a sequence. Every transfer method uses the same CTR-GCN
backbone, the same 19-joint input, and the same training budget. This makes
the comparison more meaningful: a method is not given a larger model just to
make it look better.

## What does “source” and “target” mean?

These words describe the direction of transfer:

- **Source domain:** the virtual InHARD-DT recordings.
- **Target domain:** the physical InHARD recordings.

The central question is whether knowledge learned from the source can be used
in the target. “Domain” here simply means the setting in which the recordings
were made.

## How was the test arranged?

The 16 real InHARD workers are divided into four groups, each containing four
workers. The experiment takes one group at a time as the **test group**. The
other 12 workers form the pool that can be used for adaptation.

The test workers are held out: their labels are not used to train or choose a
model. This matters because the test is meant to answer whether a model can
handle workers it has not been allowed to learn from.

The source data comes from all 12 InHARD-DT subjects. The four-fold evaluation
is repeated with three random seeds, giving 12 fold-and-seed runs for the
reported summaries. A different seed can affect random parts of training or
sampling; repeating the experiment helps show whether a result is consistent
or varies across runs.

## The two main experiments

### 1. Zero-shot transfer: no labelled real examples for adaptation

“Zero-shot” means the model receives **no labelled real InHARD clips** for
adaptation. It is trained on the virtual source data and tested on held-out
physical workers. Some methods can see unlabelled target clips, but they do not
get the correct action names for those clips.

The study compares several ways of handling the difference between the virtual
and real data:

- **Source-only:** train on labelled virtual examples and transfer the model
  directly to the real workers.
- **MMD and CORAL:** encourage the feature patterns from the virtual and real
  data to have more similar distributions. A “feature” is an internal
  representation the model builds from the motion.
- **DANN:** train the model to recognize actions while making its internal
  features less able to reveal whether a clip came from the virtual or real
  domain.
- **FixMatch:** use confident predictions on unlabelled real clips as
  temporary labels during training.
- **PI-CDA:** test the project's physics-informed approach, which combines
  feature alignment with a consistency term based on skeleton topology.

The purpose of this comparison is not to assume that one method must work. It
checks whether these different strategies improve recognition on unseen
physical workers.

### 2. Few-shot adaptation: add a small number of labelled real examples

“Few-shot” means the model is allowed to use a small labelled set from the
physical target domain. The experiment tries budgets of **1, 5, 10, 20, or 50
clips per action class**. With 13 classes, that corresponds to 13, 65, 130,
260, or 650 labelled clips in the support set for a fold.

The support clips come from the 12-worker adaptation pool. The four held-out
test workers remain separate. The experiment compares seven ways of starting
and adapting the model, including training from scratch, using the virtual
model, combining virtual and real data, and using a prototype-based objective.

A **prototype** is a representative point for a class in the model's learned
feature space. In simple terms, the model compares a new example with the
typical pattern it has learned for each action class. The label-budget table
reports results for joint prototype fine-tuning, abbreviated
`joint_proto_ft`. The separate comparison across all seven strategies finds
that joint training with a prototype objective has the highest mean at the
one-clip-per-class budget.

## How should I read the scores?

The paper reports **accuracy** and **macro-F1**:

- **Accuracy** is the fraction of all tested clips assigned the correct class.
- **Macro-F1** calculates an F1 score for each action class and then averages
  those class scores equally. This prevents common or easier classes from
  dominating the summary.

The paper emphasizes macro-F1 because some actions are harder to recognize
than others. For example, a high overall accuracy could hide poor performance
on a less frequent or more confusing action.

Each reported “±” value is the standard deviation across the 12 fold-and-seed
runs. It describes how much the score varied among those runs; it is not a
guarantee about performance on every future workcell.

## What did the zero-shot experiment find?

The following are the reported accuracy and macro-F1 results:

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| InHARD-DT in-domain reference | 75.47 ± 3.54% | 73.32 ± 4.04% |
| Source-only transfer | 50.16 ± 2.97% | 40.59 ± 2.93% |
| DANN | 46.71 ± 3.89% | 38.17 ± 3.00% |
| CORAL | 48.17 ± 4.00% | 38.88 ± 3.10% |
| MMD | 49.21 ± 3.18% | 39.66 ± 3.00% |
| PI-CDA | 48.35 ± 2.79% | 39.34 ± 2.50% |
| FixMatch | 50.75 ± 2.91% | 40.54 ± 2.93% |
| Target oracle | 82.67 ± 2.38% | 75.22 ± 4.04% |
| Source + target | 83.95 ± 1.98% | 75.89 ± 3.20% |

The main takeaway is that the tested unsupervised adaptation methods did not
reliably improve on source-only transfer. PI-CDA's macro-F1 was 1.25 points
below source-only; the paper reports that this paired difference was not
statistically significant (`p = 0.105`). The experiment therefore does not
show that the tested methods solve the virtual-to-physical transfer problem
without labelled real data.

The **target oracle** is a reference trained with target labels available
across the target training pool. It shows what the system can achieve with
much more supervision than a few-shot setup. It is a comparison point, not a
claim that such complete labelling is easy or required in deployment.

The **source + target** result also uses labelled target training data. It
tests whether adding source data helps when target labels are available. Its
macro-F1 is close to the oracle in this experiment; the paper says the two are
statistically indistinguishable in the available paired comparison.

The InHARD-DT in-domain score is another reference, but it is measured within
the virtual domain. It should not be read as a result on physical workers.

## What happened when real labels were added?

For joint prototype fine-tuning, performance rose as the number of labelled
target clips increased:

| Labelled target clips per class | Accuracy | Macro-F1 |
|---:|---:|---:|
| 1 | 50.35 ± 4.04% | 41.22 ± 4.38% |
| 5 | 57.91 ± 3.44% | 49.43 ± 3.74% |
| 10 | 61.99 ± 3.14% | 54.52 ± 3.35% |
| 20 | 67.17 ± 3.16% | 60.20 ± 3.96% |
| 50 | 74.11 ± 2.36% | 66.27 ± 3.46% |
| Target oracle reference | 82.67 ± 2.38% | 75.22 ± 4.04% |

At 50 clips per class, macro-F1 reaches 66.27%, compared with 40.59% for
zero-shot source-only transfer and 75.22% for the target oracle. That is a
large improvement over zero-shot transfer, but it does not reach the oracle.
It also does not by itself prove that the digital-twin initialization caused
all of the improvement, because the method also receives labelled physical
examples.

The separate initialization comparison addresses that question by comparing
seven strategies under the same few-shot budgets. Joint prototype training
has the highest reported mean at one clip per class. From five clips per class
onward, joint prototype fine-tuning is the numerical leader, but its mean
advantage over few-shot source-only is modest. The plots mark the highest
mean; that mark does not mean the difference is statistically significant.

## What is the overall conclusion?

The digital twin is useful for producing labelled virtual training data, but
those data do not transfer perfectly to physical workers. In this benchmark,
the tested unsupervised methods did not close the gap reliably. A small
labelled sample from the physical setting improved the results as its size
increased.

The careful conclusion is that digital-twin pretraining may give a useful
starting point for adaptation, especially when real labels are scarce. The
study does **not** show that a digital twin can replace physical data, that
PI-CDA solves zero-shot transfer, or that the system is ready to control a
robot safely.

## What the study does not establish

- It tests one virtual-to-physical dataset pair, InHARD-DT to InHARD.
- It uses one main recognition backbone, CTR-GCN.
- It evaluates action recognition on held-out workers, not a robot operating
  in a live closed-loop collaboration task.
- Its few-shot results depend on the chosen support-set sampling and class
  distribution.
- It does not establish that the same results will hold for other factories,
  robots, sensors, or action categories.

The next step is to test the approach on additional workcells and sensing
conditions, and to study whether recognition performance leads to safe,
well-calibrated robot assistance.

## A note about older project results

The project contains an archived earlier version of the PI-CDA work. The
project README states that its high reported results were produced using
synthetic sine-wave data and random numbers rather than the InHARD datasets.
Those archived numbers and figures are **not** the results described here and
should not be used as evidence for the current paper. The current explanation
above follows the real-data benchmark reported in the manuscript.

## Short glossary

| Term | Plain-language meaning |
|---|---|
| Digital twin | A virtual copy of a real workstation or process |
| Skeleton sequence | A series of body-joint positions over time |
| Source domain | The setting that supplies the initial training data; here, InHARD-DT |
| Target domain | The setting where transfer is tested; here, physical InHARD |
| Domain gap | Differences between the virtual and physical recordings |
| Zero-shot transfer | Testing on the target setting without labelled target examples for adaptation |
| Few-shot adaptation | Updating a model with a small labelled target sample |
| Subject-disjoint | The people in the test fold are kept separate from those used for adaptation |
| Macro-F1 | An average that gives each action class equal weight |
| Target oracle | A reference trained with target labels available across the target training pool |
