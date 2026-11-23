const pptxgen = require("pptxgenjs");
const React = require("react");
const RDS = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require("/home/g0amer/.claude/skills/synced/82ab380d-449b-4c1a-9efc-43946a06f306_bdc6918c-08ee-4858-8199-65c9ca844fb6/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "deck.pptx";
const IMG = __dirname + "/img/";

const THEME = {
  name: "HRC Year One",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2631", lt1: "FFFFFF", dk2: "0B3C49", lt2: "EDF3F4",
    accent1: "0E7C86", accent2: "E8952F", accent3: "5B6C7A",
    accent4: "C0392B", accent5: "4E9F76", accent6: "A9B6BF",
    hlink: "0E7C86", folHlink: "5B6C7A",
  },
};
const H = THEME.colors; // hex, for chart options that need hex

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "PhD Year 1 Progress Review";
pres.author = "Ameur Gargouri";
const C = pres.SchemeColor;

// ---------- icons ----------
async function icon(Comp, hex, size = 256) {
  const svg = RDS.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size }));
  const buf = await sharp(Buffer.from(svg)).resize(size, size).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.55, w: 8.8, h: 1.5, fontSize: 32, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.2, w: 8.8, h: 1.6, fontSize: 15, color: C.accent6, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.5, 0.5, 0.5],
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.5, y: 0.28, w: 9, h: 0.28, fontSize: 11, bold: true, color: C.accent1, charSpacing: 1.5, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.55, w: 9, h: 0.6, fontSize: 24, bold: true, color: C.text2, fontFace: THEME.headFontFace, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: "A. Gargouri  ·  PhD year 1 review  ·  ATISP & ESME Research Lab", options: { x: 0.5, y: 5.28, w: 6.5, h: 0.22, fontSize: 9, color: C.accent3, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.28, w: 0.5, h: 0.22, fontSize: 9, color: C.accent3, align: "right", margin: 0 },
});

function content(section, kicker, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(kicker.toUpperCase(), { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  return s;
}
const T = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, fontSize: 14, color: C.text1, margin: 0, valign: "top" }, o));
function card(s, x, y, w, h, fill, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.background2 }, line: { type: "none" }, objectName: name });
}
function circleIcon(s, data, x, y, d, fill, name) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" }, objectName: name + " circle" });
  const p = d * 0.25;
  s.addImage({ data, x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, altText: name, objectName: name + " icon" });
}
function stat(s, x, y, w, big, label, color, bigSize = 34) {
  T(s, big, { x, y, w, h: 0.62, fontSize: bigSize, bold: true, color: color || C.accent1, fontFace: THEME.headFontFace, valign: "bottom" });
  T(s, label, { x, y: y + 0.66, w, h: 0.5, fontSize: 12, color: C.accent3 });
}
const chartBase = () => ({
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  catAxisLabelColor: H.accent3, valAxisLabelColor: H.accent3, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
  dataLabelFontSize: 10, dataLabelColor: H.dk1, titleFontSize: 12, titleColor: H.dk2,
  valGridLine: { color: "DDE4E8", size: 0.5 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false,
});

(async () => {
  const I = {};
  const want = {
    robot: fa.FaRobot, eye: fa.FaEye, brain: fa.FaBrain, sliders: fa.FaSlidersH, check: fa.FaClipboardCheck,
    heart: fa.FaHeartbeat, shield: fa.FaShieldAlt, layers: fa.FaLayerGroup, cube: fa.FaCube, book: fa.FaBookOpen,
    db: fa.FaDatabase, flag: fa.FaFlag, search: fa.FaSearch, balance: fa.FaBalanceScale, users: fa.FaUsers,
    bolt: fa.FaBolt, memory: fa.FaMemory, redo: fa.FaRedo, filter: fa.FaFilter, tags: fa.FaTags,
    route: fa.FaRoute, flask: fa.FaFlask, file: fa.FaFileAlt, handshake: fa.FaHandshake, lock: fa.FaLock,
  };
  for (const [k, v] of Object.entries(want)) I[k] = await icon(v, "FFFFFF");

  // ===== 1. Title =====
  pres.addSection({ title: "Opening" });
  let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
  circleIcon(s, I.robot, 0.6, 0.55, 0.75, C.accent1, "Robot");
  T(s, "PHD PROGRESS REVIEW  ·  TRANSITION FROM YEAR 1 TO YEAR 2", { x: 1.55, y: 0.75, w: 7.8, h: 0.35, fontSize: 12, bold: true, color: C.accent2, charSpacing: 1.5, valign: "middle" });
  s.addText("Human-Robot Collaboration: Incremental Learning and Behavioural Adaptation", { placeholder: "title" });
  s.addText([
    { text: "Ameur Gargouri", options: { bold: true, color: C.background1, fontSize: 18, breakLine: true } },
    { text: "Supervisors: Mohamed Ksantini, University of Sfax  ·  Mohamed Karray, ESME Research Lab", options: { breakLine: true } },
    { text: "ATISP Laboratory, ENET'Com, University of Sfax  ·  ESME Research Lab, Ivry-sur-Seine" },
  ], { placeholder: "body" });
  s.addNotes("Target time: 30 seconds. Good morning, members of the jury. I am Ameur Gargouri, and I will present the work I carried out during the first year of my thesis on human-robot collaboration, incremental learning and behavioural adaptation, supervised by Mohamed Ksantini at ATISP and Mohamed Karray at the ESME Research Lab. I will first recall the research question, then present the four studies of this year with their results and their limits, and I will close with the plan for the second year.");

  // ===== 1b. Plan =====
  s = content("Opening", "Plan", "Outline of the presentation");
  const outline = [
    ["Thesis framing", "Research question, year 1 overview, state of the art and datasets", C.accent3],
    ["Study 1  ·  Task recognition", "Observing the operator through physiological fusion", C.accent1],
    ["Study 2  ·  Danger detection", "Safety from wearable sensors on the operator", C.accent1],
    ["Study 3  ·  FLAIR", "Learning new operators without forgetting", C.accent1],
    ["Study 4  ·  Digital twin", "Transfer from a digital twin to a real workstation", C.accent1],
    ["Year 2 plan", "From recognising the operator to adapting the robot", C.accent2],
  ];
  outline.forEach(([h, d, col], i) => {
    const x = 0.5 + (i % 2) * 4.6, y = 1.35 + Math.floor(i / 2) * 1.25;
    card(s, x, y, 4.4, 1.08, C.background2, h + " card");
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.22, w: 0.64, h: 0.64, fill: { color: col }, line: { type: "none" }, objectName: "Step " + (i + 1) });
    T(s, String(i + 1), { x: x + 0.2, y: y + 0.22, w: 0.64, h: 0.64, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle", fontFace: THEME.headFontFace });
    T(s, h, { x: x + 1.05, y: y + 0.17, w: 3.2, h: 0.35, fontSize: 15, bold: true, color: C.text2, valign: "middle" });
    T(s, d, { x: x + 1.05, y: y + 0.54, w: 3.2, h: 0.42, fontSize: 12, color: C.text1 });
  });
  s.addNotes("Target time: 20 seconds. The presentation follows six parts. I first frame the thesis with the research question, an overview of the year, the state of the art and the datasets. I then present the four studies of the year in order, task recognition, danger detection, incremental learning with FLAIR and transfer from a digital twin. I finish with the plan for the second year.");

  // ===== 2. Research question =====
  pres.addSection({ title: "Thesis framing" });
  s = content("Thesis framing", "Research question", "A cobot that adapts to each operator");
  card(s, 0.5, 1.3, 9, 0.85, C.background2, "Question card");
  T(s, "How can a collaborative robot learn, incrementally and without reprogramming, the preferences, strategies and working style of each operator, while remaining safe?", { x: 0.75, y: 1.38, w: 8.5, h: 0.7, fontSize: 15, italic: true, color: C.text2, valign: "middle" });
  const axes = [
    [I.eye, "A1  Observe", "Model the operator: state, actions, habits and timing", C.accent1],
    [I.brain, "A2  Learn", "Incremental learning from each new operator without forgetting", C.accent2],
    [I.sliders, "A3  Adapt", "Generate robot behaviour matched to the operator's style", C.accent3],
    [I.check, "A4  Validate", "Measure fluency and alignment of the collaboration", C.accent5],
  ];
  axes.forEach(([ic, h, d, col], i) => {
    const x = 0.5 + i * 2.33;
    card(s, x, 2.45, 2.1, 2.45, C.background2, h + " card");
    circleIcon(s, ic, x + 0.2, 2.65, 0.62, col, h);
    T(s, h, { x: x + 0.2, y: 3.4, w: 1.75, h: 0.35, fontSize: 15, bold: true, color: C.text2 });
    T(s, d, { x: x + 0.2, y: 3.8, w: 1.75, h: 1.0, fontSize: 12, color: C.text1 });
  });
  s.addNotes("Target time: 1 minute. The thesis asks how a collaborative robot can learn, step by step and without being reprogrammed, the preferences and the working style of each operator, while remaining safe. The subject is organised in four axes. The first axis observes the operator, meaning the state, the actions and the habits. The second learns incrementally, so that each new operator is integrated without forgetting the previous ones. The third adapts the robot behaviour to the operator style, and the fourth validates the collaboration with objective fluency metrics. Safety is a constraint that runs across all four axes. This first year concentrated on the perception and learning foundations, that is axes one and two together with safety.");

  // ===== 3. Year 1 at a glance =====
  s = content("Thesis framing", "Year 1 at a glance", "Four studies, four papers, one coherent foundation");
  const sc = [
    ["1", "published", "FLAIR, KES 2026, Procedia CS", C.accent5],
    ["1", "accepted", "Task recognition, ICAART 2027", C.accent1],
    ["2", "under review", "Danger detection, JIRS\nDigital twin to real transfer, journal", C.accent2],
  ];
  sc.forEach(([n, l, d, col], i) => {
    const w = 2.88, x = 0.5 + i * (w + 0.225);
    card(s, x, 1.35, w, 1.55, C.background2, l + " card");
    T(s, n, { x: x + 0.2, y: 1.42, w: 0.6, h: 0.7, fontSize: 40, bold: true, color: col, fontFace: THEME.headFontFace, valign: "middle" });
    T(s, l, { x: x + 0.75, y: 1.55, w: w - 0.8, h: 0.45, fontSize: 15, bold: true, color: C.text2, valign: "middle" });
    T(s, d, { x: x + 0.2, y: 2.15, w: w - 0.35, h: 0.65, fontSize: 11, color: C.text1 });
  });
  // timeline
  s.addShape(pres.shapes.LINE, { x: 0.7, y: 3.55, w: 8.6, h: 0, line: { color: C.accent6, width: 2 }, objectName: "Timeline axis" });
  const tl = [
    ["Nov 2025", "Machines paper published, from my engineering final-year project at ENIS"],
    ["Jan – Apr 2026", "State of the art: six reports on incremental learning and HRC datasets"],
    ["Early 2026", "Task recognition study on multimodal operator data"],
    ["Mid 2026", "Danger detection and FLAIR, KES 2026 talk"],
    ["Sep – Nov 2026", "Digital twin study on real InHARD data submitted, ICAART acceptance"],
  ];
  tl.forEach(([d, t], i) => {
    const x = 0.5 + i * 1.84;
    s.addShape(pres.shapes.OVAL, { x: x + 0.12, y: 3.44, w: 0.22, h: 0.22, fill: { color: i === 4 ? C.accent2 : C.accent1 }, line: { color: C.background1, width: 2 }, objectName: "Milestone " + (i + 1) });
    T(s, d, { x, y: 3.78, w: 1.7, h: 0.3, fontSize: 12, bold: true, color: C.text2 });
    T(s, t, { x, y: 4.08, w: 1.7, h: 1.0, fontSize: 11, color: C.text1 });
  });
  s.addNotes("Target time: 1 minute 30. This slide summarises the year. Four studies led to four papers: FLAIR is published in Procedia Computer Science after my presentation at KES 2026, the task recognition paper has just been accepted at ICAART 2027, the danger detection paper is under review at the Journal of Intelligent and Robotic Systems, and the digital twin paper has just been submitted and is also under review in a journal. The year started with the publication in Machines of the work from my end of studies project in the computer engineering cycle at the National Engineering School of Sfax, then a structured state of the art from January to April, documented in six reports on incremental learning, robotic lifelong learning and human-robot collaboration datasets. The task recognition study came first and taught me a lot about rigorous evaluation. Danger detection and FLAIR followed in mid 2026, and since September I have worked on transfer from a digital twin to a real workstation.");

  // ===== 4. State of the art =====
  s = content("Thesis framing", "State of the art", "Four gaps identified in the literature review");
  const gaps = [
    [I.redo, "Forgetting across operators", "Continual learning methods rely on regularisation or replay alone, and are rarely tested on sequences of human operators."],
    [I.shield, "Safety without force sensors", "Collision detection relies on force or torque sensors, cameras or shallow models, with no systematic study of deep models on wearable inertial data."],
    [I.heart, "Operator state evaluation", "Physiological studies report a strong operator effect, yet fusion is assumed to help rather than tested under one protocol."],
    [I.cube, "Cost of real labels", "Digital twins generate labelled data cheaply, but whether a twin-trained recogniser transfers to real workers is unclear."],
  ];
  gaps.forEach(([ic, h, d], i) => {
    const x = 0.5 + (i % 2) * 4.6, y = 1.35 + Math.floor(i / 2) * 1.85;
    card(s, x, y, 4.4, 1.65, C.background2, h + " card");
    circleIcon(s, ic, x + 0.22, y + 0.25, 0.6, C.accent1, h);
    T(s, h, { x: x + 1.0, y: y + 0.22, w: 3.2, h: 0.35, fontSize: 15, bold: true, color: C.text2 });
    T(s, d, { x: x + 1.0, y: y + 0.6, w: 3.2, h: 0.95, fontSize: 12, color: C.text1 });
  });
  s.addNotes("Target time: 1 minute 15. The state of the art carried out between January and April 2026 led to four gaps, and each gap gave one study. Continual learning methods usually rely either on regularisation or on replay, and they are rarely evaluated on a stream of human operators, which is exactly the setting of a cobot that meets new people. For safety, collision detection mostly relies on force and torque sensors, on cameras or on shallow learning, and deep models applied to wearable inertial data had not been compared systematically. For the operator state, physiological studies all report a strong effect of the individual, yet multimodal fusion is generally assumed to help rather than tested. Finally, digital twins promise cheap labelled data, but it is not clear whether a recogniser trained in a virtual workstation transfers to real workers.");

  // ===== 5. Data foundation =====
  s = content("Thesis framing", "Experimental foundation", "Reproducible pipelines on four public datasets");
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 12 } });
  const rows = [
    [hdr("Dataset"), hdr("Study"), hdr("Participants"), hdr("Signals"), hdr("Processed")],
    ["MultiPhysio-HRC", "Task recognition", "52 operators", "EEG, ECG, EDA, EMG, respiration", "5,640 windows, 100 %"],
    ["DASIG", "Danger detection", "60 subjects", "Wearable inertial units, 65 channels", "179 of 180 recordings"],
    ["HARMONIC", "Incremental learning", "10 operators in sequence", "Joystick, robot joints, end effector", "447 of 447 trials"],
    ["InHARD and InHARD-DT", "Twin to real transfer", "16 real, 12 virtual", "Full-body skeleton, 13 actions", "~9,500 clips"],
  ].map((r, i) => i === 0 ? r : r.map((c) => ({ text: c, options: { fontSize: 12, color: C.text1, fill: { color: i % 2 ? C.background1 : C.background2 } } })));
  s.addTable(rows, { x: 0.5, y: 1.35, w: 9, colW: [1.85, 1.65, 1.65, 2.25, 1.6], rowH: 0.48, border: { type: "solid", pt: 0.5, color: "DDE4E8" }, valign: "middle", margin: 0.06 });
  circleIcon(s, I.db, 0.5, 4.2, 0.55, C.accent2, "Data");
  T(s, "Every pipeline is scripted, versioned and audited before modelling, so each result in this talk can be regenerated from raw data.", { x: 1.25, y: 4.2, w: 8.2, h: 0.55, fontSize: 13, color: C.text2, valign: "middle" });
  s.addNotes("Target time: 1 minute. All the studies rest on public datasets, which makes them reproducible and comparable with other teams. MultiPhysio-HRC provides five physiological signals for fifty two operators after cleaning. DASIG provides wearable inertial data from sixty subjects with normal and dangerous movements. HARMONIC provides shared autonomy recordings with a robotic arm, from which ten operators form a sequential benchmark. InHARD and its digital twin provide full-body skeleton recordings of the same assembly workstation, in the real cell and in virtual reality. For each dataset I built a scripted preprocessing pipeline, and the only failure, one DASIG recording, comes from a malformed raw file.");

  // ===== 6. WP1 method =====
  pres.addSection({ title: "Study 1: Task recognition" });
  s = content("Study 1: Task recognition", "Study 1  ·  Observe the operator", "Recognising the task condition from physiology");
  const steps = [
    [I.tags, "Label", "21 protocol conditions mapped to 5 classes by a deterministic rule"],
    [I.search, "Audit", "257 exported columns reduced to 128 physiological features, removing identifiers and duplicates"],
    [I.layers, "Fuse", "Five modalities merged into one tabular representation"],
    [I.balance, "Compare", "Five detectors and a modality ablation under one shared protocol"],
  ];
  steps.forEach(([ic, h, d], i) => {
    const x = 0.5 + i * 2.33;
    circleIcon(s, ic, x + 0.72, 1.4, 0.65, C.accent1, h);
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 1.6, y: 1.6, w: 0.5, h: 0.25, fill: { color: C.accent6 }, line: { type: "none" }, objectName: "Arrow " + (i + 1) });
    T(s, h, { x, y: 2.2, w: 2.1, h: 0.35, fontSize: 15, bold: true, color: C.text2, align: "center" });
    T(s, d, { x: x + 0.1, y: 2.6, w: 1.9, h: 1.0, fontSize: 12, color: C.text1, align: "center" });
  });
  card(s, 0.5, 3.85, 9, 1.15, C.background2, "Classes card");
  T(s, [
    { text: "Five classes  ", options: { bold: true, color: C.text2 } },
    { text: "cognitive load, high stress, industrial task, low load, other", options: { breakLine: true } },
    { text: "Data  ", options: { bold: true, color: C.text2 } },
    { text: "MultiPhysio-HRC, 52 operators, 60 s windows with 50 % overlap, battery disassembly with a cobot, VR stress and cognitive tests" },
  ], { x: 0.75, y: 3.98, w: 8.5, h: 0.9, fontSize: 13, paraSpaceAfter: 6, valign: "middle" });
  s.addNotes("Target time: 1 minute 15. The first study belongs to the observe axis: can a robot recognise what condition the operator is in from physiological signals. I used MultiPhysio-HRC, in which operators perform cognitive tests, virtual reality stress scenarios and a real battery disassembly with a cobot. The protocol has four steps. I first defined a deterministic rule that maps the twenty one protocol conditions to five classes, so the labels are fully reproducible. I then audited the feature export, which removed recording identifiers and duplicated EEG columns and left one hundred and twenty eight physiological features. The five modalities are fused into one representation, and finally five detectors are compared, together with an ablation that tests whether fusion really helps.");

  // ===== 7. WP1 results =====
  s = content("Study 1: Task recognition", "Study 1  ·  Results", "Fusion helps every detector, operators stay hard");
  s.addChart(pres.charts.BAR, [{ name: "Macro-F1 gain of fusion", labels: ["Lowest gain", "Highest gain"], values: [0.047, 0.186] }], Object.assign(chartBase(), {
    x: 5.1, y: 1.3, w: 4.4, h: 2.35, barDir: "bar", chartColors: [H.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.000",
    valAxisHidden: true, valGridLine: { style: "none" }, showTitle: true, title: "Fused input vs best single modality, macro-F1", showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 0.22,
  }));
  stat(s, 0.5, 1.3, 2.2, "0.95", "accuracy, XGBoost on fused features");
  stat(s, 2.75, 1.3, 2.2, "0.91", "macro-F1 over five classes");
  card(s, 0.5, 2.75, 4.4, 0.9, C.background2, "Status card");
  circleIcon(s, I.flag, 0.68, 2.92, 0.55, C.accent5, "Accepted");
  T(s, "Accepted at ICAART 2027, paper 340, camera-ready prepared", { x: 1.38, y: 2.85, w: 3.4, h: 0.7, fontSize: 13, bold: true, color: C.text2, valign: "middle" });
  card(s, 0.5, 3.9, 9, 1.1, "FDF1E3", "Limit card");
  T(s, [
    { text: "Honest limit  ", options: { bold: true, color: C.accent2 } },
    { text: "the 80/20 split shares operators between train and test. Under leave-one-subject-out evaluation XGBoost falls to 0.50 macro-F1, against 0.63 accuracy for the majority class. These scores are a within-subject ceiling, and generalising to new operators is the open problem that motivates the incremental learning axis." },
  ], { x: 0.75, y: 3.98, w: 8.5, h: 0.95, fontSize: 12, valign: "middle" });
  s.addNotes("Target time: 1 minute 30. On the fused features, XGBoost reaches an accuracy of 0.95 and a macro-F1 of 0.91 over the five classes. Macro-F1 is the average of the per-class F1 scores, so it is not dominated by the frequent classes. The ablation shows that fusing the modalities improves macro-F1 over the best single modality for every detector tested, by 0.047 to 0.186. The paper has been accepted at ICAART 2027. I want to be transparent about the limit: the hold-out split shares operators between training and test. When each operator is held out in turn, the macro-F1 of XGBoost drops to 0.50, so the operator effect dominates. This result is reported in the paper, and it is precisely why the thesis needs models that adapt to each new operator.");

  // ===== 8. WP2 =====
  pres.addSection({ title: "Study 2: Danger detection" });
  s = content("Study 2: Danger detection", "Study 2  ·  Safety", "Detecting danger from wearables, not force sensors");
  T(s, [
    { text: "Input  ", options: { bold: true, color: C.text2 } },
    { text: "65 inertial channels expanded on GPU to 195 with velocity and acceleration", options: { breakLine: true } },
    { text: "Models  ", options: { bold: true, color: C.text2 } },
    { text: "five 1D deep architectures, test-time augmentation", options: { breakLine: true } },
    { text: "Post-processing  ", options: { bold: true, color: C.text2 } },
    { text: "three-tap median filter and tuned threshold" },
  ], { x: 0.5, y: 1.35, w: 4.2, h: 1.55, fontSize: 13, paraSpaceAfter: 8 });
  stat(s, 0.5, 3.0, 2.0, "97.1 %", "dangers detected, ConvNeXt 1D", C.accent4, 30);
  stat(s, 2.6, 3.0, 2.1, "−87.5 %", "false alarms, for 1.2 pt of recall", C.accent5, 30);
  card(s, 0.5, 4.3, 4.2, 0.72, "FDF1E3", "Limit card");
  T(s, [{ text: "Limit  ", options: { bold: true, color: C.accent2 } }, { text: "5-fold stratified split on 69k windows, subject-wise rerun planned" }], { x: 0.65, y: 4.36, w: 3.95, h: 0.6, fontSize: 12, valign: "middle" });
  s.addChart(pres.charts.BAR, [{ name: "Macro-F1 (%)", labels: ["ConvNeXt 1D", "TCN", "MLP-Mixer 1D", "InceptionTime", "Transformer 1D"], values: [98.68, 98.62, 98.32, 96.22, 92.50] }], Object.assign(chartBase(), {
    x: 5.0, y: 1.3, w: 4.5, h: 3.75, barDir: "bar", chartColors: [H.accent1, H.accent6, H.accent6, H.accent6, H.accent6], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
    valAxisMinVal: 90, valAxisMaxVal: 100, valAxisHidden: true, valGridLine: { style: "none" }, catAxisOrientation: "maxMin", showTitle: true, title: "Macro-F1 (%), DASIG, 60 subjects", showLegend: false,
  }));
  s.addNotes("Target time: 1 minute 30. The second study addresses safety. The aim is to detect a dangerous state from inertial sensors worn by the operator, instead of expensive force or torque sensors. The sixty five raw channels are expanded on the GPU with velocity and acceleration, giving one hundred and ninety five channels, and five one-dimensional deep architectures are compared under identical conditions. ConvNeXt 1D and the temporal convolutional network are the best, with macro-F1 of 98.68 and 98.62 percent, and ConvNeXt detects 97.1 percent of the dangers. A simple temporal post-processing, a median filter with a tuned threshold, removes 87.5 percent of the false alarms for only 1.2 points of recall, which matters because each false alarm stops production. The evaluation is a stratified five-fold split on windows, so a subject-wise evaluation is planned for the revision. The paper is under review at the Journal of Intelligent and Robotic Systems.");

  // ===== 9. WP3 method =====
  pres.addSection({ title: "Study 3: FLAIR" });
  s = content("Study 3: FLAIR", "Study 3  ·  Learn incrementally", "FLAIR: new operators without forgetting old ones");
  T(s, "Setting: shared-autonomy assistive eating with a Kinova arm. The robot learns ten operators one after another and is evaluated on all operators seen so far.", { x: 0.5, y: 1.3, w: 9, h: 0.55, fontSize: 13, color: C.text1 });
  const comps = [
    [I.sliders, "Per-operator FiLM", "Feature modulation specific to each operator, warm-started from the closest one"],
    [I.memory, "Task-aware replay", "Buffer of past samples with dark-knowledge distillation, weighted by measured forgetting"],
    [I.lock, "Fisher + RetroBoost", "Online importance regularisation that strengthens as operators accumulate"],
    [I.layers, "Head blending", "Shared and operator-specific output heads mixed for each prediction"],
  ];
  comps.forEach(([ic, h, d], i) => {
    const x = 0.5 + i * 2.33;
    card(s, x, 2.05, 2.1, 2.95, C.background2, h + " card");
    circleIcon(s, ic, x + 0.2, 2.25, 0.62, i === 0 ? C.accent2 : C.accent1, h);
    T(s, h, { x: x + 0.2, y: 3.02, w: 1.75, h: 0.5, fontSize: 14, bold: true, color: C.text2 });
    T(s, d, { x: x + 0.2, y: 3.55, w: 1.75, h: 1.35, fontSize: 12, color: C.text1 });
  });
  s.addNotes("Target time: 1 minute 15. The third study is the core of the learn axis. When a neural network learns a new operator, it tends to lose what it knew about the previous ones, which is called catastrophic forgetting. I evaluated this on HARMONIC, where operators control a robotic arm in an assistive eating task: the model learns ten operators one after another and is tested on all of them after each step. FLAIR combines four mechanisms. Each operator gets its own feature modulation, initialised from the most similar previous operator. A replay buffer rehearses past samples with distillation and gives more weight to the operators being forgotten. An online Fisher regularisation protects the important parameters and becomes stronger as operators accumulate. Finally, shared and operator-specific output heads are blended.");

  // ===== 10. WP3 results =====
  s = content("Study 3: FLAIR", "Study 3  ·  Results", "Best accuracy and lowest forgetting");
  s.addChart(pres.charts.BAR, [{ name: "Final average R²", labels: ["FLAIR", "Joint training", "DER++", "A-GEM", "Online EWC", "EWC"], values: [0.690, 0.644, 0.612, 0.420, 0.415, 0.368] }], Object.assign(chartBase(), {
    x: 0.5, y: 1.3, w: 5.0, h: 3.7, barDir: "bar", chartColors: [H.accent2, H.accent3, H.accent6, H.accent6, H.accent6, H.accent6], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisHidden: true, valGridLine: { style: "none" }, catAxisOrientation: "maxMin", showTitle: true, title: "Final average R² after 10 operators", showLegend: false,
  }));
  stat(s, 5.9, 1.3, 1.7, "0.017", "average forgetting, 8.8× below DER++", C.accent2, 30);
  stat(s, 7.75, 1.3, 1.75, "94 %", "of operator 1 performance kept after 9 more", C.accent1, 30);
  stat(s, 5.9, 2.75, 1.7, "1.3 MB", "memory overhead", C.accent3, 30);
  card(s, 5.9, 4.1, 3.6, 0.9, C.background2, "Status card");
  circleIcon(s, I.flag, 6.05, 4.27, 0.55, C.accent5, "Published");
  T(s, "Presented at KES 2026, published in Procedia Computer Science", { x: 6.72, y: 4.15, w: 2.7, h: 0.8, fontSize: 12, bold: true, color: C.text2, valign: "middle" });
  s.addNotes("Target time: 1 minute 15. After the ten operators, FLAIR reaches a final average R squared of 0.690, the best among incremental methods, ahead of DER++ at 0.612, and it matches or exceeds the joint-training reference at 0.644, which retrains on all the data at each step. Its average forgetting is 0.017, almost nine times lower than DER++, and it keeps 94 percent of its performance on the first operator after learning nine more, for a memory overhead of about 1.3 megabytes. I presented this work at KES 2026 and it is published in Procedia Computer Science. Its limit is that the targets are robot commands rather than preferences, which is why the second year extends it towards learning from operator corrections.");

  // ===== 11. WP4 question =====
  pres.addSection({ title: "Study 4: Digital twin" });
  s = content("Study 4: Digital twin", "Study 4  ·  Transfer", "Can a digital twin replace real labels?");
  s.addImage({ path: IMG + "inhard_physical_capture.png", x: 0.5, y: 1.3, w: 4.85, h: 3.08, altText: "InHARD workstation: operator with a motion-capture suit next to a cobot", objectName: "InHARD photo" });
  T(s, "InHARD physical workstation and skeleton capture, Dallel et al.", { x: 0.5, y: 4.45, w: 4.85, h: 0.3, fontSize: 10, italic: true, color: C.accent3 });
  T(s, [
    { text: "Source  ", options: { bold: true, color: C.text2 } },
    { text: "InHARD-DT, the workstation replicated in virtual reality, 12 subjects", options: { breakLine: true } },
    { text: "Target  ", options: { bold: true, color: C.text2 } },
    { text: "InHARD, the physical workstation, 16 subjects, 13 assembly actions", options: { breakLine: true } },
    { text: "Model  ", options: { bold: true, color: C.text2 } },
    { text: "CTR-GCN skeleton network, 4 subject-disjoint folds × 3 seeds", options: { breakLine: true } },
    { text: "Gap  ", options: { bold: true, color: C.text2 } },
    { text: "same motion-capture suit, so the gap is behavioural: posture, timing, objects and label proportions" },
  ], { x: 5.65, y: 1.3, w: 3.85, h: 3.7, fontSize: 13, paraSpaceAfter: 10 });
  s.addNotes("Target time: 1 minute. The fourth study started in September and is still in progress. Annotating real recordings for every worker and every workstation is expensive, and a digital twin can produce labelled data cheaply. The question is whether a skeleton action recogniser trained in the virtual replica of the InHARD workstation works on the real one. Both datasets use the same motion-capture suit, so the difference between them is behavioural: posture, timing, the absence of physical parts in virtual reality, and different label proportions. I use the CTR-GCN skeleton network with four subject-disjoint folds and three seeds, so that every test operator is unseen during training and no target label is used for model selection.");

  // ===== 12. WP4 zero-shot =====
  s = content("Study 4: Digital twin", "Study 4  ·  Without real labels", "No adaptation method beats twin-only training");
  s.addChart(pres.charts.BAR, [{ name: "Macro-F1 (%)", labels: ["Source only", "FixMatch", "MMD", "TC-MMD", "CORAL", "DANN", "Real oracle"], values: [40.59, 40.54, 39.66, 39.34, 38.88, 38.17, 75.22] }], Object.assign(chartBase(), {
    x: 0.5, y: 1.3, w: 5.6, h: 3.7, barDir: "col", chartColors: [H.accent1, H.accent6, H.accent6, H.accent6, H.accent6, H.accent6, H.accent5], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
    valAxisMinVal: 0, valAxisMaxVal: 85, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 10, showTitle: true, title: "Macro-F1 (%) on unseen real operators", showLegend: false,
  }));
  stat(s, 6.4, 1.35, 3.1, "~35 pts", "gap between twin-only training and the real oracle", C.accent4, 30);
  T(s, "Five unsupervised adaptation methods, including my topology-consistent TC-MMD, stay within 2.5 points of source-only. The gap analysis shows the shift is conditional, not a change of class proportions, so aligning feature distributions cannot close it.", { x: 6.4, y: 2.6, w: 3.1, h: 2.4, fontSize: 12, color: C.text1 });
  s.addNotes("Target time: 1 minute 15. Without any real label, the model trained on the twin reaches 40.6 percent macro-F1 on unseen real operators, against 75.2 percent for a model trained on real labels, a gap of about thirty five points. I tested five unsupervised domain adaptation methods, including my own topology-consistent variant called TC-MMD, and none of them improves on training with the twin alone. This is a negative result, but an informative one. The gap analysis shows that the real operators perform the same action differently, which is a conditional shift, and correcting the class proportions does not help. Aligning feature distributions therefore cannot close the gap without some real supervision, which led to the next experiment.");

  // ===== 13. WP4 few-shot =====
  s = content("Study 4: Digital twin", "Study 4  ·  With a few real labels", "The twin cuts real labels needed by 5 to 20 times");
  s.addChart(pres.charts.LINE, [
    { name: "Real clips only, from scratch", labels: ["1", "5", "10", "20", "50"], values: [11.36, 22.46, 34.08, 42.77, 51.39] },
    { name: "Twin model, fine-tuned", labels: ["1", "5", "10", "20", "50"], values: [29.28, 45.64, 53.10, 59.11, 64.99] },
    { name: "Joint + prototype, then fine-tuned", labels: ["1", "5", "10", "20", "50"], values: [41.22, 49.43, 54.52, 60.20, 66.27] },
  ], Object.assign(chartBase(), {
    x: 0.5, y: 1.3, w: 5.9, h: 3.75, chartColors: [H.accent3, H.accent6, H.accent2], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7,
    valAxisMinVal: 0, valAxisMaxVal: 80, valAxisMajorUnit: 20, showLegend: true, legendPos: "b", legendFontSize: 10, legendColor: H.dk1,
    showTitle: true, title: "Macro-F1 (%) vs labelled real clips per class", showCatAxisTitle: false,
  }));
  stat(s, 6.7, 1.35, 2.8, "5–20×", "fewer real labels to match training on real clips alone", C.accent2, 32);
  stat(s, 6.7, 2.65, 2.8, "66.3 %", "macro-F1 with 50 clips per class, oracle 75.2 %", C.accent1, 32);
  T(s, "The prototype term gives a significant gain at 1 and 5 clips per class.", { x: 6.7, y: 3.95, w: 2.8, h: 0.8, fontSize: 12, color: C.text1 });
  s.addNotes("Target time: 1 minute 15. When a few real clips are labelled, the twin becomes very useful. The grey curve is a model trained on real clips only, and the orange curve is my proposed strategy: joint training on the twin and the few real clips with a class prototype term, followed by fine-tuning. A model initialised on the twin matches the scratch model while using five to twenty times fewer labelled real clips, and the prototype term gives a significant gain at one and five clips per class. With fifty clips per class, the strategy reaches 66.3 percent macro-F1, against 75.2 for the oracle trained on all real labels. The paper is being written for a journal, and this result frames the digital twin as a way to reduce annotation cost rather than as a free replacement for real data.");

  // ===== 16. Year 2 plan =====
  pres.addSection({ title: "Year 2" });
  s = content("Year 2", "Plan for year 2", "From recognising the operator to adapting the robot");
  const plan = [
    ["Semester 1", C.accent1, [
      [I.file, "Close open work", "Subject-wise rerun, revisions of both journal papers, ICAART talk"],
      [I.route, "A1 Operator style", "Encode each operator's action order, timing and trajectories on InHARD skeletons, then cluster strategies"],
    ]],
    ["Semester 2", C.accent2, [
      [I.brain, "A2 Learn from corrections", "Extend FLAIR to incremental inverse RL so operator corrections update the robot's reward online"],
      [I.sliders, "A3 Adaptive behaviour", "Style-conditioned generative model for robot timing and trajectories, first tested in simulation"],
    ]],
  ];
  plan.forEach(([sem, col, items], j) => {
    const x = 0.5 + j * 4.6;
    T(s, sem, { x, y: 1.3, w: 4.4, h: 0.35, fontSize: 15, bold: true, color: col });
    items.forEach(([ic, h, d], i) => {
      const y = 1.75 + i * 1.32;
      card(s, x, y, 4.4, 1.17, C.background2, h + " card");
      circleIcon(s, ic, x + 0.2, y + 0.27, 0.6, col, h);
      T(s, h, { x: x + 0.98, y: y + 0.12, w: 3.3, h: 0.33, fontSize: 14, bold: true, color: C.text2 });
      T(s, d, { x: x + 0.98, y: y + 0.47, w: 3.3, h: 0.65, fontSize: 12, color: C.text1 });
    });
  });
  card(s, 0.5, 4.5, 9, 0.55, "FDF1E3", "Targets card");
  T(s, [{ text: "Year 2 targets  ", options: { bold: true, color: C.accent2 } }, { text: "two journal papers, one conference paper on operator style, a simulation demonstrator" }], { x: 0.7, y: 4.5, w: 8.6, h: 0.55, fontSize: 13, valign: "middle" });
  s.addNotes("Target time: 1 minute 30. The second year moves from recognising the operator to adapting the robot. In the first semester I will close the open work: a subject-wise evaluation for the danger detection revision, the answers to the reviews of the digital twin paper, and the ICAART presentation. In parallel I will start the operator style model, encoding the order, the timing and the trajectories of each operator on the InHARD skeleton data that I already processed, and clustering the strategies. In the second semester I will extend FLAIR towards incremental inverse reinforcement learning, so that operator corrections update the robot's objective online, and I will build a style-conditioned generative model for robot timing and trajectories, tested first in simulation. The targets are two journal papers, one conference paper on operator style and a simulation demonstrator, which prepares the validation with real operators in the third year.");

  // ===== 17. Closing =====
  pres.addSection({ title: "Closing" });
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Closing" });
  circleIcon(s, I.handshake, 0.6, 0.55, 0.75, C.accent1, "Collaboration");
  s.addText("Thank you, I welcome your questions", { placeholder: "title" });
  s.addText([
    { text: "Four studies across observation, learning and safety", options: { breakLine: true } },
    { text: "KES 2026 published  ·  ICAART 2027 accepted  ·  two journal papers under review", options: { breakLine: true } },
    { text: "Year 2: operator style, learning from corrections, adaptive robot behaviour", options: { color: C.accent2 } },
  ], { placeholder: "body", paraSpaceAfter: 6 });
  s.addNotes("Target time: 30 seconds. To conclude, this first year produced four studies covering the observation of the operator, incremental learning and safety, with one paper published, one accepted and two under review in journals, together with a much more rigorous evaluation practice. The second year will build on these foundations to adapt the robot to each operator. Thank you for your attention, I am ready for your questions.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
