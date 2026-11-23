const pptxgen = require("pptxgenjs");
const React = require("react");
const RDS = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require("/home/g0amer/.claude/skills/synced/82ab380d-449b-4c1a-9efc-43946a06f306_bdc6918c-08ee-4858-8199-65c9ca844fb6/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "deck.pptx";
const IMG = __dirname + "/img_fr/";

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
pres.title = "Revue d'avancement de thèse, année 1";
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
    { text: { text: "A. Gargouri  ·  Revue de thèse, année 1  ·  ATISP & ESME Research Lab", options: { x: 0.5, y: 5.28, w: 6.5, h: 0.22, fontSize: 9, color: C.accent3, margin: 0 } } },
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

  // ===== 1. Titre =====
  pres.addSection({ title: "Ouverture" });
  let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Ouverture" });
  circleIcon(s, I.robot, 0.6, 0.55, 0.75, C.accent1, "Robot");
  T(s, "REVUE D'AVANCEMENT DE THÈSE  ·  PASSAGE DE LA 1re À LA 2e ANNÉE", { x: 1.55, y: 0.75, w: 7.8, h: 0.35, fontSize: 12, bold: true, color: C.accent2, charSpacing: 1.5, valign: "middle" });
  s.addText("Collaboration humain-robot : apprentissage incrémental et adaptation comportementale", { placeholder: "title" });
  s.addText([
    { text: "Ameur Gargouri", options: { bold: true, color: C.background1, fontSize: 18, breakLine: true } },
    { text: "Encadrants : Mohamed Ksantini, Univ. de Sfax  ·  Mohamed Karray, ESME Research Lab", options: { breakLine: true } },
    { text: "Laboratoire ATISP, ENET'Com, Université de Sfax  ·  ESME Research Lab, Ivry-sur-Seine" },
  ], { placeholder: "body" });
  s.addNotes("Durée visée : 30 secondes. Bonjour aux membres du jury. Je suis Ameur Gargouri, et je vais vous présenter les travaux menés pendant la première année de ma thèse sur la collaboration humain-robot, l'apprentissage incrémental et l'adaptation comportementale, sous la direction de Mohamed Ksantini au laboratoire ATISP et de Mohamed Karray à l'ESME Research Lab. Je rappellerai d'abord la question de recherche, puis je présenterai les quatre études de l'année avec leurs résultats et leurs limites, et je terminerai par le programme de la deuxième année.");

  // ===== 1b. Plan =====
  s = content("Ouverture", "Plan", "Plan de la présentation");
  const outline = [
    ["Cadre de la thèse", "Problématique, bilan de l'année 1, état de l'art et datasets", C.accent3],
    ["Étude 1  ·  Reconnaissance", "Reconnaître la tâche de l'opérateur par fusion physiologique", C.accent1],
    ["Étude 2  ·  Détection du danger", "Détecter le danger avec des capteurs portés par l'opérateur", C.accent1],
    ["Étude 3  ·  FLAIR", "Apprendre de nouveaux opérateurs sans oublier les anciens", C.accent1],
    ["Étude 4  ·  Jumeau numérique", "Transfert d'un jumeau numérique vers un poste réel", C.accent1],
    ["Programme de l'année 2", "De la reconnaissance de l'opérateur à l'adaptation du robot", C.accent2],
  ];
  outline.forEach(([h, d, col], i) => {
    const x = 0.5 + (i % 2) * 4.6, y = 1.35 + Math.floor(i / 2) * 1.25;
    card(s, x, y, 4.4, 1.08, C.background2, h + " card");
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.22, w: 0.64, h: 0.64, fill: { color: col }, line: { type: "none" }, objectName: "Step " + (i + 1) });
    T(s, String(i + 1), { x: x + 0.2, y: y + 0.22, w: 0.64, h: 0.64, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle", fontFace: THEME.headFontFace });
    T(s, h, { x: x + 1.05, y: y + 0.17, w: 3.2, h: 0.35, fontSize: 15, bold: true, color: C.text2, valign: "middle" });
    T(s, d, { x: x + 1.05, y: y + 0.54, w: 3.2, h: 0.42, fontSize: 12, color: C.text1 });
  });
  s.addNotes("Durée visée : 20 secondes. La présentation suit six parties. Je pose d'abord le cadre de la thèse avec la problématique, le bilan de l'année, l'état de l'art et les datasets. Je présente ensuite dans l'ordre les quatre études de l'année, la reconnaissance des tâches, la détection du danger, l'apprentissage incrémental avec FLAIR et le transfert depuis un jumeau numérique. Je termine par le programme de la deuxième année.");

  // ===== 2. Problématique =====
  const fig = (sl, f, x, y, w, h, alt, cap) => {
    sl.addImage({ path: IMG + f, x, y, w, h, altText: alt, objectName: alt });
    if (cap) T(sl, cap, { x, y: y + h + 0.04, w, h: 0.22, fontSize: 9, italic: true, color: C.accent3 });
  };
  pres.addSection({ title: "Cadre de la thèse" });
  s = content("Cadre de la thèse", "Cadre de la thèse", "Problématique");
  card(s, 0.5, 1.3, 9, 0.85, C.background2, "Question card");
  T(s, "Comment un robot collaboratif peut-il apprendre, de manière incrémentale et sans reprogrammation, les préférences, les stratégies et le style de travail de chaque opérateur, tout en restant sûr ?", { x: 0.75, y: 1.38, w: 8.5, h: 0.7, fontSize: 15, italic: true, color: C.text2, valign: "middle" });
  const axes = [
    [I.eye, "A1  Observer", "Modéliser l'état et les gestes de l'opérateur", C.accent1],
    [I.brain, "A2  Apprendre", "Intégrer chaque nouvel opérateur sans oubli", C.accent2],
    [I.sliders, "A3  Adapter", "Ajuster le robot au style de l'opérateur", C.accent3],
    [I.check, "A4  Valider", "Mesurer la fluidité de la collaboration", C.accent5],
  ];
  axes.forEach(([ic, h, d, col], i) => {
    const x = 0.5 + (i % 2) * 2.3, y = 2.4 + Math.floor(i / 2) * 1.3;
    card(s, x, y, 2.15, 1.18, C.background2, h + " card");
    circleIcon(s, ic, x + 0.15, y + 0.15, 0.48, col, h);
    T(s, h, { x: x + 0.75, y: y + 0.15, w: 1.35, h: 0.48, fontSize: 13, bold: true, color: C.text2, valign: "middle" });
    T(s, d, { x: x + 0.15, y: y + 0.7, w: 1.9, h: 0.45, fontSize: 11, color: C.text1 });
  });
  fig(s, "inhard_physical_capture.png", 5.64, 2.4, 3.86, 2.45, "Opérateur et cobot sur un poste d'assemblage", "Opérateur et cobot sur un poste d'assemblage, InHARD, Dallel et al.");
  s.addNotes("Durée visée : 1 minute. La thèse se demande comment un robot collaboratif peut apprendre, pas à pas et sans être reprogrammé, les préférences et le style de travail de chaque opérateur, tout en restant sûr. La photo montre ce contexte, un opérateur qui travaille aux côtés d'un cobot sur un poste d'assemblage. Le sujet s'organise en quatre axes. Le premier axe observe l'opérateur, c'est-à-dire son état, ses actions et ses habitudes. Le deuxième apprend de manière incrémentale, afin d'intégrer chaque nouvel opérateur sans oublier les précédents. Le troisième adapte le comportement du robot au style de l'opérateur, et le quatrième valide la collaboration avec des métriques objectives de fluidité. La sécurité est une contrainte transversale aux quatre axes. Cette première année s'est concentrée sur les fondations de perception et d'apprentissage, c'est-à-dire les axes un et deux ainsi que la sécurité.");

  // ===== 3. L'année 1 en bref =====
  s = content("Cadre de la thèse", "Vue d'ensemble", "Bilan de la première année");
  const sc = [
    ["1", "publié", "FLAIR, KES 2026, Procedia CS", C.accent5],
    ["1", "accepté", "Reconnaissance des tâches, ICAART 2027", C.accent1],
    ["2", "en évaluation", "Détection du danger, JIRS\nJumeau numérique vers réel, revue", C.accent2],
  ];
  sc.forEach(([n, l, d, col], i) => {
    const w = 2.88, x = 0.5 + i * (w + 0.225);
    card(s, x, 1.35, w, 1.55, C.background2, l + " card");
    T(s, n, { x: x + 0.2, y: 1.42, w: 0.6, h: 0.7, fontSize: 40, bold: true, color: col, fontFace: THEME.headFontFace, valign: "middle" });
    T(s, l, { x: x + 0.75, y: 1.55, w: w - 0.8, h: 0.45, fontSize: 15, bold: true, color: C.text2, valign: "middle" });
    T(s, d, { x: x + 0.2, y: 2.15, w: w - 0.35, h: 0.65, fontSize: 11, color: C.text1 });
  });
  // frise
  s.addShape(pres.shapes.LINE, { x: 0.7, y: 3.55, w: 8.6, h: 0, line: { color: C.accent6, width: 2 }, objectName: "Timeline axis" });
  const tl = [
    ["Nov. 2025", "Article Machines publié, issu de mon PFE d'ingénieur à l'ENIS"],
    ["Janv. – avr. 2026", "État de l'art : six rapports sur l'apprentissage incrémental et les datasets"],
    ["Début 2026", "Étude de reconnaissance des tâches sur données multimodales"],
    ["Mi-2026", "Détection du danger et FLAIR, présentation à KES 2026"],
    ["Sept. – nov. 2026", "Étude jumeau numérique sur InHARD réel soumise, acceptation ICAART"],
  ];
  tl.forEach(([d, t], i) => {
    const x = 0.5 + i * 1.84;
    s.addShape(pres.shapes.OVAL, { x: x + 0.12, y: 3.44, w: 0.22, h: 0.22, fill: { color: i === 4 ? C.accent2 : C.accent1 }, line: { color: C.background1, width: 2 }, objectName: "Milestone " + (i + 1) });
    T(s, d, { x, y: 3.78, w: 1.7, h: 0.3, fontSize: 12, bold: true, color: C.text2 });
    T(s, t, { x, y: 4.08, w: 1.7, h: 1.0, fontSize: 11, color: C.text1 });
  });
  s.addNotes("Durée visée : 1 minute 30. Cette diapositive résume l'année. Quatre études ont conduit à quatre articles. FLAIR est publié dans Procedia Computer Science après ma présentation à KES 2026, l'article sur la reconnaissance des tâches vient d'être accepté à ICAART 2027, l'article sur la détection du danger est en cours d'évaluation au Journal of Intelligent and Robotic Systems, et l'article sur le jumeau numérique vient d'être soumis et est lui aussi en évaluation dans une revue. L'année a commencé par la publication dans Machines des travaux de mon projet de fin d'études du cycle ingénieur en génie informatique à l'École Nationale d'Ingénieurs de Sfax, puis par un état de l'art structuré de janvier à avril, documenté dans six rapports sur l'apprentissage incrémental, l'apprentissage robotique continu et les datasets de collaboration humain-robot. L'étude de reconnaissance des tâches est venue en premier et m'a beaucoup appris sur la rigueur de l'évaluation. La détection du danger et FLAIR ont suivi à la mi-2026, et depuis septembre je travaille sur le transfert d'un jumeau numérique vers un poste réel.");

  // ===== 4. État de l'art =====
  s = content("Cadre de la thèse", "État de l'art", "Verrous scientifiques");
  const gaps = [
    [I.redo, "Oubli entre opérateurs", "Les méthodes d'apprentissage continu reposent sur la régularisation ou le rejeu seuls, et sont rarement testées sur des suites d'opérateurs humains."],
    [I.shield, "Sécurité sans capteur d'effort", "La détection de collision repose sur des capteurs d'effort, des caméras ou des modèles simples, sans étude systématique du deep learning sur capteurs inertiels portés."],
    [I.heart, "État de l'opérateur", "Les études physiologiques rapportent un fort effet opérateur, mais la fusion est supposée utile plutôt que testée sous un protocole unique."],
    [I.cube, "Coût des labels réels", "Les jumeaux numériques produisent des données annotées à faible coût, mais leur transfert vers des opérateurs réels reste incertain."],
  ];
  gaps.forEach(([ic, h, d], i) => {
    const x = 0.5 + (i % 2) * 4.6, y = 1.35 + Math.floor(i / 2) * 1.85;
    card(s, x, y, 4.4, 1.65, C.background2, h + " card");
    circleIcon(s, ic, x + 0.22, y + 0.25, 0.6, C.accent1, h);
    T(s, h, { x: x + 1.0, y: y + 0.22, w: 3.2, h: 0.35, fontSize: 15, bold: true, color: C.text2 });
    T(s, d, { x: x + 1.0, y: y + 0.6, w: 3.2, h: 0.95, fontSize: 12, color: C.text1 });
  });
  s.addNotes("Durée visée : 1 minute 15. L'état de l'art mené entre janvier et avril 2026 a fait ressortir quatre lacunes, et chaque lacune a donné lieu à une étude. Les méthodes d'apprentissage continu reposent en général soit sur la régularisation, soit sur le rejeu, et elles sont rarement évaluées sur une suite d'opérateurs humains, ce qui correspond exactement à la situation d'un cobot qui rencontre de nouvelles personnes. Pour la sécurité, la détection de collision repose surtout sur des capteurs d'effort et de couple, sur des caméras ou sur un apprentissage peu profond, et les modèles profonds appliqués aux données inertielles portées n'avaient pas été comparés de manière systématique. Pour l'état de l'opérateur, les études physiologiques rapportent toutes un fort effet de l'individu, et pourtant la fusion multimodale est en général supposée utile plutôt que testée. Enfin, les jumeaux numériques promettent des données annotées à faible coût, mais il n'est pas établi qu'un modèle de reconnaissance entraîné sur un poste virtuel se transfère à des opérateurs réels.");

  // ===== 5. Base expérimentale =====
  s = content("Cadre de la thèse", "Base expérimentale", "Datasets et protocole");
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 12 } });
  const rows = [
    [hdr("Dataset"), hdr("Étude"), hdr("Participants"), hdr("Signaux"), hdr("Traités")],
    ["MultiPhysio-HRC", "Reconnaissance des tâches", "52 opérateurs", "EEG, ECG, EDA, EMG, respiration", "5\u00A0640 fenêtres, 100\u00A0%"],
    ["DASIG", "Détection du danger", "60 sujets", "Centrales inertielles portées, 65 canaux", "180 enregistrements"],
    ["HARMONIC", "Apprentissage incrémental", "10 opérateurs en séquence", "Joystick, articulations du robot, effecteur", "447 essais"],
    ["InHARD et InHARD-DT", "Transfert jumeau vers réel", "16 réels, 12 virtuels", "Squelette corps entier, 13 actions", "~9 500 clips"],
  ].map((r, i) => i === 0 ? r : r.map((c) => ({ text: c, options: { fontSize: 11, color: C.text1, fill: { color: i % 2 ? C.background1 : C.background2 } } })));
  s.addTable(rows, { x: 0.5, y: 1.35, w: 9, colW: [1.85, 1.65, 1.65, 2.25, 1.6], rowH: 0.48, border: { type: "solid", pt: 0.5, color: "DDE4E8" }, valign: "middle", margin: 0.06 });
  circleIcon(s, I.db, 0.5, 4.2, 0.55, C.accent2, "Data");
  T(s, "Chaque pipeline est scripté, versionné et audité avant la modélisation, ce qui permet de régénérer chaque résultat de cette présentation à partir des données brutes.", { x: 1.25, y: 4.2, w: 8.2, h: 0.55, fontSize: 13, color: C.text2, valign: "middle" });
  s.addNotes("Durée visée : 1 minute. Toutes les études reposent sur des datasets publics, ce qui les rend reproductibles et comparables aux travaux d'autres équipes. MultiPhysio-HRC fournit cinq signaux physiologiques pour cinquante-deux opérateurs après nettoyage. DASIG fournit des données inertielles portées par soixante sujets, avec des mouvements normaux et dangereux. HARMONIC fournit des enregistrements d'autonomie partagée avec un bras robotique, dont dix opérateurs forment un benchmark séquentiel. InHARD et son jumeau numérique fournissent des enregistrements du squelette corps entier sur le même poste d'assemblage, dans la cellule réelle et en réalité virtuelle. Pour chaque dataset, j'ai construit un pipeline de prétraitement scripté, si bien que chaque résultat peut être régénéré à partir des données brutes.");

  // ===== 6. Étude 1 méthodologie =====
  pres.addSection({ title: "Étude 1 : Reconnaissance des tâches" });
  s = content("Étude 1 : Reconnaissance des tâches", "Étude 1  ·  Reconnaissance des tâches", "Méthodologie");
  fig(s, "s1_pipeline.png", 0.5, 1.3, 4.9, 3.28, "Pipeline de traitement MultiPhysio-HRC", "Pipeline de traitement de MultiPhysio-HRC, article ICAART 2027");
  const steps = [
    [I.tags, "Étiqueter", "21 conditions du protocole ramenées à 5 classes par une règle déterministe"],
    [I.search, "Auditer", "257 colonnes réduites à 128 variables, sans identifiants ni doublons"],
    [I.layers, "Fusionner", "Cinq modalités réunies en une seule représentation"],
    [I.balance, "Comparer", "Cinq détecteurs et une ablation des modalités"],
  ];
  steps.forEach(([ic, h, d], i) => {
    const y = 1.3 + i * 0.72;
    circleIcon(s, ic, 5.7, y + 0.05, 0.45, C.accent1, h);
    T(s, h, { x: 6.3, y, w: 3.2, h: 0.28, fontSize: 13, bold: true, color: C.text2 });
    T(s, d, { x: 6.3, y: y + 0.28, w: 3.2, h: 0.42, fontSize: 11, color: C.text1 });
  });
  card(s, 5.7, 4.25, 3.8, 0.75, C.background2, "Classes card");
  T(s, [{ text: "Cinq classes  ", options: { bold: true, color: C.text2 } }, { text: "charge cognitive, stress élevé, tâche industrielle, charge faible, autre" }], { x: 5.85, y: 4.3, w: 3.55, h: 0.65, fontSize: 11, valign: "middle" });
  s.addNotes("Durée visée : 1 minute 15. La première étude relève de l'axe observer : un robot peut-il reconnaître dans quelle condition se trouve l'opérateur à partir de signaux physiologiques ? J'ai utilisé MultiPhysio-HRC, où les opérateurs réalisent des tests cognitifs, des scénarios de stress en réalité virtuelle et un véritable désassemblage de batterie avec un cobot. Le protocole comporte quatre étapes. J'ai d'abord défini une règle déterministe qui ramène les vingt et une conditions du protocole à cinq classes, de sorte que les étiquettes sont entièrement reproductibles. J'ai ensuite audité l'export des variables, ce qui a retiré les identifiants d'enregistrement et les colonnes EEG dupliquées et laissé cent vingt-huit variables physiologiques. Les cinq modalités sont fusionnées en une seule représentation, et enfin cinq détecteurs sont comparés, avec une ablation qui vérifie si la fusion aide réellement.");

  // ===== 7. Étude 1 résultats =====
  s = content("Étude 1 : Reconnaissance des tâches", "Étude 1  ·  Reconnaissance des tâches", "Résultats et limites");
  s.addChart(pres.charts.BAR, [{ name: "Gain de macro-F1 par la fusion", labels: ["Gain minimal", "Gain maximal"], values: [0.047, 0.186] }], Object.assign(chartBase(), {
    x: 5.1, y: 1.3, w: 4.4, h: 2.35, barDir: "bar", chartColors: [H.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.000",
    valAxisHidden: true, valGridLine: { style: "none" }, showTitle: true, title: "Entrée fusionnée vs meilleure modalité seule, macro-F1", showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 0.22,
  }));
  stat(s, 0.5, 1.3, 2.2, "0,95", "exactitude, XGBoost sur variables fusionnées");
  stat(s, 2.75, 1.3, 2.2, "0,91", "macro-F1 sur cinq classes");
  card(s, 0.5, 2.75, 4.4, 0.9, C.background2, "Status card");
  circleIcon(s, I.flag, 0.68, 2.92, 0.55, C.accent5, "Accepted");
  T(s, "Accepté à ICAART 2027, article 340, version finale prête", { x: 1.38, y: 2.85, w: 3.4, h: 0.7, fontSize: 13, bold: true, color: C.text2, valign: "middle" });
  card(s, 0.5, 3.9, 9, 1.1, "FDF1E3", "Limit card");
  T(s, [
    { text: "Limite assumée  ", options: { bold: true, color: C.accent2 } },
    { text: "le découpage 80/20 partage des opérateurs entre apprentissage et test. En leave-one-subject-out, XGBoost tombe à 0,50 de macro-F1, contre 0,63 d'exactitude pour la classe majoritaire. Ces scores sont un plafond intra-sujet, et la généralisation à de nouveaux opérateurs est le problème ouvert qui motive l'axe d'apprentissage incrémental." },
  ], { x: 0.75, y: 3.98, w: 8.5, h: 0.95, fontSize: 12, valign: "middle" });
  s.addNotes("Durée visée : 1 minute 30. Sur les variables fusionnées, XGBoost atteint une exactitude de 0,95 et un macro-F1 de 0,91 sur les cinq classes. Le macro-F1 est la moyenne des scores F1 par classe, il n'est donc pas dominé par les classes fréquentes. L'ablation montre que la fusion des modalités améliore le macro-F1 par rapport à la meilleure modalité seule pour chaque détecteur testé, de 0,047 à 0,186. L'article a été accepté à ICAART 2027. Je tiens à être transparent sur la limite : le découpage hold-out partage des opérateurs entre l'apprentissage et le test. Lorsque chaque opérateur est mis de côté à tour de rôle, le macro-F1 de XGBoost descend à 0,50, l'effet opérateur domine donc. Ce résultat figure dans l'article, et c'est précisément pour cela que la thèse a besoin de modèles qui s'adaptent à chaque nouvel opérateur.");

  // ===== 8. Étude 2 méthodologie =====
  pres.addSection({ title: "Étude 2 : Détection du danger" });
  s = content("Étude 2 : Détection du danger", "Étude 2  ·  Détection du danger", "Méthodologie");
  fig(s, "s2_hrc.png", 0.5, 1.3, 4.2, 2.8, "Placement des centrales inertielles", "Placement des cinq centrales inertielles sur l'opérateur, DASIG");
  fig(s, "s2_pipeline.png", 4.95, 1.3, 4.55, 2.88, "Pipeline de détection du danger", "Pipeline en quatre phases, des capteurs à la décision SAFE ou DANGER");
  card(s, 0.5, 4.5, 9, 0.52, C.background2, "Method card");
  T(s, [
    { text: "Entrée  ", options: { bold: true, color: C.text2 } }, { text: "65 → 195 canaux      " },
    { text: "Modèles  ", options: { bold: true, color: C.text2 } }, { text: "cinq réseaux profonds 1D      " },
    { text: "Post-traitement  ", options: { bold: true, color: C.text2 } }, { text: "filtre médian et seuil" },
  ], { x: 0.7, y: 4.5, w: 8.6, h: 0.52, fontSize: 11, valign: "middle" });
  s.addNotes("Durée visée : 45 secondes. La deuxième étude porte sur la sécurité. L'objectif est de détecter un état dangereux à partir de capteurs inertiels portés par l'opérateur, au lieu de capteurs d'effort ou de couple coûteux. Le dataset DASIG utilise cinq centrales inertielles placées sur le sternum, les bras et les avant-bras, comme le montre la figure de gauche. Le pipeline de droite comporte quatre phases. Les soixante-cinq canaux bruts sont découpés en fenêtres puis étendus sur GPU avec la vitesse et l'accélération, ce qui donne cent quatre-vingt-quinze canaux. Cinq architectures profondes unidimensionnelles sont ensuite entraînées dans des conditions identiques, et un post-traitement temporel produit la décision finale.");

  // ===== 8b. Étude 2 résultats =====
  s = content("Étude 2 : Détection du danger", "Étude 2  ·  Détection du danger", "Résultats");
  stat(s, 0.5, 1.35, 2.0, "97,1 %", "dangers détectés, ConvNeXt 1D", C.accent4, 30);
  stat(s, 2.6, 1.35, 2.1, "−87,5 %", "fausses alarmes, pour 1,2 pt de rappel", C.accent5, 30);
  T(s, "ConvNeXt 1D et le réseau convolutif temporel dépassent 98,6 % de macro-F1. Le post-traitement supprime la plupart des fausses alarmes, qui arrêtent la production.", { x: 0.5, y: 2.75, w: 4.2, h: 1.3, fontSize: 13, color: C.text1 });
  card(s, 0.5, 4.3, 4.2, 0.72, "FDF1E3", "Limit card");
  T(s, [{ text: "Limite  ", options: { bold: true, color: C.accent2 } }, { text: "validation stratifiée en 5 plis sur 69k fenêtres, réévaluation par sujet prévue" }], { x: 0.65, y: 4.36, w: 3.95, h: 0.6, fontSize: 12, valign: "middle" });
  s.addChart(pres.charts.BAR, [{ name: "Macro-F1 (%)", labels: ["ConvNeXt 1D", "TCN", "MLP-Mixer 1D", "InceptionTime", "Transformer 1D"], values: [98.68, 98.62, 98.32, 96.22, 92.50] }], Object.assign(chartBase(), {
    x: 5.0, y: 1.3, w: 4.5, h: 3.75, barDir: "bar", chartColors: [H.accent1, H.accent6, H.accent6, H.accent6, H.accent6], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
    valAxisMinVal: 90, valAxisMaxVal: 100, valAxisHidden: true, valGridLine: { style: "none" }, catAxisOrientation: "maxMin", showTitle: true, title: "Macro-F1 (%), DASIG, 60 sujets", showLegend: false,
  }));
  s.addNotes("Durée visée : 1 minute. ConvNeXt 1D et le réseau convolutif temporel sont les meilleurs, avec des macro-F1 de 98,68 et 98,62 pour cent, et ConvNeXt détecte 97,1 pour cent des dangers. Un post-traitement temporel simple, un filtre médian avec un seuil ajusté, supprime 87,5 pour cent des fausses alarmes pour seulement 1,2 point de rappel, ce qui compte car chaque fausse alarme arrête la production. L'évaluation repose sur une validation stratifiée en cinq plis sur les fenêtres, une évaluation par sujet est donc prévue pour la révision. L'article est en cours d'évaluation au Journal of Intelligent and Robotic Systems.");

  // ===== 9. Étude 3 méthodologie =====
  pres.addSection({ title: "Étude 3 : FLAIR" });
  s = content("Étude 3 : FLAIR", "Étude 3  ·  Apprentissage incrémental", "Architecture FLAIR");
  T(s, "Quatre mécanismes : FiLM par opérateur, rejeu adapté aux tâches, régularisation Fisher + RetroBoost et fusion des têtes. Dix opérateurs HARMONIC appris l'un après l'autre, évaluation sur tous les opérateurs déjà vus.", { x: 0.5, y: 1.25, w: 9, h: 0.5, fontSize: 12, color: C.text1 });
  fig(s, "s3_arch.png", 0.5, 1.85, 5.5, 2.89, "Architecture FLAIR", "Architecture FLAIR, article KES 2026");
  fig(s, "s3_photo.jpg", 6.45, 1.85, 3.05, 2.89, "Opérateur et bras Kinova en autonomie partagée", "Opérateur et bras Kinova, dataset HARMONIC");
  s.addNotes("Durée visée : 1 minute 15. La troisième étude est le cœur de l'axe apprendre. Lorsqu'un réseau de neurones apprend un nouvel opérateur, il tend à perdre ce qu'il savait des précédents, c'est ce qu'on appelle l'oubli catastrophique. J'ai évalué ce phénomène sur HARMONIC, où des opérateurs pilotent un bras robotique Kinova dans une tâche d'aide au repas, comme sur la photo : le modèle apprend dix opérateurs l'un après l'autre et il est testé sur tous après chaque étape. FLAIR combine quatre mécanismes. Chaque opérateur reçoit sa propre modulation des variables, initialisée à partir de l'opérateur précédent le plus proche. Une mémoire de rejeu répète des échantillons passés avec distillation et donne plus de poids aux opérateurs en train d'être oubliés. Une régularisation de Fisher en ligne protège les paramètres importants et se renforce à mesure que les opérateurs s'accumulent. Enfin, des têtes de sortie partagée et spécifiques à chaque opérateur sont combinées.");

  // ===== 10. Étude 3 résultats =====
  s = content("Étude 3 : FLAIR", "Étude 3  ·  Apprentissage incrémental", "Résultats");
  s.addChart(pres.charts.BAR, [{ name: "R² moyen final", labels: ["FLAIR", "Entraînement joint", "DER++", "A-GEM", "Online EWC", "EWC"], values: [0.690, 0.644, 0.612, 0.420, 0.415, 0.368] }], Object.assign(chartBase(), {
    x: 0.5, y: 1.3, w: 5.0, h: 3.7, barDir: "bar", chartColors: [H.accent2, H.accent3, H.accent6, H.accent6, H.accent6, H.accent6], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisHidden: true, valGridLine: { style: "none" }, catAxisOrientation: "maxMin", showTitle: true, title: "R² moyen final après 10 opérateurs", showLegend: false,
  }));
  stat(s, 5.9, 1.3, 1.7, "0,017", "oubli moyen, 8,8× inférieur à DER++", C.accent2, 30);
  stat(s, 7.75, 1.3, 1.75, "94 %", "conservés sur l'opérateur 1 après 9 autres", C.accent1, 30);
  stat(s, 5.9, 2.75, 1.7, "1,3 Mo", "surcoût mémoire", C.accent3, 30);
  card(s, 5.9, 4.1, 3.6, 0.9, C.background2, "Status card");
  circleIcon(s, I.flag, 6.05, 4.27, 0.55, C.accent5, "Published");
  T(s, "Présenté à KES 2026, publié dans Procedia Computer Science", { x: 6.72, y: 4.15, w: 2.7, h: 0.8, fontSize: 12, bold: true, color: C.text2, valign: "middle" });
  s.addNotes("Durée visée : 1 minute 15. Après les dix opérateurs, FLAIR atteint un R carré moyen final de 0,690, le meilleur parmi les méthodes incrémentales, devant DER++ à 0,612, et il égale ou dépasse la référence en entraînement joint à 0,644, qui réentraîne sur toutes les données à chaque étape. Son oubli moyen est de 0,017, presque neuf fois plus faible que DER++, et il conserve 94 pour cent de sa performance sur le premier opérateur après en avoir appris neuf autres, pour un surcoût mémoire d'environ 1,3 mégaoctet. J'ai présenté ces travaux à KES 2026 et ils sont publiés dans Procedia Computer Science. Leur limite est que les cibles sont des commandes du robot plutôt que des préférences, c'est pourquoi la deuxième année les étend vers l'apprentissage à partir des corrections de l'opérateur.");

  // ===== 11. Étude 4 méthodologie =====
  pres.addSection({ title: "Étude 4 : Jumeau numérique" });
  s = content("Étude 4 : Jumeau numérique", "Étude 4  ·  Jumeau numérique", "Méthodologie");
  fig(s, "s4_ga.png", 0.5, 1.3, 4.9, 3.5, "Résumé graphique du transfert du jumeau numérique vers le poste réel", "Du jumeau numérique au poste réel, résumé graphique de l'article");
  T(s, [
    { text: "Source  ", options: { bold: true, color: C.text2 } },
    { text: "InHARD-DT, le poste reproduit en réalité virtuelle, 12 sujets", options: { breakLine: true } },
    { text: "Cible  ", options: { bold: true, color: C.text2 } },
    { text: "InHARD, le poste physique, 16 sujets, 13 actions d'assemblage", options: { breakLine: true } },
    { text: "Modèle  ", options: { bold: true, color: C.text2 } },
    { text: "réseau squelettique CTR-GCN, 4 plis disjoints par sujet × 3 graines", options: { breakLine: true } },
    { text: "Écart  ", options: { bold: true, color: C.text2 } },
    { text: "même combinaison de capture, l'écart est donc comportemental : posture, rythme, objets et proportions des classes" },
  ], { x: 5.7, y: 1.3, w: 3.8, h: 2.45, fontSize: 12, paraSpaceAfter: 6 });
  fig(s, "s4_vr.jpg", 5.7, 3.85, 3.8, 1.15, "Capture en réalité virtuelle, InHARD-DT", "Capture en réalité virtuelle, InHARD-DT");
  s.addNotes("Durée visée : 1 minute. La quatrième étude a commencé en septembre. Annoter des enregistrements réels pour chaque opérateur et chaque poste coûte cher, alors qu'un jumeau numérique peut produire des données annotées à faible coût. Le résumé graphique de gauche présente la démarche. La question est de savoir si un modèle de reconnaissance d'actions sur squelette, entraîné sur la réplique virtuelle du poste InHARD, fonctionne sur le poste réel. Les deux datasets utilisent la même combinaison de capture de mouvement, la différence entre eux est donc comportementale : la posture, le rythme, l'absence de pièces physiques en réalité virtuelle et des proportions de classes différentes. J'utilise le réseau squelettique CTR-GCN avec quatre plis disjoints par sujet et trois graines, de sorte que chaque opérateur de test est inconnu à l'entraînement et qu'aucun label cible ne sert à la sélection du modèle.");

  // ===== 12. Étude 4 sans labels =====
  s = content("Étude 4 : Jumeau numérique", "Étude 4  ·  Jumeau numérique", "Transfert sans labels réels");
  s.addChart(pres.charts.BAR, [{ name: "Macro-F1 (%)", labels: ["Source seule", "FixMatch", "MMD", "TC-MMD", "CORAL", "DANN", "Oracle réel"], values: [40.59, 40.54, 39.66, 39.34, 38.88, 38.17, 75.22] }], Object.assign(chartBase(), {
    x: 0.5, y: 1.3, w: 5.6, h: 3.7, barDir: "col", chartColors: [H.accent1, H.accent6, H.accent6, H.accent6, H.accent6, H.accent6, H.accent5], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0",
    valAxisMinVal: 0, valAxisMaxVal: 85, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 10, showTitle: true, title: "Macro-F1 (%) sur des opérateurs réels inconnus", showLegend: false,
  }));
  stat(s, 6.4, 1.35, 3.1, "~35 pts", "d'écart entre l'entraînement sur le jumeau seul et l'oracle réel", C.accent4, 30);
  T(s, "Cinq méthodes d'adaptation non supervisée, dont ma variante TC-MMD qui préserve la topologie, restent à moins de 2,5 points de la source seule. L'analyse de l'écart montre un décalage conditionnel, et non un changement des proportions de classes, donc aligner les distributions de variables ne peut pas le combler.", { x: 6.4, y: 2.6, w: 3.1, h: 2.4, fontSize: 12, color: C.text1 });
  s.addNotes("Durée visée : 1 minute 15. Sans aucun label réel, le modèle entraîné sur le jumeau atteint 40,6 pour cent de macro-F1 sur des opérateurs réels inconnus, contre 75,2 pour cent pour un modèle entraîné sur les labels réels, soit un écart d'environ trente-cinq points. J'ai testé cinq méthodes d'adaptation de domaine non supervisée, dont ma propre variante qui préserve la topologie, appelée TC-MMD, et aucune n'améliore l'entraînement sur le jumeau seul. C'est un résultat négatif, mais informatif. L'analyse de l'écart montre que les opérateurs réels réalisent la même action différemment, ce qui est un décalage conditionnel, et corriger les proportions de classes n'aide pas. Aligner les distributions de variables ne peut donc pas combler l'écart sans une part de supervision réelle, ce qui a conduit à l'expérience suivante.");

  // ===== 13. Étude 4 quelques labels =====
  s = content("Étude 4 : Jumeau numérique", "Étude 4  ·  Jumeau numérique", "Transfert avec peu de labels réels");
  s.addChart(pres.charts.LINE, [
    { name: "Clips réels seuls, depuis zéro", labels: ["1", "5", "10", "20", "50"], values: [11.36, 22.46, 34.08, 42.77, 51.39] },
    { name: "Modèle jumeau, affiné", labels: ["1", "5", "10", "20", "50"], values: [29.28, 45.64, 53.10, 59.11, 64.99] },
    { name: "Joint + prototype, puis affiné", labels: ["1", "5", "10", "20", "50"], values: [41.22, 49.43, 54.52, 60.20, 66.27] },
  ], Object.assign(chartBase(), {
    x: 0.5, y: 1.3, w: 5.9, h: 3.75, chartColors: [H.accent3, H.accent6, H.accent2], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7,
    valAxisMinVal: 0, valAxisMaxVal: 80, valAxisMajorUnit: 20, showLegend: true, legendPos: "b", legendFontSize: 10, legendColor: H.dk1,
    showTitle: true, title: "Macro-F1 (%) selon le nombre de clips réels annotés par classe", showCatAxisTitle: false,
  }));
  stat(s, 6.7, 1.35, 2.8, "5–20×", "moins de labels réels pour égaler l'entraînement sur clips réels seuls", C.accent2, 32);
  stat(s, 6.7, 2.65, 2.8, "66,3 %", "macro-F1 avec 50 clips par classe, oracle 75,2 %", C.accent1, 32);
  T(s, "Le terme prototype apporte un gain significatif à 1 et 5 clips par classe.", { x: 6.7, y: 3.95, w: 2.8, h: 0.8, fontSize: 12, color: C.text1 });
  s.addNotes("Durée visée : 1 minute 15. Lorsque quelques clips réels sont annotés, le jumeau devient très utile. La courbe grise correspond à un modèle entraîné sur les clips réels seuls, et la courbe orange à la stratégie que je propose : un entraînement joint sur le jumeau et les quelques clips réels avec un terme de prototype de classe, suivi d'un affinage. Un modèle initialisé sur le jumeau égale le modèle entraîné depuis zéro tout en utilisant cinq à vingt fois moins de clips réels annotés, et le terme prototype apporte un gain significatif à un et cinq clips par classe. Avec cinquante clips par classe, la stratégie atteint 66,3 pour cent de macro-F1, contre 75,2 pour l'oracle entraîné sur tous les labels réels. L'article est en cours d'évaluation dans une revue, et ce résultat présente le jumeau numérique comme un moyen de réduire le coût d'annotation plutôt que comme un substitut gratuit aux données réelles.");

  // ===== 14. Programme de l'année 2 =====
  pres.addSection({ title: "Année 2" });
  s = content("Année 2", "Perspectives", "Programme de l'année 2");
  const plan = [
    ["Semestre 1", C.accent1, [
      [I.file, "Clore les travaux en cours", "Réévaluation par sujet, révisions des deux articles de revue, présentation à ICAART"],
      [I.route, "A1  Style de l'opérateur", "Encoder l'ordre des actions, le rythme et les trajectoires de chaque opérateur sur InHARD, puis regrouper les stratégies"],
    ]],
    ["Semestre 2", C.accent2, [
      [I.brain, "A2  Apprendre des corrections", "Étendre FLAIR à l'IRL incrémental pour que les corrections de l'opérateur mettent à jour la récompense du robot"],
      [I.sliders, "A3  Comportement adaptatif", "Modèle génératif conditionné par le style pour le rythme et les trajectoires du robot, testé en simulation"],
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
  T(s, [{ text: "Objectifs de l'année 2  ", options: { bold: true, color: C.accent2 } }, { text: "deux articles de revue, un article de conférence sur le style de l'opérateur, un démonstrateur en simulation" }], { x: 0.7, y: 4.5, w: 8.6, h: 0.55, fontSize: 13, valign: "middle" });
  s.addNotes("Durée visée : 1 minute 30. La deuxième année passe de la reconnaissance de l'opérateur à l'adaptation du robot. Au premier semestre, je clôturerai les travaux en cours : une évaluation par sujet pour la révision de l'article sur la détection du danger, les réponses aux relecteurs de l'article sur le jumeau numérique, et la présentation à ICAART. En parallèle, je commencerai le modèle de style de l'opérateur, en encodant l'ordre, le rythme et les trajectoires de chaque opérateur sur les données squelettiques InHARD déjà traitées, puis en regroupant les stratégies. Au second semestre, j'étendrai FLAIR vers l'apprentissage par renforcement inverse incrémental, afin que les corrections de l'opérateur mettent à jour en ligne l'objectif du robot, et je construirai un modèle génératif conditionné par le style pour le rythme et les trajectoires du robot, testé d'abord en simulation. Les objectifs sont deux articles de revue, un article de conférence sur le style de l'opérateur et un démonstrateur en simulation, ce qui prépare la validation avec de vrais opérateurs en troisième année.");

  // ===== 15. Conclusion =====
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusion" });
  circleIcon(s, I.handshake, 0.6, 0.55, 0.75, C.accent1, "Collaboration");
  s.addText("Merci de votre attention", { placeholder: "title" });
  s.addText([
    { text: "Quatre études couvrant l'observation, l'apprentissage et la sécurité", options: { breakLine: true } },
    { text: "KES 2026 publié  ·  ICAART 2027 accepté  ·  deux articles de revue en évaluation", options: { breakLine: true } },
    { text: "Année 2 : style de l'opérateur, apprentissage par corrections, comportement adaptatif du robot", options: { color: C.accent2 } },
  ], { placeholder: "body", paraSpaceAfter: 6 });
  s.addNotes("Durée visée : 30 secondes. Pour conclure, cette première année a produit quatre études couvrant l'observation de l'opérateur, l'apprentissage incrémental et la sécurité, avec un article publié, un article accepté et deux articles en cours d'évaluation dans des revues, ainsi qu'une pratique d'évaluation beaucoup plus rigoureuse. La deuxième année s'appuiera sur ces fondations pour adapter le robot à chaque opérateur. Merci pour votre attention, je suis prêt à répondre à vos questions.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  // tag all text as French so spell-check matches
  const JSZip = require("jszip"), fs = require("fs");
  const src = await JSZip.loadAsync(fs.readFileSync(OUT)), zip = new JSZip();
  for (const f of Object.keys(src.files).filter((f) => !src.files[f].dir)) {  // copy files only, no folder entries
    zip.file(f, /^ppt\/.*\.xml$/.test(f) ? (await src.file(f).async("string")).replace(/lang="en-US"/g, 'lang="fr-FR"') : await src.file(f).async("nodebuffer"), { createFolders: false });
  }
  fs.writeFileSync(OUT, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log("wrote", OUT);
})();
