<!-- converted from these.docx -->

Sujet de thèse doctorat de Monsieur Ameur Gargouri

Titre : Collaboration Humain-Robot : apprentissage incrémental et adaptation comportementale

1. Contexte et problématique :
Afin d'aligner dynamiquement son comportement sur celui de son partenaire humain, comment un cobot peut apprendre de manière incrémentale et continuer les préférences, les stratégies et le style opératoire spécifiques de chaque opérateur ?
L'objectif est de créer un système où la collaboration s'affine avec le temps, évoluant d'une interaction standardisée vers un partenariat personnalisé et hautement efficace, sans nécessiter de reprogrammation explicite.
2. Approches de résolution :
- Modélisation des schémas opératoires de l'humain
Il s’agit de modéliser les habitudes, les méthodes de travail et les séquences d'actions préférées d'un opérateur sur une tâche répétitive. Par exemple, un opérateur peut toujours commencer par serrer les vis du centre, tandis qu'un autre commence par les coins.
- Méthodologie :
- Apprentissage non supervisé des séquences : Utiliser des modèles de séquence (comme les LSTM « Long Short-Term Memory » à sortie temporelle ou les Transformers) pour apprendre et encoder le style d'un opérateur à partir de l'observation de ses trajectoires de main, de son ordre d'assemblage et de ses temps de pause.
- Clustering des stratégies : Identifier et classifier les différentes stratégies observées pour une même tâche globale.
- Apprentissage par renforcement inverse incrémental et interaction
Le robot doit générer la fonction de récompense (le but et les préférences) de l'humain non pas à partir d'un jeu de données fixe, mais en continu, pendant l'interaction. Si l'opérateur corrige le robot, ce dernier doit mettre à jour son modèle immédiatement.
- Méthodologie :
- Algorithme d'Apprentissage par Renforcement Inverse ou « Artificial Referencement Intelligence »  (ARI) en ligne : Adapter des algorithmes comme Maximum Entropy IRL ou Deep Maximum Entropy IRL pour fonctionner de manière incrémentale.
- Intégration des feedbacks physiques et verbaux : Utiliser les corrections physiques (exemple : guider le bras du robot), les signaux verbaux (non, pas comme ça) ou même les expressions faciales (froncement de sourcils) comme signaux de récompense négative pour mettre à jour rapidement le modèle de préférences.
- Mécanisme d’oubli adaptatif : Le robot doit pouvoir oublier les anciennes habitudes si l'opérateur change de méthode, sans pour autant être instable.
- Génération de comportements robotiques alignés et personnalisés
Après une phase d’entrainement du style de l'opérateur, une approche sera développée qui doit permettre au robot de générer des trajectoires, des vitesses et un temps qui s'alignent et se calent sur ce style pour être perçu comme plus naturel et plus fluide.
- Méthodologie :
- Générateur de trajectoires conditionné par le style : Utilisation de modèle génératif (comme Conditional Variational Autoencoder - CVAE) qui, étant donnée une tâche et un vecteur de style en entrée, génère une trajectoire qui imite le style de l'opérateur.
- Ordonnancement temporel collaboratif : Ajustement du temps des actions du robot pour qu'il anticipe les moments où l'opérateur aura besoin d'une pièce, en se calant sur le rythme naturel de ce dernier, évitant ainsi les attentes ou les interruptions.
- Optimisation conjointe sécurité/efficacité/confort : La trajectoire générée doit respecter les contraintes de sécurité et d'efficacité énergétique.
- Mesure de l'alignement et de la fluidité de la collaboration
Il s’agit de quantifier objectivement le degré d'alignement entre le comportement du robot et celui de l'humain, et comment le corréler à la perception subjective de fluidité.
- Méthodologie :
- Entropie des actions conjointes : Une collaboration fluide montre des séquences d'actions prévisibles et complémentaires (faible entropie).
- Taux de chevauchement des zones de travail : Un bon alignement devrait réduire les interférences spatiales.
- Temps d'inactivité mutuel : Mesurer les temps où l'un attend l'autre.
3. Démarche de travail
Phase 1 : Collecte de données et modélisation initiale. Il s’agit d’enregistrer les actions, les regards et les décisions de plusieurs opérateurs effectuent la même tâche (d'assemblage par exemple) pour construire un modèle de base des variations de style.
Phase 2 : Développement de l'algorithme d'apprentissage incrémental. Il s’agit d’implémenter et d’adapter des algorithmes d'ARI en ligne. Puis concevoir un mécanisme d'intégration des feedbacks (exemple : un signal vocal "stop" remet à zéro la dernière action et fournit une forte récompense négative).
Phase 3 : Conception du générateur de comportements alignés. Il s’agit d’entraîner un modèle CVAE sur les données de la Phase 1 pour qu'il apprenne à générer des trajectoires conditionnées par un style. Puis intégrer ce générateur dans le planificateur de trajectoires du robot.
Phase 4 : Validation expérimentale. Il s’agit dans cette phase de réaliser :
- Une étude longitudinale où un opérateur travaille avec le cobot sur plusieurs sessions.
- Un groupe de contrôle : Le robot a un comportement standard et fixe.
- Un groupe expérimental : Le robot utilise le cadre d'apprentissage incrémental et d'alignement.
- Une mesure de la courbe d'apprentissage, de l'évolution des métriques de fluidité et des questionnaires de satisfaction.
4. Originalité et impact de l’étude :
- Nouveau cadre formel pour l'apprentissage incrémental des préférences humaines en contexte collaboratif physique.
- Algorithme d'ARI interactif et robuste.
- Modèle génératif pour la synthèse de comportements robotiques personnalisés.
Cette étude permet d’avoir différents impacts sur :
- Réduction du temps de configuration pour un nouvel opérateur ou un nouveau produit.
- Augmentation du confort et de l'acceptation car le robot s'adapte à l'humain, et non l'inverse.
- Robustesse aux variations individuelles, rendant la cobotique accessible à un plus large éventail de profils.

Mots clés : intelligence artificielle, robotique, cobotique, apprentissage à long terme, interaction homme-machine.

Bibliographies :
Avaei, A.; van der Spaa, L.; Peternel, L.; Kober, J. An Incremental Inverse Reinforcement Learning Approach for Motion Planning with Separated Path and Velocity Preferences. Robotics 2023, 12, 61. https://doi.org/10.3390/robotics12020061
Deshpande, S., Walambe, R., Kotecha, K. et al. Advances and applications in inverse reinforcement learning: a comprehensive review. Neural Comput & Applic 37, 11071–11123 (2025). https://doi.org/10.1007/s00521-025-11100-0
Urrea, C. Hybrid Deep Learning-Reinforcement Learning for Adaptive Human-Robot Task Allocation in Industry 5.0. Systems 2025, 13, 631. https://doi.org/10.3390/systems13080631
Ahmad Farooq and Kamran Iqbal ; A Survey of Reinforcement Learning for Optimization in Automation; arXiv:2502.09417v1, 13 Feb 2025




