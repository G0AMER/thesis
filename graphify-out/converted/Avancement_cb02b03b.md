<!-- converted from Avancement.docx -->

# Rapport d’Avancement de Thèse
## 1. Contexte général et problématique
Les systèmes robotiques autonomes ont connu des avancées significatives ces dernières années, notamment en matière de contrôle adaptatif, de planification autonome et d’apprentissage automatique. Toutefois, une limite persistante concerne leur capacité à s’adapter de manière incrémentale dans des environnements dynamiques et dans des contextes d’interaction collaborative avec l’humain.
Les travaux récents en apprentissage sûr pour la robotique mettent en évidence la nécessité d’intégrer des méthodes issues du contrôle optimal et du reinforcement learning tout en garantissant des contraintes de sûreté, particulièrement en environnement partagé homme–robot. À ce titre, la revue Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning
https://doi.org/10.1146/annurev-control-042920-093225
souligne les défis liés à la stabilité, à la robustesse et aux garanties formelles lors de l’intégration de mécanismes d’apprentissage dans des systèmes physiques.
Par ailleurs, l’apprentissage incrémental (continual learning) est identifié comme un axe stratégique pour dépasser les limites des modèles classiques, notamment le phénomène d’oubli catastrophique lors de l’acquisition séquentielle de nouvelles données (cf.
https://arxiv.org/abs/1807.11154).
Ces approches sont essentielles dans des contextes où le robot doit évoluer continuellement sans réentraînement complet.
En robotique collaborative (Human–Robot Collaboration — HRC), les travaux de synthèse montrent que les robots doivent être capables d’interpréter les intentions humaines, d’adapter leur comportement et de partager un objectif commun, ce qui nécessite des modèles adaptatifs intégrant apprentissage et perception multimodale (voir par exemple
https://doi.org/10.1177/0278364917690593
et
https://doi.org/10.1109/TCDS.2024.3454779).
Des travaux récents en robotique sociale et adaptative mettent également en évidence l’importance de l’apprentissage continu dans les interactions prolongées (https://doi.org/10.3389/frobt.2024.1455375).
### Problématique générale identifiée
Comment améliorer l’adaptabilité des robots autonomes et collaboratifs en intégrant des mécanismes d’apprentissage incrémental compatibles avec :
les contraintes temps réel embarquées,
les exigences de sûreté et de sécurité,
et les spécificités des interactions humain–robot en environnement partagé ?
## 2. Travaux réalisés : continuité du Projet de Fin d’Études
La première phase des travaux a consisté en la conception et la validation d’un robot mobile autonome dans le cadre du Projet de Fin d’Études.
Ces travaux ont conduit à la publication de l’article :
“Design and Development of an Autonomous Mobile Robot for Unstructured Indoor Environments”
https://doi.org/10.3390/machines13111044
Ce papier présente :
la conception mécanique et électronique de la plateforme,
l’intégration de capteurs (LiDAR, IMU, encodeurs),
l’architecture logicielle sous ROS 2,
la navigation autonome en environnement intérieur non structuré,
la détection d’objets en temps réel.
Cette contribution constitue la plateforme expérimentale de référence pour la thèse.
À ce stade, les travaux portent principalement sur l’architecture robotique et l’autonomie navigationnelle. Aucun mécanisme d’apprentissage incrémental n’a encore été implémenté.
## 3. Évolution vers la robotique collaborative
Après la publication de l’article initial, les travaux ont évolué vers une analyse approfondie de la robotique collaborative.
Les systèmes HRC impliquent :
la co-présence humain–robot,
la coordination de tâches,
la gestion des distances de sécurité,
l’interprétation d’intentions humaines.
Les revues structurantes du domaine (https://doi.org/10.1177/0278364917690593) mettent en évidence plusieurs niveaux de collaboration : coexistence, coopération et collaboration complète.
Les travaux récents en interaction cognitive et sociale (https://doi.org/10.1109/ACCESS.2024.3451663) soulignent l’importance de modèles adaptatifs intégrant perception multimodale, raisonnement contextuel et apprentissage dynamique.
Ces analyses confirment que la collaboration homme–robot dépasse la simple planification de trajectoire et nécessite des mécanismes d’adaptation comportementale intégrés.
## 4. Recherche approfondie sur l’apprentissage adaptatif en robotique
### 4.1 Apprentissage incrémental et apprentissage par renforcement en robotique
Une étude bibliographique détaillée a été menée sur les approches d’apprentissage adaptatif applicables aux systèmes robotiques autonomes, incluant :
Les architectures dynamiques extensibles et les méthodes de croissance de réseau (Progressive Neural Networks, DEN, PackNet) pour isoler ou étendre les paramètres spécifiques aux nouvelles tâches.
https://doi.org/10.3390/electronics10161879
https://doi.org/10.48550/arXiv.2103.16788	
https://openaccess.thecvf.com/content_cvpr_2018/papers/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.pdf
Les méthodes de régularisation (EWC, SI, MAS, LwF) qui pénalisent les modifications des paramètres critiques afin de limiter l’oubli catastrophique.
 https://arxiv.org/pdf/2105.04093
 https://arxiv.org/pdf/1703.04200
 https://arxiv.org/pdf/1606.09282
 https://arxiv.org/pdf/1711.09601
Les mécanismes de mémoire explicite (replay) et distillation de connaissances, incluant des algorithmes comme iCaRL, GDumb, DER / DER++ et les méthodes de replay génératif (Generative Replay) qui conservent les informations essentielles des tâches précédentes.
 https://arxiv.org/pdf/1611.07725
 https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf
 https://arxiv.org/pdf/2103.16788
Les stratégies basées sur les prompts et modules adaptateurs (Prompt Tuning, Adapter Modules) pour apprendre de nouvelles tâches avec un impact minimal sur les paramètres préexistants.
https://arxiv.org/abs/2503.07979
Les approches d’apprentissage par renforcement adaptées aux environnements dynamiques et séquentiels, intégrant dans certains cas l’Inverse Reinforcement Learning pour déduire des fonctions de récompense à partir de comportements experts.
En complément des travaux spécifiques à l’apprentissage continu, des contributions récentes disponibles sur arXiv (https://arxiv.org/pdf/2404.18713v1 ; https://doi.org/10.48550/arXiv.2209.11908) analysent l’intégration de mécanismes d’apprentissage par renforcement (Reinforcement Learning) et, dans certains cas, d’apprentissage par renforcement inverse (Inverse Reinforcement Learning) dans des architectures robotiques autonomes.
Ces travaux mettent en évidence :
les enjeux de stabilité et de convergence de l’apprentissage,
les difficultés de généralisation entre simulation et environnement réel,
la gestion de l’exploration sous contraintes de sûreté,
les exigences de robustesse lors du déploiement sur plateformes physiques,
ainsi que les contraintes computationnelles et énergétiques propres aux systèmes embarqués.
L’ensemble de ces éléments souligne la nécessité de concevoir des mécanismes d’apprentissage compatibles avec les exigences opérationnelles de la robotique réelle, en particulier lorsque l’apprentissage doit être intégré dans des environnements interactifs et potentiellement collaboratifs.
### 4.2 Adaptation comportementale des robots
L’article suivant :
https://doi.org/10.1016/j.chbah.2023.100018
analyse expérimentalement l’impact de l’adaptation comportementale excessive d’un robot social sur la perception utilisateur. Les résultats montrent que l’adaptation doit être contrôlée et contextualisée.
Par ailleurs, des travaux en robotique bio-inspirée et adaptative (https://doi.org/10.1038/s42003-021-02891-8) illustrent comment des mécanismes adaptatifs inspirés du vivant peuvent améliorer la robustesse comportementale.
## 4.3 Datasets identifiés
Dans le cadre de la revue bibliographique, plusieurs datasets pertinents pour l’apprentissage incrémental et la robotique collaborative ont été identifiés :
https://arxiv.org/abs/1807.11154 (benchmarks d’apprentissage continu)
https://www.mdpi.com/2218-6581/14/12/184 (plateformes robotiques expérimentales et données associées)
https://doi.org/10.3390/robotics14120176 (environnements robotiques collaboratifs et jeux de données)
Ces ressources pourront servir de base :
pour l’évaluation comparative d’algorithmes,
pour la validation expérimentale,
pour la reproductibilité scientifique.
## 5. Contribution actuelle
À ce stade, les contributions de la thèse peuvent être résumées comme suit :
Publication d’un article scientifique décrivant une plateforme robotique autonome fonctionnelle (https://doi.org/10.3390/machines13111044).
Identification des limites des systèmes robotiques classiques en matière d’adaptabilité incrémentale.
Définition d’une problématique scientifique centrée sur l’intégration de l’apprentissage incrémental dans des architectures collaboratives.
Analyse détaillée des approches théoriques existantes en :
apprentissage sûr,
robotique collaborative,
apprentissage continu.
Identification de jeux de données pertinents pour la future phase expérimentale.
À ce jour, aucun développement algorithmique spécifique n’a encore été engagé ; les travaux réalisés sont exclusivement théoriques, analytiques et positionnement scientifique.
## 6. État d’avancement
Au moment du présent rapport :
Le travail initial (robot mobile autonome) est achevé et publié.
La problématique scientifique est formalisée.
Une revue bibliographique approfondie des thématiques clés a été menée.
Des datasets pertinents ont été identifiés.
Aucune implémentation ni expérimentation en apprentissage incrémental n’a encore été réalisée.
## 7. Perspectives
Les prochaines étapes prévues sont :
Sélection et formalisation d’un cadre d’apprentissage incrémental adapté aux contraintes embarquées.
Définition de scénarios HRC concrets (co-manipulation, navigation partagée, adaptation comportementale).
Conception d’une architecture logicielle intégrant :
perception multimodale,
interaction humain–robot,
mécanismes d’apprentissage progressif.
Implémentation et validation expérimentale sur la plateforme développée.
Évaluation comparative sur datasets identifiés et scénarios réels.