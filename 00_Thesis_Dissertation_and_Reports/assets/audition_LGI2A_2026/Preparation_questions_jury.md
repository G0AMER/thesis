# Audition LGI2A, sujet n°329220 : préparation des questions (10 min)

Réponses courtes, 30 à 60 secondes chacune. Toujours relier la réponse à une phase du sujet ou à un travail déjà fait.

## Sujet et méthode

**1. Pourquoi un apprentissage incrémental et pas un modèle entraîné une fois pour tous les opérateurs ?**
Parce que les opérateurs arrivent les uns après les autres et que leur style évolue avec la pratique. Réentraîner sur toutes les données à chaque nouvel opérateur coûte cher et n'est pas toujours possible (données personnelles, mémoire embarquée). Le risque principal est l'oubli catastrophique : c'est exactement ce que FLAIR traite, avec un oubli F = 0,017 sur HARMONIC pour 1,28 Mo de mémoire.

**2. Comment évitez-vous l'oubli catastrophique ?**
Trois mécanismes complémentaires dans FLAIR : un petit tampon de rejeu des exemples passés, une régularisation de Fisher (type EWC) qui protège les poids importants, et des modulations FiLM spécifiques à chaque opérateur sur une base partagée. Pour la thèse, la même logique s'applique à la fonction de récompense apprise par IRL : une base commune plus une composante propre à chaque opérateur.

**3. Pourquoi l'apprentissage par renforcement inverse (IRL) ?**
On ne connaît pas la récompense de l'opérateur, on observe ses démonstrations et ses corrections. L'IRL infère cette récompense, ce qui est plus transférable qu'une politique copiée. La référence du sujet (Avaei et al., 2023) montre une IRL incrémentale qui sépare préférences de chemin et de vitesse : c'est un bon point de départ.

**4. Comment intégrer un feedback comme le « stop » vocal ?**
Comme une correction forte : annulation de la dernière action et récompense très négative sur la paire état-action concernée, ce qui met à jour la récompense en ligne. Point important : la sécurité ne doit jamais dépendre de l'apprentissage. Le stop déclenche d'abord un arrêt immédiat par une couche de sécurité indépendante, puis sert de signal d'apprentissage.

**5. Pourquoi un CVAE pour générer les comportements ?**
Il apprend un espace latent structuré conditionné par un vecteur de style, fonctionne avec relativement peu de données, et l'échantillonnage est rapide, compatible avec une planification en temps réel. Les modèles de diffusion sont plus expressifs mais plus lourds ; ils peuvent être une comparaison intéressante en fin de thèse.

**6. Comment garantir la sécurité d'un robot qui apprend ?**
En séparant l'apprentissage de la sécurité : un filtre de sécurité (surveillance vitesse et distance, limitation puissance et force selon ISO/TS 15066) contraint toutes les trajectoires générées. L'apprentissage choisit uniquement parmi les actions admissibles. Mon travail sur la détection d'états dangereux (plus de 98,6 % de macro-F1 sur capteurs inertiels) peut fournir un signal de danger supplémentaire.

**7. Quelles métriques de fluidité ?**
Les métriques classiques de Hoffman : temps d'inactivité de l'humain et du robot, proportion d'activité simultanée, délai fonctionnel entre la fin d'une action humaine et la réaction du robot. S'y ajoutent le temps de cycle, la courbe d'apprentissage sur les sessions, et des questionnaires : satisfaction, confiance, charge de travail (NASA-TLX).

**8. Comment collecter les données de la phase 1 ?**
Squelette ou vidéo RGB-D pour les actions, eye-tracker pour les regards, journaux du robot pour les décisions et l'état. Plusieurs opérateurs sur une même tâche d'assemblage, avec consentement et avis éthique. Pour démarrer avant la collecte, j'utiliserais des jeux publics que je connais déjà : HARMONIC, DASIG, MultiPhysio-HRC.

**9. Comment organiser l'étude longitudinale de la phase 4 ?**
Deux groupes appariés, robot fixe contre robot adaptatif, plusieurs sessions par opérateur sur la même tâche. On compare l'évolution des métriques de fluidité et de satisfaction au fil des sessions, pas seulement la valeur finale. Simulation d'abord pour régler le système, puis cobot réel.

**10. Que pensez-vous de l'état de l'art cité dans le sujet ?**
Avaei et al. traitent l'IRL incrémentale pour la planification de mouvement mais sur un seul utilisateur ; Urrea combine apprentissage profond et RL pour l'allocation de tâches en industrie 5.0 sans modéliser le style individuel. Le manque que vise la thèse : une adaptation continue à plusieurs opérateurs, avec génération de trajectoires conditionnée par le style et validée dans la durée.

## Profil et parcours

**11. Le sujet demande un master en IA. Votre diplôme correspond-il ?**
Le diplôme d'ingénieur de l'ENIS est un bac+5, de grade équivalent au master. Mon cursus en génie informatique a inclus l'apprentissage automatique, et je l'ai complété par la pratique : apprentissage continu, adaptation de domaine, réseaux profonds sur signaux et squelettes.

**12. Quelles sont les limites de vos travaux actuels ?**
FLAIR est évalué hors ligne sur un jeu public, pas encore en boucle fermée avec un vrai opérateur. La détection de danger est aussi évaluée hors ligne. La reconnaissance de tâche (93,71 % d'exactitude) repose sur un découpage 80/20 et non sur une évaluation par sujet exclu ; l'article est soumis à ICAART 2027. Ce sont exactement les étapes que la thèse doit franchir : boucle fermée, robot réel, étude utilisateur.

**13. Pourquoi vous plutôt qu'un autre candidat ?**
Parce que je combine les deux moitiés du sujet : la robotique réelle (un robot conçu, intégré sous ROS2 et publié) et l'apprentissage incrémental (un algorithme présenté à KES 2026). Et j'ai déjà construit les pipelines de données sur des jeux de collaboration humain-robot, donc je peux être productif dès les premiers mois.

**14. Vous travaillez déjà sur ce thème à Sfax. Qu'apporte la cotutelle ?**
À adapter à votre situation exacte. Piste : le LGI2A apporte l'automatique et une équipe IA, un accès à un environnement expérimental et un double encadrement ; la cotutelle donne un cadre formel et un double diplôme à un travail que j'ai commencé avec le Pr Ksantini.

## Organisation et projet

**15. Comment voyez-vous le calendrier ?**
Le financement court du 1er octobre 2026 au 31 décembre 2028, donc il faut aller vite. Première année : état de l'art, protocole, données, première version de l'IRL incrémentale en simulation. Deuxième année : générateur CVAE, intégration au planificateur, essais pilotes. Fin : étude longitudinale, publications, rédaction.

**16. Comment gérez-vous la cotutelle et la distance ?**
Un séjour annuel en France, des points réguliers en visioconférence avec les deux équipes, un dépôt de code commun et des rapports d'avancement écrits. J'ai déjà travaillé ainsi entre Sfax et l'ESME à Paris.

**17. Pourquoi l'enseignement dans votre projet ?**
J'enseigne déjà à l'ESC Sfax et cela m'oblige à expliquer clairement, ce qui aide aussi la recherche. Mon objectif est un poste d'enseignant-chercheur, après un post-doctorat et la qualification.

## Parcours personnel

**18. Parlez-nous de vos compétitions de robotique.**
Concours nationaux de robots suiveurs de ligne : 1re place à ESPRIT ROBOTS 2.0 et à INSAT ROBOLYMPICS 3.0, 2e place à ENIS ROBOTS 5.0. Préciser votre rôle dans l'équipe (conception, électronique, programmation, réglage de la commande) et une difficulté surmontée le jour de l'épreuve. Message à faire passer : savoir livrer un système fiable sous contrainte de temps.

**19. Que vous a apporté la vie associative ?**
Membre du bureau IEEE RAS de la branche étudiante IEEE de l'ENIS, et développeur au Microsoft Tech Club ENIS : organiser des activités, travailler avec des profils différents, transmettre des connaissances. Ce sont des compétences directement utiles en thèse, notamment en cotutelle avec deux équipes.

**20. Comment conciliez-vous travail intense et équilibre personnel ?**
Le sport, football et natation, et la musique, au piano, m'aident à garder un rythme régulier sur la durée, ce qui est important pour un projet de trois ans.

## Publications

**21. Où en sont vos publications ?**
Publiés : Machines (MDPI) 2025 pour le robot mobile, et FLAIR, présenté à KES 2026 et publié dans Procedia Computer Science. En attente : l'article sur la détection de danger, soumis au Journal of Intelligent & Robotic Systems (Springer Nature), et l'article sur la reconnaissance de tâche, soumis à ICAART 2027.
