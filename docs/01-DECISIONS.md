# Hydra — Décisions actées

Chaque entrée est datée, motivée, et indique ce qu'il faudrait observer pour la remettre en cause.

---

## D-001 — Hydra est une preuve de concept ; le débit n'est pas un critère d'acceptation
**2026-08-05 — validé**

L'objectif est de **démontrer qu'un modèle qui ne rentre pas en mémoire peut y tourner**. Les
3,4–4,4 tok/s projetés pour GPT-OSS 120B sur M4/24 Gio sont acceptés comme résultat, pas subis
comme échec.

**Conséquences directes sur le plan de phases :**
- Les seuils de débit disparaissent des critères GO/NO-GO. Le jalon 2.5 devient *« génère une sortie
  correcte en restant dans le budget mémoire »*, et le débit est une **mesure publiée**, pas une barrière.
- Le point de décision après la trace de routage (jalon 1.7) est **supprimé en tant que blocage**.
  La courbe hit/slots reste produite — elle sert à dimensionner le cache et à prédire les machines
  cibles — mais un taux de hit faible ne suspend plus le projet.
- L'ordre de priorité devient sans ambiguïté : **invariant mémoire > correction > débit.**

---

## D-017 — Réglages d'échantillonnage : s'écarter de la recommandation d'OpenAI
**2026-08-05 — décidé sur observation**

OpenAI recommande `temperature = 1.0` et `top_p = 1.0` pour GPT-OSS, c'est-à-dire la
distribution brute sans troncature. **On s'en écarte** : par défaut, 0,7 et 0,9.

**Ce qui l'a motivé.** Sur l'invite « hi », le 20B a raisonné ainsi : *« User says hi. Need
to respond in Bengali because user asked earlier to translate. »* — puis a répondu en
bengali. Aucune consigne de ce genre n'existait ; le modèle a halluciné un antécédent.

**Le gabarit n'est pas en cause.** L'invite Harmony rendue est identique au
`chat_template.jinja` publié, et des tests le verrouillent. Le problème vient de
l'échantillonnage : sur une invite de deux jetons, la distribution brute d'un modèle de
20 B laisse une masse notable à des continuations aberrantes, et rien dans le contexte ne
vient les écarter.

**Ce que ça reste.** Un réglage, pas une contrainte : le curseur monte jusqu'à 1,5, et la
recommandation d'OpenAI reste atteignable. Un champ de consignes développeur est également
exposé, qui permet d'épingler une langue de réponse — c'est le mécanisme prévu par le
format pour cela.

---

## D-016 — Périmètre de la première application
**2026-08-05 — validé**

**La jauge mémoire est l'élément central.** Elle affiche les gigaoctets réellement en
mémoire face au poids total du modèle. Trois grandeurs distinctes, jamais confondues :
mémoire engagée par le processus, poids mappés (repris par le système sous pression),
taille installée sur disque. Annoncer un seul chiffre flatteur serait contraire à
l'honnêteté du projet.

**Écartés :** le visualiseur de slots d'experts (spectaculaire mais coûteux en place pour
peu d'information) et l'affichage du taux de hit et des lectures SSD (pertinents pour
régler, pas pour démontrer).

**Retenus :** historique multi-session persistant, raisonnement dans un panneau
déroulant, et réglages exposés — longueur de contexte et nombre de slots d'experts au
chargement, température et effort de raisonnement par conversation. Le nombre de slots est
exposé délibérément : c'est lui qui rend l'arbitrage mémoire/vitesse tangible.

**Catalogue figé** aux deux modèles officiels d'OpenAI. Accepter un dépôt Hugging Face
quelconque ouvrirait la porte à des architectures non supportées, qu'il faudrait détecter
et refuser proprement — du travail sans rapport avec l'objectif.

**Installation en arrière-plan pendant une conversation : oui.** C'est la voie facile
*et* la correcte : installer ne mobilise aucun runtime, seulement de l'I/O bornée à
~50 Mio. En revanche **un seul modèle est chargé en inférence à la fois** — deux runtimes
concurrents doubleraient l'empreinte, à rebours du projet.

**Distribution :** build depuis les sources, publié sur GitHub. Pas de notarisation pour
l'instant. Le CLI est conservé : c'est l'outil de mesure du projet.

---

## D-015 — Ne pas dégrader le modèle : l'objectif est l'intelligence à empreinte réduite, pas l'empreinte minimale
**2026-08-05 — validé, précise D-012**

**Refusé : quantifier les poids denses.** Passer l'attention, les routeurs et la tête LM de BF16
en MXFP4 économiserait 1,67 Gio, soit 73 % du plancher résident. C'est le plus gros gain mémoire
disponible, et il est **écarté** s'il coûte de la qualité.

**Ce que le projet démontre, exactement.** Pas « faire tenir un LLM dans peu de mémoire » — n'importe
quelle quantization agressive y parvient, au prix d'un modèle abruti et sans intérêt. Mais :
**faire tourner un gros modèle *intelligent* sur une petite configuration, sans le dégrader.**

Le streaming d'experts est précisément ce qui rend les deux compatibles. On réduit l'empreinte en
limitant le nombre d'experts **résidents**, pas en abîmant les poids. Hydra exécute GPT-OSS
**dans le format exact publié par OpenAI** : experts en MXFP4 natif, attention et tête LM en BF16.
Aucune requantization, aucune perte introduite par nous.

**Conséquence pratique.** Le plancher résident reste à 2,27 Gio pour le 20B et 2,88 Gio pour le
120B, et c'est assumé. L'objectif de 3,68 Gio d'empreinte pour un modèle de 12,82 Gio est déjà
atteint ; le grignoter au prix de la qualité serait contraire au but.

**À rouvrir si :** une mesure montre qu'une quantization donnée est **sans effet mesurable** sur la
qualité. La charge de la preuve est de ce côté-là, et elle est exigeante : pas « l'écart numérique
est petit », mais « les sorties restent équivalentes sur une évaluation sérieuse ».

**Reste autorisé** : toute optimisation qui ne touche pas aux poids. Le prefill par blocs et le
recouvrement I/O en font partie — ils changent l'ordonnancement, pas les valeurs.

---

## D-014 — Sémantiques d'opérateurs de GPT-OSS qui ne se devinent pas
**2026-08-05 — vérifié sur `gpt_oss/torch/model.py`**

Chacun de ces points a été relevé dans le code de référence, pas déduit. Se tromper sur
l'un d'eux ne lève **aucune erreur** : le modèle charge, génère du texte plausible, et
sort dégradé. Ils sont figés par des vecteurs de référence dans `tools/gen_reference_fixtures.py`.

**Le SwiGLU découpe en indices pairs et impairs**, pas en deux moitiés :
`x_glu, x_linear = x[..., ::2], x[..., 1::2]`. Les lignes de `gate_up_proj` sont donc
**entrelacées** `[gate₀, up₀, gate₁, up₁, …]`. Découper en deux moitiés donnerait un
modèle qui fonctionne mais mélange les canaux.

**Le RoPE, lui, découpe bien en deux moitiés** (`torch.chunk`). Deux conventions opposées
dans la même architecture — c'est précisément ce qui rend l'erreur facile.

**L'écrêtage du SwiGLU est asymétrique** : la branche gate est bornée **seulement par le
haut** (`min(x, 7)`), la branche linéaire des deux côtés. Et la branche linéaire reçoit
**+1** avant le produit. Le swish utilise `sigmoid(1,702·x)`, pas `sigmoid(x)`.

**YaRN applique une concentration** en plus du réétalement des fréquences :
`0,1·ln(facteur) + 1`, soit **1,3466** pour GPT-OSS. Elle multiplie cos et sin. L'omettre
ne casse rien de visible mais décale toute l'attention.

**Les puits d'attention sont une colonne de logits supplémentaire** dans le softmax,
retirée après. Ils n'apportent aucune valeur au résultat : ils grossissent le
dénominateur, ce qui permet à une tête de ne rien regarder. Dans le noyau, cela se
traduit élégamment — le softmax en ligne démarre avec `max = puits` et `dénominateur = 1`.

**Le routeur applique son softmax aux seuls top-k logits**, après sélection, pas à la
distribution complète.

**La fenêtre glissante vaut 128 et s'applique aux couches d'indice pair.** Le masque
`tril(diagonal=-128)` autorise exactement 128 positions.

---

## D-013 — La borne mémoire du repacker vient du streaming, pas du découpage des requêtes
**2026-08-05 — décidé sur mesure, corrige un choix initial**

**Ce que j'avais choisi.** Découper chaque plage source en sous-requêtes de 4 Mio, pour que la
borne mémoire soit une propriété du découpage — trivialement vérifiable — plutôt qu'une dépendance
au comportement de mise en tampon d'`URLSession`.

**Ce que la mesure a dit.** Sur le vrai dépôt :

| Motif | Débit |
| --- | ---: |
| une requête `Range` de 64 Mio | **33,5 Mo/s** |
| huit requêtes `Range` de 4 Mio en série | **5,2 Mo/s** |

Soit un facteur **6,4**. La cause : Hugging Face répond un **302 vers un CDN signé**, et chaque
requête repaie la redirection puis une poignée TLS vers un autre hôte. Réutiliser l'URL résolue est
impossible — sa policy contient une condition `ByteRange` liée à la plage exacte demandée.

**Ce qu'on fait à la place.** Une **seule requête par région contiguë** du checkpoint source, dont
la réponse est **consommée au fil de l'eau** : chaque bloc livré par la pile réseau est routé vers
sa destination puis relâché avant l'arrivée du suivant. Le plan couvrant exactement le checkpoint
sans trou, les tenseurs voisins forment de longues régions — le fichier source est lu quasiment de
bout en bout, séquentiellement.

**Résultat mesuré sur l'installation réelle du 20B : 44 Mo/s**, soit **8,5×** l'approche initiale.

**Ce que ça coûte.** La borne n'est plus une propriété du découpage : elle dépend de la taille des
blocs que livre `URLSession` (mesurés jusqu'à 3,5 Mio). Elle est donc désormais **vérifiée par les
tests et instrumentée en production** — le repacker suit le plus gros bloc reçu et l'expose dans sa
progression — plutôt que garantie par construction. C'est un compromis assumé : une borne mesurée
à chaque exécution vaut mieux qu'une borne théorique qui divise le débit par six.

---

## D-012 — Minimiser la mémoire est l'objectif, pas remplir le plafond disponible
**2026-08-05 — validé, corrige D-001**

Le cache d'experts n'est **jamais** dimensionné par « ce que le matériel autorise ». C'est une
**politique explicite** (`ExpertCachePolicy`), et le défaut est `.minimal` : un slot par expert
sélectionné, soit `top_k = 4` par couche.

**Ce que ça corrige.** L'étude de faisabilité présentait « GPT-OSS 20B tient entièrement en
mémoire, donc pas besoin de streaming » comme une bonne nouvelle. C'est un contresens sur
l'objectif du projet : le 20B doit lui aussi tourner en empreinte réduite, sinon il ne démontre
rien. Il reste utilisable en mode entièrement résident, mais **comme référence de correction**,
pas comme mode de fonctionnement.

**Ce que ça donne, mesuré par `hydra budget` :**

| Modèle | Politique | Empreinte | Part du modèle installé |
| --- | --- | ---: | ---: |
| 20B (12,82 Gio installés) | `.minimal` | **3,77 Gio** | 28 % |
| 20B | `.maximize` (référence) | 12,06 Gio | 93 % |
| 120B (60,77 Gio installés) | `.minimal` | **5,07 Gio** | **8 %** |

**Le test de correction qui en découle**, et qui est le meilleur du projet : à prompt identique et
décodage glouton, `.minimal` et `.maximize` doivent produire **exactement la même séquence de
tokens** sur le 20B. La taille du cache est une caractéristique de performance ; elle ne doit avoir
aucun effet observable sur les sorties. Toute divergence signale un bug d'éviction ou de propriété
de slot.

**Corollaire sur la portabilité** : rien dans le dimensionnement n'est spécifique à la machine de
développement. `HardwareProfile` est injecté, et un test vérifie que le 20B tient au minimum sur un
plafond de 5 Gio — soit une machine de 8 Gio.

**Découverte associée : le plancher n'est plus les experts, ce sont les poids résidents.**
GPT-OSS garde attention, routeurs et tête LM en **BF16 non quantifié** — 2,27 Gio pour le 20B,
2,88 Gio pour le 120B, dont 1,08 Gio pour la seule tête LM. C'est ce qui explique que
TurboFieldfare atteigne ~2 Go au total sur Gemma 4 alors que nous plafonnons vers 3,8 Gio : chez
eux, ces mêmes tenseurs sont en 4 bits. **Descendre plus bas exigerait de quantifier la tête LM et
l'attention**, ce qui modifie les sorties — donc une expérience à valider contre référence, pas une
décision de conception. C'est le principal levier restant.

---

## D-002 — La portabilité est un objectif déclaré, mais aucune abstraction n'est écrite d'avance
**2026-08-05 — validé**

Cibles à terme évoquées : M3 Ultra / M5 Max sous-dotés en mémoire unifiée, et un portage
**x86_64 + CUDA** permettant à une RTX 5090 (32 Gio de VRAM) de faire tourner des modèles calibrés
pour une RTX Pro 6000.

**Ce qu'on fait maintenant :** rien de spéculatif. Pas de protocole `ComputeBackend`, pas de couche
d'abstraction GPU. Conformément au brief, on ne généralise qu'à partir de code qui marche.

**Ce qu'on s'interdit en revanche dès maintenant**, parce que c'est gratuit et que l'inverse coûte
cher à défaire : **les modules `HydraCore`, `HydraFormat`, `HydraInstall` et `HydraTokenize`
n'importent pas Metal.** La logique de format, de streaming, de cache et de tokenisation reste du
Swift pur, portable tel quel sur Linux et Windows. Seuls `HydraMetal`, `HydraRuntime` et `HydraApp`
sont liés à la plateforme. C'est une discipline de couches, pas une abstraction.

**Note technique pour le futur portage CUDA** — à ne pas implémenter, seulement à garder en tête :
une machine à GPU discret offre une hiérarchie à **trois niveaux** (VRAM / RAM système / SSD) là où
le Mac n'en a que deux. C'est structurellement **plus favorable** : une 5090 dispose de 32 Gio de
VRAM, d'une centaine de Gio de RAM système utilisable comme cache de second niveau, et d'une bande
passante mémoire d'un ordre de grandeur supérieure à celle du M4. Le goulot y redevient le SSD et
le bus PCIe, pas le calcul.

---

## D-003 — Pas de troisième modèle pour l'instant
**2026-08-05 — validé**

Qwen3.6-35B-A3B est écarté du périmètre initial : ses 30 couches Gated DeltaNet représentent
autant de travail de noyaux que les deux GPT-OSS réunis.

Le périmètre devient **GPT-OSS 20B puis GPT-OSS 120B**. La phase 4 (troisième modèle) est retirée
du plan ; la phase 3 (généralisation) sera menée sur la base de deux modèles réels, et le choix
d'une troisième cible se fera plus tard, avec du recul.

**À rouvrir si :** la phase 3 montre que deux modèles de la même famille ne suffisent pas à
dégager les bons axes de variation pour le contrat de modèle.

---

## D-004 — Gestion du stockage à la manière de LM Studio
**2026-08-05 — validé**

L'utilisateur installe et désinstalle les modèles depuis l'interface, en connaissance de cause.
Zéro, un ou plusieurs modèles peuvent coexister.

**Seule règle automatique :** Hydra calcule l'espace restant **après** l'installation envisagée et
**avertit** s'il tomberait sous **10 Go**. C'est un avertissement, pas un blocage — l'utilisateur
reste décisionnaire.

L'interface affiche pour chaque modèle : taille sur disque, état (installé / partiel / absent),
espace libre courant.

---

## D-005 — Longueur de contexte choisie au chargement du modèle
**2026-08-05 — validé**

Comme LM Studio : au moment de charger un modèle, une fenêtre propose la longueur de contexte.

Conséquence technique importante : **le nombre de slots du cache d'experts est calculé au
chargement**, pas figé à la compilation. Le budget est dérivé à chaud de
`recommendedMaxWorkingSetSize`, du contexte choisi et de la taille des poids résidents. L'interface
affiche le nombre de slots obtenus et le débit attendu **avant** de confirmer.

Valeurs proposées : 4k, 8k, 16k, 32k, 64k, 128k. Défaut : 32k.

---

## D-006 — Plafond Metal : détecter et proposer, jamais imposer
**2026-08-05 — recommandation, non contestée**

Hydra lit `recommendedMaxWorkingSetSize` et calcule tout dessus. Il **détecte** si
`iogpu.wired_limit_mb` a été relevé et en tient compte. Il **propose** la commande à l'utilisateur,
documentée, avec son effet chiffré (+5 slots/couche sur le 120B) et son risque. Il ne l'exécute
jamais lui-même et fonctionne correctement au budget par défaut.

---

## D-007 — macOS 26 minimum, Apple Silicon uniquement
**2026-08-05 — recommandation, à contester si besoin**

Motifs : c'est la version de la machine de développement et de validation ; elle donne accès à
Metal 4 et aux Metal Performance Primitives pour le prefill ; et supporter des versions antérieures
signifierait valider sur du matériel dont nous ne disposons pas.

Le code détecte la famille GPU à l'exécution. **Cette machine est apple9, pas apple10** : le chemin
TensorOps qui accélère l'attention en contexte long chez TurboFieldfare ne nous est pas accessible.

---

## D-008 — Harmony réimplémenté en Swift, sous harnais de conformité
**2026-08-05 — recommandation**

La bibliothèque officielle est en Rust. Plutôt que d'introduire Rust dans la chaîne de build, on
réimplémente en Swift et on fige un corpus de conversations rendues par la bibliothèque officielle
comme **fixtures**. Le test exige l'égalité **octet à octet** du rendu.

Une divergence Harmony dégrade les sorties **sans lever d'erreur** : c'est exactement le genre de
bug qu'il faut rendre impossible par construction.

---

## D-009 — Pas de chemin d'exécution dense
**2026-08-05 — recommandation**

Hydra est un moteur MoE à streaming et l'assume comme limite volontaire. Un chemin dense résident
serait un second runtime pour un usage déjà bien couvert par MLX et llama.cpp, sans apport
différenciant.

---

## D-010 — Source Hugging Face : le dépôt racine
**2026-08-05 — décidé sur l'audit**

Pour GPT-OSS, le repacker lit la **racine** du dépôt (14 shards pour le 120B), pas `original/` ni
`metal/model.bin`. Les valeurs MXFP4 y sont déjà séparées en `blocks` / `scales` / `bias`, chaque
sous-tenseur étant une plage d'octets contiguë directement adressable par requête HTTP `Range`.

---

## D-011 — Détails de format MXFP4 vérifiés sur l'implémentation de référence
**2026-08-05 — vérifié**

Verrouillé contre `openai/gpt-oss` (`gpt_oss/torch/weights.py`) :

- table E2M1 : `[+0, +0.5, +1, +1.5, +2, +3, +4, +6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]` ;
- **nibble bas → index pair, nibble haut → index impair** dans chaque `uint8` ;
- échelle E8M0 : `valeur = fp4 * 2^(octet_échelle - 127)`, appliquée par `ldexp` ;
- bloc de **32 valeurs** sur la dernière dimension : 16 octets packés + 1 octet d'échelle.

Se tromper sur l'ordre des demi-octets produit un modèle qui génère du texte plausible mais
dégradé, sans erreur — d'où la vérification en amont plutôt qu'au débogage.
