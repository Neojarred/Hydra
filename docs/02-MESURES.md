# Hydra — Journal des mesures

Une entrée par expérience : ce qui a été mesuré, comment, ce qu'on en retient. Les
résultats négatifs y figurent au même titre que les positifs — ce sont eux qui coûtent le
plus cher à redécouvrir.

Machine : MacBook Apple M4, 10 cœurs GPU, 24 Gio, macOS 26.5.2, GPU famille **apple9**.
Modèle : GPT-OSS 20B installé au format `.hydra` (12,82 Gio).
Reproduction : `hydra bench 20b`, `hydra probe 20b`.

---

## M-001 — Le repacker tient l'invariant mémoire
**Jalon 1.1 — ✔**

| Grandeur | Mesure |
| --- | ---: |
| Installé | 12,82 Gio en 778 s |
| Débit crête | 44 Mo/s |
| **Empreinte du processus** | **50,9 Mio** |
| Plus gros bloc réseau reçu | 3,6 Mio |
| Part de la taille du checkpoint | **0,39 %** |

L'empreinte est restée à 50,9 Mio de 0 % à 100 % : elle ne dépend pas du volume
transféré, ce qui est exactement la propriété que le projet doit démontrer.

Vérification : 200 fenêtres tirées au hasard dans les fichiers installés, redemandées à
Hugging Face, comparées octet par octet — toutes concordent (6,78 Mo).

---

## M-002 — Les grandes requêtes battent le découpage, d'un facteur 6,4
**Résultat qui a invalidé un choix de conception**

| Motif | Débit |
| --- | ---: |
| une requête `Range` de 64 Mio | 33,5 Mo/s |
| huit requêtes `Range` de 4 Mio en série | 5,2 Mo/s |

Cause : Hugging Face répond un 302 vers un CDN signé ; chaque requête repaie redirection
et poignée TLS. L'URL résolue n'est pas réutilisable, sa policy contient une condition
`ByteRange` liée à la plage demandée.

**Retenu :** une requête par région contiguë, réponse consommée au fil de l'eau. La borne
mémoire devient une propriété mesurée plutôt que garantie par construction — compromis
documenté en D-013.

---

## M-003 — Le mappage sans copie est effectivement gratuit

| Étape | Empreinte du processus |
| --- | ---: |
| avant mappage | 557,9 Mio |
| après mappage de 3,35 Gio | 558,7 Mio |

`mmap` + `makeBuffer(bytesNoCopy:)` sur 2,27 Gio de `resident.bin` et 1,08 Gio de
`embed.bin` coûtent **0,8 Mio**. Les pages ne deviennent résidentes qu'à l'accès.

---

## M-004 — Les lectures d'experts en parallèle valent un facteur 1,8 à 2,0

Quatre experts (52,9 Mo), cache neuf à chaque tour, couche jamais lue, médiane sur 5 tours.

| Configuration | Série | Parallèle | Gain |
| --- | ---: | ---: | ---: |
| `F_NOCACHE` | 7,3 ms — 7,3 Go/s | **4,1 ms — 12,8 Go/s** | **×1,76** |
| cache de pages autorisé | 7,6 ms — 7,0 Go/s | 4,0 ms — 13,3 Go/s | ×1,91 |

**Retenu :** les lectures manquantes d'une couche sont émises en parallèle. C'est le seul
levier d'I/O qui ait payé jusqu'ici.

**Réserve d'honnêteté sur le chiffre absolu.** 13 Go/s dépasse ce qu'on attend d'un SSD
Apple en lecture. `F_NOCACHE` contourne le cache de pages du système mais pas celui du
contrôleur, et ces couches avaient été lues plus tôt dans la session. **Le rapport 1,8-2,0
est solide, la bande passante absolue ne l'est pas.** Une mesure vraiment froide demande
`sudo purge`, à faire avant de figer un modèle de débit.

Pour mémoire, un banc dédié en C sur un fichier de 24 Gio jamais lu, à décalages
aléatoires, donnait 3,0 Go/s à un thread et 5,3-5,7 Go/s à partir de quatre.

---

## M-005 — Le noyau GEMV atteint 52 % de la bande passante mémoire
**Après correction d'une erreur de mesure qui l'avait fait paraître cinq fois plus lent**

`gate_up` d'un expert réel, [5760 × 2880] MXFP4, médiane sur 7 tours de 50 passes.

| Variante | ms | Go/s | écart / référence CPU |
| --- | ---: | ---: | ---: |
| `mxfp4_gemv` (référence) | 0,20 | 44,6 | 1,15e-07 |
| **`mxfp4_gemv_vectorized`** | **0,19** | **46,5** | 1,14e-07 |
| `mxfp4_gemv_simd` | 0,23 | 38,3 | 1,00e-07 |
| `mxfp4_gemv_tiled` | 0,24 | 37,1 | 1,00e-07 |

Bande passante mémoire de la machine : 89-98 Go/s selon la mesure. Le meilleur noyau en
exploite **52 %**.

Les quatre variantes sont numériquement correctes : écart de l'ordre de 1e-7 contre le
décodeur CPU validé bit à bit, sommé en double précision.

---

## M-006 — L'erreur de mesure : 45 µs de synchronisation par tampon de commandes
**Le résultat négatif le plus instructif de la série**

Une première campagne donnait 0,95 ms pour **toutes** les variantes, indistinctement. Un
plateau parfaitement plat entre quatre noyaux très différents n'est pas un résultat, c'est
un symptôme.

Cause : chaque mesure encodait **une** passe dans un tampon de commandes, le validait et
attendait. L'aller-retour CPU-GPU à vide coûte **45 µs**, et le temps mesuré était dominé
par la synchronisation, pas par le noyau.

En encodant 50 passes par tampon, le temps réel apparaît : **0,19 ms**, soit cinq fois
moins. Les écarts entre variantes redeviennent visibles.

**Conséquence de conception, pas seulement de mesure.** Un token du 20B demande
4 experts × 24 couches × 2 GEMV = 192 dispatches. À 45 µs de synchronisation par tampon,
les émettre séparément coûterait 8,6 ms de pure latence. Le graphe d'exécution doit donc
encoder **une couche entière, voire un token entier, par tampon de commandes** — ce qui
justifie après coup le découpage `cb1` / `io` / `cb2` de TurboFieldfare.

---

## M-007 — Deux optimisations plausibles qui ne paient pas
**Résultats négatifs**

**Tuilage des activations en mémoire partagée.** Hypothèse : avec un threadgroup par
ligne, le vecteur d'activations est relu 5 760 fois, soit 66 Mo de trafic contre 8,3 Mo de
poids. Mettre `x` en mémoire partagée devait diviser ce trafic.

Mesure : **0,24 ms contre 0,19**, soit 21 % plus lent. L'hypothèse était fausse — les
11,5 Kio d'activations tiennent largement dans le cache du GPU, qui les sert déjà sans
passer par la mémoire principale. La copie explicite et la barrière de threadgroup
coûtent plus que ce qu'elles économisent.

**Un seul groupe SIMD par ligne.** Hypothèse : supprimer la mémoire partagée et la
barrière de la réduction en deux temps. Mesure : **0,23 ms contre 0,19**, 18 % plus lent.
Avec 32 voies pour 90 blocs, chaque voie traite trois blocs et le parallélisme manque.

**Retenu :** aucune des deux ne devient le défaut. La vectorisation des chargements
(`uint4` au lieu de 16 `uchar`) gagne ×1,04 — un écart faible mais reproductible et sans
changement de sortie, donc conservé.

---

## M-008 — Extrapolation du MoE seul

Sur la base du meilleur noyau et de l'architecture du 20B :

```
0,19 ms × 1,5 (gate_up puis down) × 4 experts × 24 couches ≈ 27 ms/token
```

Soit **~37 tok/s pour le MoE seul**, hors attention, tête LM et I/O. C'est une **borne
haute optimiste** : elle ignore le reste du graphe et suppose zéro miss de cache.

À rapprocher du plancher théorique calculé en phase 0, 39,4 ms/token pour le 20B tous
postes confondus. Les deux sont cohérents.

---

## Ce que ces mesures ont changé dans le code

1. Le repacker télécharge par régions contiguës — ×8,5.
2. Les lectures d'experts d'une couche partent en parallèle — ×1,8.
3. Le GEMV vectorisé devient le défaut — ×1,04.
4. Le tuilage et la variante SIMD sont écartés, et documentés comme tels.
5. Le futur graphe d'exécution devra grouper les dispatches par couche au minimum.

## Méthode

- Contrôle et candidat **alternent** au sein d'une même campagne.
- On rapporte la **médiane**, pas la moyenne ni le meilleur temps.
- Tout candidat est comparé numériquement à la référence CPU **avant** d'être jugé sur le
  temps ; un gain qui change les sorties n'est pas un gain.
- Un écart dans le bruit laisse le défaut inchangé.

---

## M-009 — La couche complète concorde avec la référence dès le premier assemblage
**Jalon 1.3 étendu — ✔**

Test d'intégration sur un modèle miniature installé par le **vrai repacker** : 4 couches,
6 experts, top-2, `hidden` 64, GQA groupe 2, fenêtre glissante de 8. Douze tokens décodés
d'affilée, cache KV et fenêtre en jeu.

Écart relatif maximal contre `ReferenceLayer` : **< 2e-3** à chaque position.

Ce que ce test couvre et qu'aucun test unitaire ne couvrait :

- l'**ordre** des opérations dans la couche ;
- les **décalages** de chaque tenseur dans `resident.bin` ;
- la disposition du **cache KV** et l'indexation circulaire de la fenêtre glissante ;
- le câblage du **routeur** — les identifiants produits par le GPU pilotent bien le
  chargement SSD puis le calcul ;
- l'**entrelacement** de `gate_up` consommé par le SwiGLU, et le découpage en deux
  moitiés du RoPE, dans le même graphe.

Il est passé sans correction, ce qui n'était pas acquis : c'est le premier point où le
repacker, le format, le mappage sans copie, le cache d'experts et sept noyaux Metal se
rencontrent sur les mêmes octets.

---

## M-010 — La frontière du routeur impose deux tampons de commandes par couche

Le routeur produit les identifiants d'experts **sur le GPU**, et le CPU doit les lire pour
savoir quels blobs charger depuis le SSD. Aucun réordonnancement ne contourne cette
dépendance : elle coupe la couche en deux.

```
cb1 : norme → QKV → RoPE → écriture KV → attention → projection O → résidu
      → norme post-attention → logits du routeur → top-k
I/O : lecture des identifiants, chargement parallèle des experts manquants
cb2 : par expert — gate_up → SwiGLU → down → accumulation pondérée → résidu
```

À 45 µs par aller-retour (M-006), cela coûte **2,2 ms par token sur le 20B** — 24 couches
× 2 tampons — avant même le moindre calcul. C'est le prix structurel de l'absence
d'expert partagé : chez TurboFieldfare, la branche dense s'exécute pendant les lectures ;
ici il n'y a rien à faire pendant ce temps.

---

## M-011 — Campagne prefill : 18,0 s → 5,6 s, et quatre hypothèses fausses

Invite de 78 jetons, GPT-OSS 20B, 4 slots d'experts par couche.

| Étape | Prefill |
| --- | ---: |
| Jeton par jeton (départ) | 18,0 s |
| Par blocs, GEMM naïf | 6,5 s |
| + sélection du noyau selon le nombre de jetons | 4,4 s |
| **État final** | **5,6 s — 14 jetons/s** |

**Ce que j'avais annoncé : 10 à 30×. Livré : 3,2×.** L'estimation ne comptait que le
trafic sur les **poids** (92,9 Gio → 1,2 Gio, ce facteur 78 est réel) en ignorant le
trafic sur les **activations**, qui s'est révélé être le terme dominant.

### Les hypothèses testées et écartées

1. **Les barrières de threadgroup écrasent la réduction.** Passer de 32 barrières par
   tuile à une seule : *aucun changement*.
2. **Les conflits de bancs en mémoire partagée.** Décalage d'un flottant par ligne pour
   rendre le pas premier avec 32 bancs : *aucun changement*.
3. **Le tableau `float weights[32]` déborde en registres.** Remplacé par huit `float4` :
   *aucun changement*.
4. **Grouper quatre lignes par threadgroup amortit les activations.** *+10 %.*

Quatre corrections plausibles, une seule marginalement utile. Le point commun : toutes
partaient d'une intuition sur le noyau, aucune d'une mesure du noyau.

### Ce qui a réellement payé

**Un banc isolé, passe par passe.** Il aurait dû venir en premier. Il a montré que les
noyaux d'une couche totalisent **10,9 ms** — soit 0,26 s pour 24 couches — alors que la
mesure de bout en bout en attribuait 2,09 s à la même portion. Les 1,8 s d'écart
n'étaient pas du calcul.

**Le préchargement des pages.** L'écart venait des défauts de page : la première passe
faisait entrer les 2,27 Gio de `resident.bin` une page à la fois. Une lecture séquentielle
au chargement fait passer `cb1` de **2,09 s à 0,88 s**. Le coût est déplacé au chargement,
donc neutre pour une invite unique et gagnant sur une session entière.

**Le choix du noyau selon le nombre de jetons.** Le GEMM tuilé (`TILE_ROWS = 128`) ne
lance que `rows/128` threadgroups — 45 pour `gate_up`, très en deçà de ce qu'il faut pour
occuper dix cœurs. Or en prefill un expert ne sert qu'environ huit jetons sur soixante-dix.
Sous 32 jetons, la variante à une ligne par groupe SIMD lance trente fois plus de
threadgroups et l'emporte. **Le bon noyau dépend du nombre de jetons, pas du type
d'opération.**

### Le GEMM tuilé, isolé

`q_proj` [4096 × 2880] BF16, médiane sur 5 tours de 20 passes :

| Jetons | GEMV répété | GEMM tuilé | Gain |
| ---: | ---: | ---: | ---: |
| 1 | **0,24 ms — 99 Go/s** | 1,21 ms | ×0,20 |
| 16 | 3,63 ms | 1,67 ms | ×2,18 |
| 68 | 15,66 ms | 3,57 ms | ×4,47 |
| 128 | 31,19 ms | 3,80 ms | **×8,22** |

Le GEMV atteint 99 Go/s sur un jeton, soit la bande passante de la machine : il est
optimal, et il reste le bon choix en décodage. Le modèle de trafic qui gouverne le GEMM
tuilé est `cols × rows × tokens × (2/TILE_TOKENS + 4/TILE_ROWS)`.

---

## M-012 — Le recouvrement I/O ne paie pas
**Résultat négatif, reproductible, conforme à TurboFieldfare**

Lectures d'experts lancées en tâche de fond, calcul de chaque expert soumis dès qu'il
arrive. Mesures alternées :

| Tour | Avec recouvrement | Sans |
| --- | ---: | ---: |
| 1 | 5,21 tok/s | **7,27 tok/s** |
| 2 | 5,26 tok/s | **7,28 tok/s** |

**Le recouvrement coûte 28 %.** La cause est identifiée : soumettre un tampon de commandes
par expert au lieu d'un par couche ajoute quatre synchronisations à 45 µs, et surtout
sérialise le GPU sur des tampons plus courts. Le gain d'I/O recherché est plus que
compensé.

Impossible de faire mieux sans lever la contrainte d'ordre : les quatre experts partagent
le scratch et accumulent tous dans le même tampon de mélange, donc leurs tampons de
commandes ne peuvent pas se recouvrir sans se corrompre.

TurboFieldfare avait mesuré exactement le même effet — 4,799 → 4,648 tok/s. Le code est
conservé derrière `overlapExpertIO`, **désactivé par défaut**, pour que la mesure reste
reproductible.


## M-013 — Le goulot du décodage n'est pas l'I/O

Mesure sur GPT-OSS 20B, 32 jetons, M4 24 Gio. Quatre configurations de cache :

| Slots/couche | Débit | Taux de hit | Mémoire du cache | Part de l'I/O |
|---|---|---|---|---|
| 4 (minimum) | 4,58 tok/s | 86,2 % | 1,18 Gio | 19 % |
| 8 | 6,50 tok/s | 93,4 % | 2,37 Gio | 9 % |
| 16 | 5,29 tok/s | 95,2 % | 4,73 Gio | 14 % |
| 32 (tout le pool) | 4,66 tok/s | 95,5 % | 9,47 Gio | 13 % |

Deux conclusions contre-intuitives.

**Le taux de hit était déjà bon au minimum.** 86 % avec quatre slots pour quatre experts
actifs : le routage a assez de localité temporelle pour que le cache travaille même à sa
taille plancher. La littérature sur le déchargement d'experts (HOBBIT, arXiv 2411.01433)
part d'un régime où le chargement pèse 94,5 % du temps ; ici il n'en pèse que 9 à 19 %.
Les techniques de préchargement prédictif, de graphe de transitions ou de précision mixte
optimiseraient donc une fraction marginale du temps.

**Plus de mémoire ne rachète pas de la vitesse.** Trente-deux slots — le pool entier
résident, huit fois l'empreinte — sont *plus lents* que huit. Le gain de hit (93,4 → 95,5 %)
ne compense pas la pression mémoire. C'est un argument direct pour la thèse du projet :
l'empreinte minimale n'est pas seulement suffisante, elle est proche de l'optimum.

Huit slots par couche sont retenus par défaut : meilleurs sur les deux axes à la fois.

## M-014 — Fusion des tampons de commandes : ×1,45

Le décodage faisait sept allers-retours CPU-GPU par couche : attention, début de mélange,
un par expert sélectionné, fin. Sur vingt-quatre couches, **168 attentes par jeton**.

Le seul point de synchronisation réellement nécessaire est la lecture du routeur — le CPU
doit connaître les experts choisis avant de lire leurs poids. Tout le reste tient dans le
même tampon, y compris l'attention de la couche *suivante*, encodée derrière le mélange de
la couche courante. Les encodeurs d'un même tampon s'exécutant dans l'ordre de création,
la dépendance sur le résidu est respectée. Il reste **une attente par couche**.

Le recouvrement des lectures avec le calcul a été retiré dans le même mouvement : il
imposait une attente par expert pour masquer une I/O qui ne pèse que 9 % du temps.

Mesure appariée, 48 jetons, 8 slots, médiane de trois exécutions :

| | avant | après |
|---|---|---|
| débit | 6,36 tok/s | **9,22 tok/s** |
| ms/jeton | 132 | 84 |
| attention + routeur | 44,5 ms (33 %) | 1,2 ms (1 %) |

Les 43 ms disparus de l'attention étaient de l'attente pure.

## M-015 — Deux réécritures de noyaux sans effet

Hypothèse : les GEMV paient une réduction trop coûteuse pour le travail fourni. Le noyau
MXFP4 donne une ligne à 96 threads pour 90 blocs — un bloc par thread, puis `simd_sum`,
mémoire partagée, barrière et somme sérielle. Le noyau BF16 est pire encore : 256 threads
pour 360 groupes, soit 16 à 32 octets par thread, et une réduction sur huit groupes SIMD.

Deux variantes écrites, donnant une ligne entière à chaque groupe SIMD : réduction limitée
à `simd_sum`, sans barrière ni mémoire partagée, et trois à onze fois plus de travail par
thread.

| noyau | référence | variante SIMD par ligne |
|---|---|---|
| MXFP4 | 47,2 Go/s | 47,2 Go/s |
| BF16 (bout en bout) | 9,22 tok/s | 8,82 tok/s |

**Aucun gain, l'un des deux légèrement négatif.** Les deux noyaux tournent déjà à ~50 % de
la bande passante mémoire de la machine, et c'est elle la limite — pas la structure de la
réduction. Les deux variantes ont été retirées.

Le plancher théorique est atteignable par le calcul : 3,7 Gio lus par jeton — 1,27 pour
l'attention, 1,27 pour les experts, 1,16 pour la tête LM — soit 39 ms à 94 Go/s, ou
25,6 jetons/s. Descendre sous ce plancher demanderait de lire moins d'octets, c'est-à-dire
de quantifier les poids denses. C'est exclu (D-015).

## M-016 — L'échantillonnage top-p coûtait plus cher que la tête LM

Un écart d'un facteur deux séparait le banc d'essai (9,2 jetons/s) de l'application
(4 à 5). Trois causes, dont deux hors du moteur.

**Le débit affiché comptait le prefill.** Il partait du début du traitement de l'invite.
Sur une conversation établie, l'invite fait plusieurs milliers de jetons et son traitement
pèse plus que toute la réponse : le même moteur affichait 6 jetons/s sur une conversation
neuve et 4 sur une conversation chargée, en décodant exactement à la même vitesse. Le coût
du prefill reste visible — c'est le temps avant réponse, affiché juste à côté.

**L'interface rendait la conversation à chaque jeton.** `MarkdownView` reparse la totalité
du message à chaque rendu : sur un message qui grandit, le coût est quadratique en sa
longueur. Ce travail s'exécute sur le fil principal mais consomme la bande passante
mémoire, celle-là même qui limite le décodage. Les fragments sont désormais regroupés à
20 Hz, et la mise en forme n'est calculée qu'une fois la réponse terminée.

**L'échantillonneur triait le vocabulaire entier.** Le banc d'essai décode en glouton ;
l'application échantillonne avec `top_p = 0,9`, et cette branche construisait deux tableaux
de 201 088 entrées par jeton — 2,4 Mio à allouer, remplir et jeter — puis les triait
intégralement pour n'en garder qu'une trentaine.

Remplacé par un tas-min de 64 entrées : une passe sur le vocabulaire, aucune allocation
proportionnelle à sa taille, et le paquet n'est élargi que si la masse visée n'est pas
atteinte. La recherche du maximum est fusionnée dans la même passe.

| | avant | après |
|---|---|---|
| moteur de l'application, 8 slots | 4,73 / 4,87 / 4,61 | 6,08 / 5,93 / 7,51 / 7,91 / 6,15 |

Trois tests vérifient que le tas rend exactement les mêmes candidats qu'un tri complet, sur
des vocabulaires jusqu'à 201 088 entrées, y compris avec des valeurs égales — une sélection
qui décalerait le noyau d'un seul rang changerait le comportement du modèle sans rien
signaler.

**Réserve sur ces chiffres.** La variance d'une exécution à l'autre atteint ±30 % à
configuration identique. Les médianes vont dans le même sens et les allocations supprimées
sont un fait indépendant de la mesure, mais l'ampleur exacte du gain n'est pas établie à
mieux que cet ordre de grandeur.

## M-017 — Pourquoi le cache KV n'est pas réutilisé entre tours

Le prefill est repayé intégralement à chaque tour, alors que l'invite d'un tour est un
préfixe de celle du suivant. Réutiliser le travail déjà fait supposerait de rembobiner le
cache KV jusqu'au point de divergence.

Ce n'est pas possible dans l'état actuel : les couches à fenêtre glissante gardent leurs
clés dans un anneau de 128 positions. Une réponse plus longue que 128 jetons a déjà écrasé
les positions qu'il faudrait retrouver. Le rembobinage n'est sûr que sur moins de 128
jetons — c'est-à-dire presque jamais.

Le rendre possible demande de donner à ces couches un cache de pleine longueur, donc plus
de mémoire. C'est un arbitrage à poser explicitement au regard de la thèse du projet, pas
un raccourci à prendre au passage.

## M-018 — L'avance des huit slots était celle des surcoûts qu'ils masquaient

M-013 mesurait 4,58 jetons/s à quatre slots contre 6,50 à huit — 42 % d'écart — et le
défaut avait été porté à huit sur cette base.

Après correction de la synchronisation GPU (M-014) et de l'échantillonnage (M-016),
mesure appariée dans l'application, **même conversation, même contexte** :

| slots/couche | débit | cache d'experts |
|---|---|---|
| 4 (minimum) | 7,7 tok/s | 1,18 Gio |
| 8 | 8,5 tok/s | 2,37 Gio |

L'écart tombe de 42 % à 10 %. Les 42 % n'étaient pas ceux du cache : les surcoûts fixes
dominaient tellement le décodage qu'ils amplifiaient toute différence de temps d'I/O.
Une fois retirés, le cache minimal se révèle presque aussi rapide.

Trois exécutions du banc en ligne de commande à chaque configuration donnent 5,61–7,68
contre 5,53–7,33 : les distributions se recouvrent, le banc ne sait pas séparer les deux.
La mesure retenue est celle faite dans l'application sur une conversation unique, qui est
appariée là où le banc ne l'est pas.

**Le défaut repasse au minimum.** Dix pour cent ne valent pas 1,19 Gio dans une application
dont l'objet est de montrer ce qu'il suffit d'avoir en mémoire. Le réglage reste exposé
pour qui préfère l'arbitrage inverse.

Résultat d'ensemble sur cette session : **4 à 5 jetons/s → plus de 7**, et le débit reste
au-dessus de 7 à mesure que le contexte se remplit.

## M-019 — Le prefill, lui, est bien limité par l'I/O

Le temps avant réponse s'effondre avec la longueur du contexte : 3 s sur une conversation
neuve, **30 s au-delà de mille jetons**. La cause est l'inverse exacte de celle du décodage.

Un bloc de prefill sollicite quasiment tous les experts d'une couche — 128 jetons routés
vers 4 experts chacun couvrent presque les 32 — alors que le cache n'en tient que 4. Chaque
bloc relit donc le pool entier :

```
32 experts × 13,2 Mo × 24 couches  =  10,1 Go par bloc de 128 jetons
1000 jetons = 8 blocs              =  81 Go
à 4,3 Go/s                         ≈  19 s
```

C'est bien l'ordre de grandeur observé. **Ici, et seulement ici, les techniques de
streaming d'experts de la littérature s'appliquent** : le décodage a un taux de hit de
86 %, le prefill n'en a aucun.

Le coût par jeton étant inversement proportionnel à la taille du bloc, celle-ci est le
levier direct. L'anneau des couches glissantes se dimensionnant déjà sur
`slidingWindow + prefillChunk`, l'augmenter reste correct par construction — et les deux
tests d'équivalence du prefill par blocs le vérifient.

Invite de mille jetons, temps total jusqu'à la fin de la réponse, deux exécutions :

| bloc | temps total | empreinte |
|---|---|---|
| 128 | 39,2 / 43,6 s | 2683 Mio |
| **512** | **28,5 / 29,1 s** | 2727 Mio |
| 1024 | 25,1 / 27,7 s | 2802 Mio |

512 est retenu : 44 Mio pour un tiers du temps total, là où 1024 demande 119 Mio pour
gagner 12 % de plus. Sur des invites plus longues que mille jetons l'arbitrage se
déplacerait en faveur de 1024.

## M-020 — Réutilisation du cache KV entre tours : temps avant réponse ÷ 6

M-019 s'était trompé de diagnostic : le prefill n'est pas limité par l'I/O mais par le
calcul. Sur mille jetons, mesuré via `HYDRA_PROFILE` :

```
prefill 1031 jetons :  I/O 3,8 s · calcul 8,6 s · attention 3,8 s
```

L'I/O ne pèse que 23 %. Le pool d'experts fait 9,47 Gio sur une machine de 24 : le cache de
pages du système en absorbe l'essentiel après la première passe. Le gain des blocs de 512
était donc réel, mais parce qu'il amortit mieux les GEMM, pas parce qu'il lit moins.

Conséquence : accélérer ce calcul est difficile — deux tentatives sur les noyaux ont déjà
échoué (M-015). Mais ce calcul est aussi **entièrement redondant** : chaque tour re-rend la
conversation entière et recalcule une invite dont trois jetons sont nouveaux.

### Ce qui bloquait

Rembobiner le cache jusqu'au point de divergence. Impossible avec un anneau : au-delà de sa
capacité il a écrasé ce qu'il faudrait retrouver.

Les couches à fenêtre glissante reçoivent donc un stockage **linéaire** jusqu'à 8192 de
contexte. Le point délicat est que `ringSize` gouvernait à la fois le stockage *et* la
portée de l'attention : un stockage linéaire aurait rendu ces couches pleines, changeant le
modèle sans rien signaler. Les deux notions sont désormais distinctes — `ringSize` pour
l'indexation, `windowed` pour la portée.

### Mesure

Invite de 1031 jetons, puis une question de suite dans la même conversation :

| | avant réponse |
|---|---|
| premier tour | 18,8 s |
| **tour de suite** | **3,1 s** |

Empreinte : 2727 → 2808 Mio, soit **81 Mio mesurés**.

Le tour de suite n'est pas gratuit parce que Harmony ne remet pas le canal d'analyse dans
l'historique : le préfixe commun s'arrête à la fin de l'invite précédente, et la réponse
puis la nouvelle question sont recalculées. C'est la grosse invite qui est épargnée.

Au-delà de 8192 de contexte, l'anneau reprend la main et le comportement antérieur aussi.

### Correction

Trois tests couvrent ce qui échouerait en silence : le stockage linéaire doit fenêtrer
exactement comme l'anneau ; rembobiner puis reprendre doit égaler un calcul complet ; une
invite modifiée doit repartir du point de divergence et non réutiliser des clés périmées.

## M-021 — Le 120B est dans un régime inverse du 20B

Le 120B tourne : 6,28 Gio en mémoire — 2,32 engagés, 3,96 mappés — pour un modèle de
60,77 Gio installé, soit **10 %**. Sans rien quantifier au-delà du MXFP4 publié.

Répartition d'un jeton, 24 jetons, 4 slots par couche :

```
I/O  lecture des experts   155,9 ms   47 %
cb2  mélange d'experts     158,5 ms   48 %
tête LM                     16,0 ms    5 %
cb1  attention + routeur     1,7 ms    1 %
```

**L'I/O pèse 47 %, contre 9 % pour le 20B.** Le taux de hit tombe à 76 % : quatre slots
pour 128 experts au lieu de 32. C'est le régime que décrit la littérature sur le
déchargement d'experts — et donc le seul endroit du projet où ses techniques
s'appliqueraient réellement. La conclusion de M-013 vaut pour le 20B, pas pour le 120B.

### Agrandir le cache dégrade le débit

| slots/couche | débit | taux de hit | cache | temps d'I/O |
|---|---|---|---|---|
| **4** | **2,15 tok/s** | 76,0 % | 1,78 Gio | 155,9 ms |
| 8 | 1,99 tok/s | 82,8 % | 3,55 Gio | 157,2 ms |
| 16 | 1,74 tok/s | 86,1 % | 7,10 Gio | 176,0 ms |

Le taux de hit s'améliore franchement — 76 → 86 % — et **le temps d'I/O n'en profite pas,
il augmente**. Le cache du processus prend la place du cache de pages du système, qui
servait les défauts à moindre coût. On paie deux fois : la mémoire, et la perte de ce qui
la rendait inutile.

C'est le résultat le plus net du projet en faveur de sa propre thèse : sur une machine
contrainte, **agrandir le cache résident est contre-productif**, pas seulement inutile.

### Marge restante

Le plancher de bande passante est de ~4,9 Gio lus par jeton, soit 52 ms à 94 Go/s, ou
19 tok/s. À 311 ms on en exploite 17 %. Deux gisements séparés, d'environ 155 ms chacun :
l'I/O, qu'un préchargement idéal supprimerait, et le calcul, à ~31 Go/s effectifs contre
47 mesurés sur le noyau seul.

## M-022 — Le recouvrement lecture/calcul ne peut rien recouvrir

Le 120B passe 46 % de son temps en lecture d'experts. Recouvrir ces lectures avec le calcul
paraissait donc valoir près d'un facteur deux : par couche, 4,3 ms d'I/O contre 4,4 ms de
calcul, deux grandeurs idéalement appariées.

Mesure appariée, 120B, 4 slots, 24 jetons :

| | sans recouvrement | avec |
|---|---|---|
| ms/jeton | 311 | 309 |
| I/O | 152,8 ms | 0,0 ms *(absorbée)* |
| mélange | 160,1 ms | 310,6 ms |

**Aucun gain.** Le temps a changé de compteur, pas de valeur.

La cause est dans le cache : `load(layer:experts:)` lit déjà les `top_k` experts **en
parallèle** via `concurrentPerform`. Les quatre arrivent donc ensemble, et l'expert 0 n'est
pas disponible avant le 3. Il n'existe aucune disponibilité échelonnée à exploiter. Les
étaler pour en créer une reviendrait à sérialiser les lectures — 3,0 Go/s au lieu de 5,7 —
et coûterait plus que le recouvrement ne rapporte.

Recouvrir la couche `L+1` pendant le calcul de `L` supposerait de connaître son routage
avant que `L` ne soit calculée. C'est circulaire : l'entrée du routeur de `L+1` est la
sortie de `L`. Seule une *prédiction* le permettrait (HOBBIT), avec le risque de charger
des experts inutiles.

## M-023 — Où en est la marge sur cette machine

État du 120B après toutes les corrections : 314 ms/jeton, 4 slots, 6,28 Gio en mémoire.

**La moitié I/O — 150 ms — est au plafond matériel.** Le taux de hit de 76 % laisse en
moyenne 0,96 défaut par couche : la plupart des lectures sont donc **isolées**, servies à
3,0 Go/s et non aux 5,7 Go/s du régime parallèle. Les 436 Mo lus par jeton à 2,9 Go/s
effectifs correspondent exactement à ce régime. Grouper ces défauts demanderait de connaître
plusieurs couches à l'avance, ce que la dépendance séquentielle du routage interdit.

**La moitié calcul — 159 ms — garde de la marge, mais moins que le plancher théorique ne le
suggère.** Le GEMV MXFP4 atteint 47 Go/s au banc, mais celui-ci relit cinquante fois le même
expert de 8,8 Mo, qui tient largement dans le cache système : le chiffre est optimiste. En
production chaque expert est lu une fois, à froid, et le débit effectif est de 11,5 Go/s.
Le vrai plafond est probablement vers 20-25 Go/s, soit un plancher de calcul autour de
80 ms plutôt que les 19 ms de la borne de bande passante pure.

Marge totale réaliste sur cette machine : **314 → ~235 ms**, soit ×1,3. Pas ×4.

Descendre plus bas demande de lire moins d'octets — donc de quantifier les poids denses,
exclu (D-015) — ou une machine dont la bande passante mémoire et le débit disque ne sont
pas ceux d'un M4.

## M-024 — Calculer d'abord ce qui est déjà là

Le recouvrement de M-022 avait échoué pour une raison que la mesure du taux de hit rendait
pourtant visible : **avec 76 % de hit, il ne manque en moyenne qu'un expert sur quatre.**
Trois sont déjà en mémoire et n'attendent que le GPU — mais `load(layer:experts:)` bloquait
sur les quatre avant de lancer le moindre calcul.

Le décodage calcule désormais d'abord les experts résidents, pendant que les manquants se
lisent.

### Ce que cela imposait à la structure

Réordonner le calcul changerait l'ordre de la somme flottante, donc les sorties — et cet
ordre dépendrait de l'état du cache, rendant le modèle non déterministe. Chaque expert
écrit donc dans **sa propre case**, et la somme se fait ensuite dans l'ordre fixe des
slots. L'ordre de calcul devient libre, l'ordre d'addition reste figé.

### Un piège coûteux

Première version : lancer les lectures, puis encoder les experts résidents. Le taux de hit
est tombé de 76 à 63,6 %. L'encodage est ce qui **épingle** un slot ; tant qu'il n'a pas eu
lieu, les lectures lancées en arrière-plan choisissent comme victimes les slots libres,
c'est-à-dire précisément ceux qu'on s'apprêtait à utiliser. L'encodage doit précéder le
lancement des lectures.

### Mesure

120B, 4 slots, 24 jetons :

| | ms/jeton | I/O | mélange | hit |
|---|---|---|---|---|
| avant | 311–314 | 150 ms (46 %) | 159 ms (48 %) | 76,0 % |
| lectures avant épinglage | 301–325 | 96 ms | 220 ms | 63,6 % |
| **après** | **284–290** | 105–116 ms | 183–192 ms | 69,4 % |

**Gain net : 9 %.** Le 20B ne régresse pas (médiane 7,68 tok/s à 4 slots).

Loin des 35 % que la seule arithmétique laissait espérer — cacher 3,3 ms de calcul derrière
4,2 ms de lecture aurait dû faire mieux. L'écart tient probablement à la mémoire unifiée :
la lecture aboutit à une copie en RAM, qui consomme la bande passante dont le GPU a
justement besoin. Deux opérations limitées par la même ressource ne se recouvrent qu'en
partie.

## M-025 — Décodage spéculatif : attaquer l'intensité arithmétique

Les corrections précédentes cachaient ou réorganisaient le travail. Celle-ci en supprime :
une passe ordinaire relit tous les poids pour produire **un** jeton ; une passe groupée les
relit une fois pour en vérifier `n`. Sur le 120B, les poids denses — attention, routeurs,
tête LM — font 2,88 Gio relus à chaque jeton ; sur un lot de quatre, c'est une lecture pour
quatre.

### Exactitude

La sortie est identique jeton pour jeton, à graine égale, que le brouillon soit juste ou
faux. Deux propriétés le garantissent :

- **Chaque jeton émis consomme exactement un tirage**, comme sans spéculation : la suite
  pseudo-aléatoire est donc la même. C'est le point qui casse en premier si l'on code
  l'algorithme naïvement.
- **Les logits de la position `P+i` ne sont utilisés que si les jetons `P..P+i-1` ont été
  acceptés** — c'est-à-dire si l'hypothèse sous laquelle ils ont été calculés était vraie.

Le premier jeton est tiré *avant* la passe groupée : si le brouillon se trompe d'emblée, on
retombe sur un pas ordinaire sans avoir rien dépensé.

Quatre tests couvrent l'équivalence sur les deux modes de tirage, avec brouillons justes,
partiellement faux et absurdes. Un vérificateur faux ne planterait pas — il produirait du
texte plausible et faux.

### Source des brouillons

Recherche de motif dans ce qui a déjà été écrit : on cherche la dernière occurrence des
trois (puis deux) derniers jetons et on propose ce qui suivait. Coût nul, mémoire nulle.

Un second modèle ne convenait pas : le 20B n'est que 4,5 fois moins cher que le 120B, il en
faudrait dix.

### Mesure

Tâche de recopie sur le 20B — le cas favorable, où la réponse reprend l'invite :

| | jetons/s |
|---|---|
| sans spéculation | 5,79 / 6,21 / 6,13 |
| **avec** | **7,55 / 6,89 / 6,52** |

Médiane 6,13 → 6,89, soit **+12 %**. Les trois exécutions avec dépassent les trois sans,
ce qui est un signal net malgré la variance habituelle de ±30 %.

**Sur une invite sans reprise, le gain est nul** — les motifs ne se retrouvent pas, aucun
brouillon n'est proposé, et le décodage est exactement celui d'avant. Le gain dépend donc
de l'usage : élevé en résumé, réécriture, code, questions sur un document joint ; nul en
discussion ouverte.
