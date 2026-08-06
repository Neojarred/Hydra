# Hydra — Étude de faisabilité (Phase 0)

État : **à valider**. Aucun code de runtime écrit à ce stade.
Machine de référence : MacBook **Apple M4** (10 cœurs GPU), **24 Gio** de mémoire unifiée,
macOS **26.5.2**, Swift **6.3.2**, **126 Gio** libres sur le SSD interne.

Tous les chiffres ci-dessous sont **mesurés ou dérivés d'en-têtes safetensors réels**.
Là où une valeur reste une hypothèse, c'est écrit explicitement.
Le calculateur est reproductible : `python3 tools/budget.py`.

---

## Verdict en dix lignes

1. **GPT-OSS 120B tient sur 24 Gio.** Le budget boucle avec de la marge : 3,96 Gio de poids
   résidents, 1,13 Gio de KV à 32k, et il reste **12,2 Gio pour le cache d'experts** sous le
   plafond Metal par défaut. Le projet n'est pas bloqué par une contrainte physique.
2. **Mais il sera lent** : **3 à 5 tok/s** attendus. Le plafond absolu, cache parfait et I/O nulle,
   est de **18,8 tok/s** — imposé par la bande passante mémoire, pas par le SSD.
3. **GPT-OSS 20B *peut* tenir entièrement en mémoire (12,82 Gio sous un plafond de 17,76 Gio),
   mais ce n'est pas le mode visé** — voir D-012. Il tourne au minimum en **3,77 Gio**, soit 28 %
   de sa taille installée. Le mode entièrement résident sert de **référence de correction** :
   à décodage glouton, les deux doivent produire exactement les mêmes tokens.
4. **GPT-OSS n'a pas d'expert partagé.** Confirmé sur l'index safetensors. Le mécanisme qui masque
   la latence I/O chez TurboFieldfare n'existe pas ici.
5. Bonne nouvelle : ce mécanisme ne valait que **+7,5 %** chez TurboFieldfare (mesure publiée).
   Sa perte est réelle mais pas rédhibitoire.
6. **Le SSD est deux fois plus rapide que prévu** : 5,5 Go/s à froid sur le vrai motif d'accès.
7. **Le vrai plafond Metal de cette machine est 17,76 Gio**, pas 24. Mesuré, pas supposé.
8. **La fenêtre glissante de GPT-OSS fait 128 tokens** — 18 couches sur 36 ne gardent presque rien.
   Le KV cache est étonnamment bon marché : 4,51 Gio à 128k.
9. **Qwen3.6-35B-A3B est bien le bon candidat MoE**, mais 30 de ses 40 couches sont du
   **Gated DeltaNet** — une famille de noyaux entièrement nouvelle. C'est le poste le plus coûteux
   du projet, et il faut en discuter.
10. **Le stockage est la contrainte la plus serrée**, pas la mémoire : 126 Gio libres pour ~91 Gio
    de modèles. Le repack en streaming n'est pas une élégance, c'est une nécessité.

11. **Le plancher mémoire n'est pas le cache d'experts, ce sont les poids résidents BF16.**
    2,27 Gio pour le 20B, 2,88 Gio pour le 120B, dont 1,08 Gio pour la seule tête LM. GPT-OSS ne
    quantifie que ses experts ; TurboFieldfare atteignait ~2 Go au total parce que Gemma 4 avait
    ces tenseurs-là en 4 bits. C'est le principal levier restant, et il coûte une validation
    numérique.

---

## 1. La machine, mesurée

| Grandeur | Valeur mesurée | Comment |
| --- | ---: | --- |
| `MTLDevice.recommendedMaxWorkingSetSize` | **19 069 665 280 o (17,76 Gio)** | sonde Swift/Metal |
| `MTLDevice.maxBufferLength` | 14 302 248 960 o (13,32 Gio) | sonde Swift/Metal |
| Mémoire physique | 25 769 803 776 o (24,00 Gio) | `ProcessInfo` |
| Famille GPU | **apple9** (pas apple10) | `supportsFamily` |
| Bande passante mémoire GPU, lecture streaming | **~94 Go/s** | noyau Metal, 2 Go, 3 passes |
| SSD, preads 13,24 Mo aléatoires parallèles, `F_NOCACHE` | **5,3 – 5,7 Go/s** (4 à 8 threads) | banc C dédié |
| SSD, même motif, 1 thread | 3,0 Go/s | idem |
| SSD, même motif, cache de pages autorisé | jusqu'à 18,0 Go/s | idem |
| Espace disque libre | 126 Gio | `df` |

Trois conséquences immédiates.

**Le plafond de 17,76 Gio confirme ton avertissement** — c'est 74 % des 24 Gio. Le contournement
`sudo sysctl iogpu.wired_limit_mb=<Mo>` reste valide sous macOS 26. Le passer à 20 480 (20 Gio) ne
gagne que **+5 slots d'experts par couche** sur le 120B (27 → 32). C'est un gain réel mais faible,
au prix d'une commande `sudo` et d'un risque de pression mémoire système. **Ma recommandation :
détecter et afficher le plafond, proposer le relèvement comme option explicite et documentée,
mais ne jamais l'exiger ni l'appliquer sans action de l'utilisateur.** L'application doit être
correcte dans le budget par défaut.

**Nous sommes sur apple9, pas apple10.** Le chemin « TensorOps » de TurboFieldfare, qui rend
l'attention 11× plus rapide à 64k, est réservé à apple10 (M5). Sur M4 nous héritons du chemin
tuilé, plus lent. Corollaire important : **le benchmark « 31-35 tok/s sur M5 Pro 24 Go » n'est pas
une cible atteignable pour nous** — le M5 Pro cumule une bande passante mémoire nettement
supérieure et une famille GPU plus récente. La bonne référence mentale est le M2 8 Go (5-6 tok/s),
avec notre avantage de RAM en plus.

**Le `maxBufferLength` de 13,32 Gio** interdit tout `MTLBuffer` unique couvrant le pool d'experts.
Sans importance ici — l'architecture alloue un buffer par slot — mais cela ferme définitivement
l'option « mapper tout le pool en un seul buffer ».

---

## 2. Audit des checkpoints

### 2.1 GPT-OSS — structure exacte

Extrait des en-têtes safetensors réels (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`) :

| Clé (couche 0) | dtype | shape | octets |
| --- | --- | --- | ---: |
| `self_attn.q_proj.weight` | BF16 | [4096, 2880] | 23 592 960 |
| `self_attn.k_proj.weight` | BF16 | [512, 2880] | 2 949 120 |
| `self_attn.v_proj.weight` | BF16 | [512, 2880] | 2 949 120 |
| `self_attn.o_proj.weight` | BF16 | [2880, 4096] | 23 592 960 |
| `self_attn.sinks` | BF16 | [64] | 128 |
| `mlp.router.weight` | BF16 | [E, 2880] | E × 5 760 |
| `mlp.experts.gate_up_proj_blocks` | **U8** | [E, 5760, **90, 16**] | E × 8 294 400 |
| `mlp.experts.gate_up_proj_scales` | **U8** | [E, 5760, **90**] | E × 518 400 |
| `mlp.experts.gate_up_proj_bias` | BF16 | [E, 5760] | E × 11 520 |
| `mlp.experts.down_proj_blocks` | **U8** | [E, 2880, **90, 16**] | E × 4 147 200 |
| `mlp.experts.down_proj_scales` | **U8** | [E, 2880, **90**] | E × 259 200 |
| `mlp.experts.down_proj_bias` | BF16 | [E, 2880] | E × 5 760 |

**Le layout MXFP4 est entièrement déterminé par ces shapes.** `[…, 90, 16]` avec une dimension
d'entrée de 2880 donne 2880 / 90 = **32 valeurs par bloc**, stockées sur **16 octets** (deux FP4
E2M1 par `uint8`), plus **1 octet d'échelle par bloc** (`scales` en U8, soit E8M0). Total
**4,25 bits par poids**. C'est bien le standard OCP Microscaling, avec un bloc de 32 — pas 64
comme le format affine MLX de Gemma 4.

Ton estimation « ~13,2 Mo par blob d'expert » était juste : la valeur exacte est
**13 236 480 octets**, et elle est **identique pour le 20B et le 120B** (même `hidden_size` 2880,
même `intermediate_size` 2880). Ton chiffre de 4,2 Go de poids résidents pour le 120B était juste
aussi : **4 255 115 904 o = 3,963 Gio**.

### 2.2 Les quatre découvertes qui changent la conception

**(a) Il n'y a pas d'expert partagé.** L'inventaire complet des clés du 120B ne contient que
`mlp.router.*` et `mlp.experts.*` — aucun `shared_expert`. La question que tu signalais comme
structurante est tranchée : **le recouvrement CPU/GPU de TurboFieldfare n'est pas transposable.**

Ce que ça coûte réellement : TurboFieldfare a mesuré ce recouvrement à **4,404 → 4,736 tok/s,
soit +7,5 %**. Ce n'est pas l'effondrement qu'on pourrait craindre. Mais chez nous le rapport est
défavorable — leur I/O est de 88 ms/token, la nôtre de 173 à 347 ms — donc la part masquable est
plus grande, et la perte réelle sera supérieure à 7 %.

**Il n'existe aucun substitut propre**, et il faut le dire franchement :
- le routeur dépend de la sortie d'attention de la même couche → impossible de lancer les lectures
  avant la fin de `cb1` ;
- la couche L+1 dépend de la couche L → pas de pipeline inter-couches en décodage ;
- la tête LM du token *t−1* précède l'échantillonnage qui détermine le token *t* → pas de
  recouvrement là non plus ;
- la préchargement spéculatif inter-couches est mort : TurboFieldfare a mesuré que les choix d'une
  couche ne prédisent que **7 %** des experts de la suivante.

**Conclusion assumée : pour le 120B, on ne se bat pas pour le recouvrement, on se bat pour le taux
de hit.** C'est le seul levier qui compte, et il commande toute la conception du cache.

**(b) `sliding_window = 128`.** Une couche sur deux ne regarde que 128 tokens en arrière. Seules
les 18 couches full-attention portent le contexte long. Le KV cache s'effondre à **4,51 Gio à
128k** au lieu des ~9 Gio qu'on obtiendrait sans fenêtre. Le contexte long est presque gratuit —
c'est une excellente nouvelle qui mérite d'être exploitée.

**(c) `embed_tokens` n'a pas besoin d'être résident.** 1,079 Gio, mais on n'en lit qu'**une ligne
par token** (128 en prefill chunké). Il peut rester en `mmap` paginé à la demande, hors du working
set Metal. **1,079 Gio récupérés gratuitement**, soit +87 slots d'experts. À l'inverse `lm_head`
est lu intégralement à chaque token et doit rester résident.

**(d) La tête LM est un poste de calcul majeur.** 1,079 Gio en BF16 lus par token = **11,5 ms sur
les 53 ms** du plancher de calcul du 120B, soit 22 %. La quantifier en MXFP4 la ramènerait à
0,29 Gio : ~8 ms/token économisés **et** 0,8 Gio libérés. C'est le meilleur rapport
gain/effort identifié, mais il modifie les sorties — donc c'est une expérience à valider contre
une référence, pas une décision de conception.

### 2.3 Quelle source Hugging Face pour le repacker

Le dépôt `openai/gpt-oss-120b` propose trois formes. Mesurées :

| Forme | Contenu | Taille |
| --- | --- | ---: |
| racine | 14 safetensors, HF standard | **65 248 815 744 o** (60,77 Gio) |
| `original/` | 7 safetensors, référence PyTorch | ~65,25 Go |
| `metal/model.bin` | fichier unique pour l'implémentation Metal d'OpenAI | 65 238 253 568 o |

**Recommandation : la racine.** Les valeurs MXFP4 y sont déjà séparées en `blocks` / `scales` /
`bias`, exactement le découpage dont le repacker a besoin. Chaque sous-tenseur d'expert est une
plage d'octets contiguë et adressable, donc le repack se fait par requêtes HTTP `Range` bornées,
sans jamais matérialiser un shard. Les deux autres formes imposeraient une transformation de
layout supplémentaire pour aucun bénéfice.

### 2.4 Qwen3.6-35B-A3B — vérification

**Ta correction était juste, et ton candidat aussi** : `Qwen/Qwen3.6-35B-A3B` est bien MoE.
Config réelle vérifiée :

| Paramètre | Valeur |
| --- | ---: |
| `num_experts` | **256** |
| `num_experts_per_tok` | **8** |
| `shared_expert_intermediate_size` | **512** → **oui, il y a un expert partagé** |
| `num_hidden_layers` | 40 |
| `hidden_size` | 2048 |
| `moe_intermediate_size` | 512 |
| `full_attention_interval` | 4 → **10 couches full attn, 30 `linear_attention`** |
| `head_dim` / `num_key_value_heads` | 256 / 2 |
| `max_position_embeddings` | 262 144 |
| `vision_config` | **présent — le modèle est multimodal** |

Conséquences chiffrées : blob d'expert ≈ **1,67 Mo** à 4,25 bits, pool complet **15,94 Gio**,
I/O au pire **535 Mo/token → 97 ms**. Le streaming y fonctionnerait très bien, et l'expert partagé
redonne le recouvrement perdu sur GPT-OSS.

**Mais il y a un problème que je veux signaler tôt.** Les 30 couches `linear_attention` sont du
**Gated DeltaNet** : convolution causale, règle delta récurrente, gating — une famille d'opérateurs
qui n'a **rien en commun** avec l'attention de GPT-OSS. Ce n'est pas « un troisième modèle MoE de
plus » : c'est un second moteur de séquence à écrire et à valider intégralement. À la louche, cela
représente **autant de travail de noyaux que les deux GPT-OSS réunis**. S'ajoute une tour de vision
à ignorer explicitement.

Je ne dis pas que c'est infaisable. Je dis que le placer en priorité 3 sous-estime probablement
son coût d'un facteur 2 à 3, et que **c'est un arbitrage qui t'appartient** — je le pose dans les
questions ouvertes.

---

## 3. Budget mémoire

Hypothèses : scratch réutilisable 512 Mio, KV en FP16, anneaux SWA de 256 lignes
(128 de fenêtre + 128 de marge pour le prefill chunké).

### GPT-OSS 20B — tient entièrement, mais n'en profite pas par défaut

| Poste | Taille |
| --- | ---: |
| Poids résidents | 3,349 Gio |
| Pool d'experts **complet** | 9,467 Gio |
| KV cache à 32k | 0,756 Gio |
| Scratch | 0,500 Gio |
| **Total** | **14,07 Gio** |
| Plafond Metal par défaut | 17,76 Gio |

**Le 20B tient intégralement en mémoire, experts compris.** Le calculateur indique 44 slots
disponibles par couche alors que le modèle n'en a que 32 : le cache est saturé par construction,
le taux de hit est de 100 %, l'I/O est nulle après le chargement. C'est un banc d'essai idéal —
il valide toute la chaîne (téléchargement, repack, noyaux MXFP4, attention, sinks, Harmony, UI)
**sans que le streaming ne masque une erreur de noyau**.

### GPT-OSS 120B — la cible

| Poste | Taille |
| --- | ---: |
| Poids résidents | **3,963 Gio** |
| KV cache 8k / 32k / 128k | 0,290 / 1,134 / 4,509 Gio |
| Scratch | 0,500 Gio |
| Pool d'experts sur disque | 56,805 Gio |

Slots de cache d'experts disponibles (sur 128 par couche) :

| Plafond Metal | ctx 8k | ctx 32k | ctx 128k |
| --- | ---: | ---: | ---: |
| **17,76 Gio (défaut)** | **29** | **27** | 19 |
| 20 Gio (`wired_limit`) | 34 | 32 | 24 |
| 21 Gio (`wired_limit`) | 36 | 34 | 27 |

En sortant `embed_tokens` du working set (§2.2c), ajouter **+2 slots par couche**.

**Lecture : on peut garder 21 à 28 % des experts en cache.** C'est le chiffre qui décide du projet.

### Qwen3.6-35B-A3B

Pool 15,94 Gio + résidents ≈ 2,3 Gio ≈ **18,2 Gio à 4 bits**. Juste au-dessus du plafond par
défaut — donc streaming léger, ou quantization un peu plus agressive, ou `wired_limit` relevé.
Les 30 couches DeltaNet n'ont pas de KV cache mais un état récurrent borné, ce qui rend le
contexte 256k réaliste côté mémoire.

---

## 4. Débit — l'estimation que tu as demandée avant d'investir des semaines

Modèle : `t_token = t_calcul + t_io`, **sans aucun recouvrement**. C'est délibérément pessimiste,
mais c'est le régime honnête pour GPT-OSS puisqu'il n'y a pas d'expert partagé (§2.2a).

### Plancher de calcul (indépendant du SSD)

| Modèle | Octets lus par le GPU / token | à 94 Go/s | plafond |
| --- | ---: | ---: | ---: |
| GPT-OSS 20B | 3 708 077 568 | 39,4 ms | **25,4 tok/s** |
| GPT-OSS 120B | 5 002 896 384 | 53,2 ms | **18,8 tok/s** |

**Ce plafond est structurel.** Même avec un cache parfait et un SSD infiniment rapide, le 120B ne
dépassera pas ~19 tok/s sur ce M4, parce qu'il faut faire transiter 5 Go par token dans une
bande passante de 94 Go/s. Aucune optimisation d'I/O ne franchit ce mur.

### GPT-OSS 120B en fonction du taux de hit

| Taux de hit | I/O / token | t_io | t_token | **tok/s** |
| ---: | ---: | ---: | ---: | ---: |
| 0 % | 1 906 Mo | 347 ms | 400 ms | 2,50 |
| 20 % | 1 525 Mo | 277 ms | 330 ms | 3,03 |
| **30 %** | 1 334 Mo | 243 ms | 296 ms | **3,38** |
| **40 %** | 1 144 Mo | 208 ms | 261 ms | **3,83** |
| **50 %** | 953 Mo | 173 ms | 226 ms | **4,42** |
| 60 % | 762 Mo | 139 ms | 192 ms | 5,21 |
| 80 % | 381 Mo | 69 ms | 123 ms | 8,16 |

Avec 27 slots sur 128 (21 %), un routage uniforme donnerait 21 % de hit. Le routage MoE réel est
sensiblement biaisé — certains experts sont bien plus sollicités — donc **la fourchette réaliste
est 30 à 50 %, soit 3,4 à 4,4 tok/s.** Un recouvrement partiel et le cache de pages macOS peuvent
ajouter 10 à 20 %.

**Réponse directe à ta question « 1 tok/s ou 15 tok/s ? » : environ 4 tok/s.**

C'est utilisable pour de l'analyse de documents, de la revue de code, du travail asynchrone. C'est
inconfortable pour du chat temps réel — environ 3 mots par seconde. Je pense que c'est un résultat
qui vaut le projet, mais tu dois le savoir maintenant, pas en phase 4.

### L'incertitude qu'il faut lever en premier

**Tout repose sur le taux de hit, et je ne peux pas le calculer — il faut le mesurer.**

Il se trouve qu'on l'obtient **gratuitement en phase 1** : le 20B est entièrement résident, donc on
peut instrumenter son routeur et enregistrer la distribution réelle des experts sur des milliers de
tokens, puis simuler hors ligne la courbe « taux de hit LFU en fonction du nombre de slots ». Même
famille, même top-4, même entraînement. Ce n'est pas une preuve pour le 120B, mais c'est un
prédicteur solide obtenu sans écrire une ligne de code supplémentaire.

**Critère de décision proposé : si la courbe extrapolée donne moins de 25 % de hit à 27 slots, on
s'arrête et on réévalue avant d'écrire le repacker du 120B.**

---

## 5. Architecture proposée

### 5.1 Modules

```
HydraCore        contrat de modèle, types, budget, télémétrie
HydraFormat      format .hydra : manifeste, layout, intégrité, lecture par plages
HydraInstall     repacker en streaming HTTP Range -> .hydra, reprise, vérification
HydraMetal       contexte Metal, bibliothèque de noyaux, spécialisation de pipelines
HydraRuntime     graphe d'exécution, cache d'experts, KV cache, prefill/décodage
HydraTokenize    tokenizers + rendu/parsing du format Harmony
HydraCLI         outil en ligne de commande (install, bench, gen, verify)
HydraApp         SwiftUI + AppKit
```

Séparer `HydraInstall` de `HydraRuntime` est délibéré : l'installeur est le seul module autorisé à
parler au réseau, ce qui rend l'invariant mémoire vérifiable par revue de code.

### 5.2 Format `.hydra`

Repris de `.gturbo`, avec deux différences motivées par MXFP4 :

```
gpt-oss-120b.hydra/
  manifest.json          architecture, tailles, SHA-256, version de format
  verified-install.json  reçu de vérification
  resident.bin           attention, routeurs, normes, lm_head, sinks
  embed.bin              embed_tokens, séparé car mappé et non résident (§2.2c)
  experts/
    layout.json          offsets des sous-tenseurs dans un blob
    layer_00.bin ... layer_35.bin
  tokenizer/
```

Chaque `layer_XX.bin` contient E blobs à **stride fixe, aligné sur 16 Kio**. Un blob regroupe
`gate_up_blocks`, `gate_up_scales`, `gate_up_bias`, `down_blocks`, `down_scales`, `down_bias`
dans cet ordre, chaque sous-tenseur aligné sur 64 octets.

Deux points appris de TurboFieldfare et intégrés dès la conception :
- **l'alignement des sous-tenseurs conditionne la largeur des chargements dans les shaders.** TF a
  eu un bug où un chemin 32 bits passait les tests à l'offset zéro puis produisait n'importe quoi
  en décodage réel, parce que les offsets vivants n'étaient alignés que sur 2 octets. On aligne sur
  64 octets **et** les tests de noyaux utilisent des offsets réalistes, jamais zéro.
- **le repack ne déquantifie jamais.** Il recopie les octets MXFP4 tels quels. Le manifeste
  enregistre le format ; toute métadonnée de quantization inconnue est rejetée.

### 5.3 Cache d'experts

Puisque le taux de hit est le seul levier (§2.2a), le cache mérite plus qu'un LFU uniforme :

- **LFU avec récence en départage**, comme TurboFieldfare (mesuré meilleur que LRU : 72,6 → 64,8 ms/token).
- **Allocation de slots non uniforme entre couches.** Toutes les couches n'ont pas la même entropie
  de routage. Répartir les ~1000 slots proportionnellement à la concentration mesurée du routage,
  plutôt qu'également, est une expérience à faible coût et potentiellement le meilleur gain
  disponible. À valider sur la trace du 20B.
- **`pread` explicite, pas `mmap`.** TF a mesuré 0,50 tok/s en `mmap` contre 3,97 en `pread`
  parallèle. La question est tranchée, on ne la rejoue pas.
- **4 à 8 lectures parallèles**, c'est l'optimum mesuré sur cette machine (§1).
- **`F_RDADVISE` désactivé par défaut** — TF a montré que le gain n'est pas reproductible.
- **`F_NOCACHE` sur les fichiers d'experts : à mesurer, pas à décider maintenant.** Avec 12 Gio de
  cache applicatif, laisser macOS en cacher une copie de plus est un gaspillage… ou un cache de
  second niveau utile. Nos mesures montrent 5,5 Go/s sans cache contre 18 Go/s avec. C'est une
  expérience prioritaire.

### 5.4 Pipeline d'exécution (décodage, GPT-OSS)

```
cb1  : norme d'entrée, QKV, RoPE+YaRN, écriture KV, attention (avec sinks),
       projection O, résidu, norme post-attention, routeur -> top-4
CPU  : plan LFU, pread bornés et parallèles pour les slots manquants
       (les experts déjà en cache démarrent immédiatement)
cb2  : MoE top-4 en workgroups persistants, réduction pondérée, résidu
```

Les **workgroups persistants** sont repris de TF sans discussion : c'est leur plus gros gain de
noyau mesuré (phase MoE 239 → 60 ms, débit +51 %). Le noyau coopératif SIMD est un piège documenté
(230 → 527 ms) — on ne le tente pas.

Détails spécifiques à GPT-OSS à implémenter, absents de Gemma 4 :
- **attention sinks** : un logit appris par tête, ajouté au dénominateur du softmax ;
- **RoPE + YaRN** (`factor` 32, base 4096 → 131072, `beta_fast` 32, `beta_slow` 1) ;
- **GQA groupe 8** (64 têtes Q, 8 têtes KV) ;
- **SwiGLU avec `swiglu_limit = 7.0`** — la variante exacte est à recopier depuis
  l'implémentation de référence d'OpenAI, pas à deviner ;
- **biais partout** : `attention_bias = true`, et les experts ont des biais BF16 ;
- **alternance SWA(128)/full** à partir de la couche 0.

### 5.5 Harmony

`openai/harmony` est une bibliothèque **Rust** avec des bindings PyO3. Trois options :

1. **Réimplémenter en Swift.** Contrôle total, zéro dépendance, mais le format est riche (rôles,
   canaux `analysis`/`commentary`/`final`, appels d'outils, niveaux d'effort) et une divergence
   silencieuse dégrade les sorties sans erreur visible.
2. **Lier la crate Rust en statique** via une façade C. Fidélité garantie, mais introduit Rust dans
   la chaîne de build et la signature de l'application.
3. **Réimplémenter en Swift, mais avec un harnais de conformité** : un corpus de conversations de
   référence rendues par la bibliothèque officielle, figées en fixtures, et un test qui exige
   l'égalité octet à octet du rendu et l'équivalence du parsing.

**Je recommande l'option 3.** Elle donne la fidélité de la 2 sans la dépendance, et elle transforme
« est-ce que notre Harmony est correct » en une question testable. C'est cohérent avec l'exigence
« correction avant performance ».

---

## 6. Le compromis généricité / performance

Tu as demandé plusieurs options explicites avant de trancher. Les voici.

**Option A — runtime spécialisé par modèle (choix de TurboFieldfare).**
Un moteur par famille, invariants codés en dur. Performance maximale. Ajouter un modèle = écrire un
moteur. C'est exactement ce que tu refuses.

**Option B — noyaux génériques paramétrés à l'exécution.**
Les dimensions arrivent par `constant` buffers. Un seul jeu de noyaux couvre tous les modèles.
Coût : les bornes de boucle ne sont plus connues du compilateur, le déroulage et l'allocation de
registres se dégradent. Sur des noyaux mémoire-bound comme les nôtres, j'estime la perte à
**10-25 %** — mais c'est une estimation, pas une mesure.

**Option C — contrat déclaratif + spécialisation des pipelines à la construction. ⟵ recommandé**

Le modèle est décrit par une valeur `ModelContract` (couches, motif d'attention, topologie MoE,
format de quantization, gabarit de conversation). Cette description alimente des
**`function_constant` Metal**, résolus **au moment de la construction du pipeline**, une fois au
chargement du modèle.

Le point clé : `hidden_size`, `head_dim`, la taille de bloc de quantization, le top-k deviennent
des **constantes de compilation** dans le shader. Le compilateur déroule et alloue exactement comme
s'ils étaient écrits en dur. **Le code GPU est aussi spécialisé qu'en option A ; seule
l'orchestration Swift est générique.**

Ce que ça coûte honnêtement :
- construction des pipelines au chargement (quelques centaines de ms, une seule fois) ;
- la généricité s'arrête aux **familles d'opérateurs**. Un nouveau format de quantization ou un
  nouvel opérateur de séquence (le Gated DeltaNet de Qwen, précisément) demande du vrai code
  nouveau. Aucune abstraction ne l'évite, et prétendre le contraire serait malhonnête.

Donc : `ModelContract` absorbe les **variations de forme** gratuitement, et rend explicites les
**variations d'opérateur**, qui restent du travail.

**Et je suis d'accord avec ta priorisation : on n'écrit `ModelContract` qu'en phase 3**, une fois
le 20B et le 120B fonctionnels. Deux modèles réels donnent les bons axes de variation ; zéro modèle
donne une abstraction spéculative.

### Sur le double chemin d'exécution (dense résident / streaming MoE)

Tu demandes mon avis argumenté. **Je recommande de ne pas le faire.**

Un chemin dense n'a ni cache d'experts, ni routeur, ni streaming, ni le pipeline `cb1`/io/`cb2`.
C'est un second runtime, pas une variante. Et sa valeur ajoutée est faible : faire tourner
Qwen3.6-27B dense en Q4_K_M sur 24 Gio, c'est précisément ce que llama.cpp, MLX et LM Studio font
déjà très bien. Hydra n'apporterait rien sur ce terrain, tout en doublant sa surface de maintenance
et en diluant la seule chose qui le distingue.

**Ma proposition : Hydra assume d'être un moteur MoE à streaming, et le documente comme une
limite volontaire.** Si tu veux le 27B dense, MLX le fait bien aujourd'hui.

---

## 7. Plan de phases

Chaque jalon a un critère objectif. Un jalon non atteint déclenche une décision, pas un contournement.

### Phase 1 — Chaîne complète sur GPT-OSS 20B, tout résident

Aucun streaming. On valide le repack, les noyaux MXFP4, l'attention, Harmony, l'UI minimale.

| Jalon | Critère de validation |
| --- | --- |
| 1.1 Repacker `.hydra` | **✔ ATTEINT.** 12,82 Gio installés en 778 s (44 Mo/s crête). Empreinte processus **50,9 Mio**, plus gros bloc réseau 3,6 Mio — 0,39 % du checkpoint. 200 fenêtres recomparées à la source amont : toutes concordent. Reprise et atomicité testées. |
| 1.2 Déquantization MXFP4 | **✔ ATTEINT.** Concordance **bit à bit** avec une implémentation de référence indépendante, sur un vecteur couvrant les 16 valeurs E2M1 et les exposants extrêmes. |
| 1.3 Attention + sinks + YaRN | **✔ ATTEINT pour les opérateurs isolés.** RMSNorm, RoPE+YaRN, SwiGLU, attention avec puits et routeur concordent à **1e-12** entre `HydraReference` et une transcription indépendante du code d'OpenAI, et à **< 1e-4** entre les noyaux Metal et `HydraReference`. Reste à assembler la couche complète. |
| 1.4 Forward complet | Décodage glouton, 64 tokens, **séquence identique** à la référence sur 3 prompts. |
| 1.5 Harmony | Rendu **identique octet à octet** aux fixtures de la bibliothèque officielle. |
| 1.6 Débit | **≥ 10 tok/s** en décodage, 20B, contexte 4k. |
| **1.7 Trace de routage** | **Distribution des experts enregistrée sur ≥ 20 000 tokens ; courbe hit/slots produite.** |

> **Point de décision GO/NO-GO.** Si 1.7 extrapole à moins de 25 % de hit à 27 slots pour le 120B,
> on s'arrête et on réévalue avant d'engager la phase 2.

### Phase 2 — Streaming et GPT-OSS 120B

| Jalon | Critère |
| --- | --- |
| 2.1 Streamer d'experts | Slots + LFU + preads parallèles. Débit ≥ 5,0 Go/s soutenu, mesuré en conditions réelles. |
| 2.2 Repack du 120B | 60,77 Gio installés, pic tas < 8 Mio, reprise après interruption vérifiée. |
| 2.3 Correction du 120B | Concordance des logits contre la référence sur un prompt court. |
| 2.4 Budget tenu | RSS + working set Metal **< plafond mesuré**, sur une génération de 512 tokens à 8k. |
| 2.5 Débit du 120B | **≥ 3,0 tok/s** à 8k. En dessous, on ouvre la discussion « mode asynchrone ». |
| 2.6 Réglage du cache | Slots non uniformes, `F_NOCACHE`, granularité de recouvrement — mesurés, pas supposés. |

### Phase 3 — `ModelContract` et généralisation

Extraire l'abstraction **à partir** des deux moteurs qui marchent. Critère : le 20B et le 120B
tournent tous deux via le contrat, **sans régression de débit supérieure à 3 %** (mesurée en
alternance contrôle/candidat).

### Phase 4 — Troisième modèle

Cible à décider (voir questions). Critère : installation et génération correctes, budget tenu.

### Phase 5 — Finition

CLI, serveur local optionnel, télémétrie UI (tok/s, mémoire, taux de hit), distribution.

---

## 8. Ce que je recommande de ne pas faire

Tiré des 102 expériences de TurboFieldfare — autant ne pas repayer ces erreurs :

- `mmap` pour le pool d'experts (0,50 vs 3,97 tok/s) ;
- noyau MoE coopératif SIMD (230 → 527 ms) ;
- préchargement spéculatif inter-couches (7 % de prédictibilité) ;
- `F_RDADVISE` par défaut (gains non reproductibles) ;
- KV cache quantifié en 4 bits (perd son avantage mémoire au-delà de 4k **et** échoue en qualité) ;
- fusion monolithique post-attention/pré-FFN (2,756 → 1,811 tok/s) ;
- réutilisation des argument buffers Metal (−9 % en prefill long) ;
- optimiser un noyau représentant < 1 % du pas de token.

Et une règle de méthode que je propose d'adopter telle quelle : **un candidat devient le défaut
s'il montre un gain reproductible de bout en bout ; s'il est dans le bruit, le défaut ne change
pas.**

---

## 9. Risques

| Risque | Impact | Atténuation |
| --- | --- | --- |
| Taux de hit réel < 25 % | 120B à ~2,5 tok/s | Mesuré en phase 1.7, avant tout engagement |
| Absence d'expert partagé plus coûteuse que prévu | −10 à 20 % | Acceptée et chiffrée ; on optimise le hit, pas le recouvrement |
| **Stockage : 126 Gio pour ~91 Gio de modèles** | Bloquant en phase 4 | Politique de stockage à décider **maintenant** (question 2) |
| SwiGLU/YaRN/sinks subtilement faux | Sorties dégradées sans erreur | Comparaison numérique par couche dès 1.3 |
| Harmony divergent | Modèle inutilisable, silencieusement | Fixtures officielles, égalité octet à octet |
| Gated DeltaNet (Qwen) | Phase 4 × 2-3 en durée | Arbitrage à faire maintenant (question 1) |
| apple9 sans TensorOps | Attention lente en contexte long | Accepté ; contexte prioritaire à décider (question 3) |

---

## 10. Questions ouvertes

Traitées dans le message d'accompagnement. Les six bloquantes :

1. **Le troisième modèle** — Qwen3.6-35B-A3B avec son Gated DeltaNet, ou une cible moins coûteuse ?
2. **Le stockage** — 126 Gio libres, ~91 Gio de modèles. Politique ?
3. **Le contexte prioritaire** — 8k, 32k ou 128k ? Il commande le cache d'experts.
4. **Le plafond Metal** — proposer `iogpu.wired_limit_mb` dans l'application, ou rester au défaut ?
5. **Si le 120B plafonne vers 4 tok/s** — accepter un mode asynchrone assumé ?
6. **macOS minimum et distribution** — macOS 26 et build depuis les sources, ou plus large ?

Et les non bloquantes, qui peuvent attendre la phase 5 : CLI, serveur compatible OpenAI,
tool calling.
