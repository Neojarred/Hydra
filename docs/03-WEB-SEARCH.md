# Web search: what was built, and why it is two things

**Written 2026-08-22, against Hydra 0.7.0.** This replaces the feasibility study that stood here
until the feature existed. The study's central claim survived contact: on a machine that reads
40 tokens a second, a design that minimises tokens fed to the model beats a design that
maximises information. Almost everything else it proposed was changed by measurement, and the
changes are the interesting part.

The conclusion first, because it decides the shape: **the hard problem was not the network, the
API or the tool protocol. It was that one of the two models cannot be trusted to think.**

---

## 1. What a search costs

Snippets, never pages. The study's arithmetic held:

| | tokens | to read, on Qwen | on Gemma |
| --- | ---: | ---: | ---: |
| one extracted page | 2,000 to 4,000 | 50 to 100 s | 60 to 120 s |
| eight snippets, one chunk each | ~1,000 | 19 s | 25 s |

A searching turn adds roughly twenty seconds, which is about what an image already costs and
which users accept. The block is budgeted **in tokens, through the loaded model's own
tokenizer**, not in characters: on one real result the ratio ran from 3.1 to 4.9 characters a
token, so a character budget is wrong by half on the pages that tokenize worst, and wrong in the
optimistic direction.

Where the tokens go, measured on a recorded response at a 1,000-token ceiling:

| | tokens |
| --- | ---: |
| snippets | 699 |
| titles | 157 |
| preamble | 111 |
| hosts | 34 |

URLs are rendered as **hosts**. A full URL costs 23 tokens and a host costs 4, which over eight
results is 136 tokens, and trimming them took the block from five results to seven inside the
same ceiling. The whole URL stays in the structured response and is what the interface links to,
so nothing the user can click is lost.

Nothing is dropped silently. Results are ranked, so the block fills from the top and stops; it
never skips a long third result to fit a short fourth, which would hand the model a ranking the
provider did not produce. What did not fit is counted and shown.

## 2. The provider

Tavily, and the user brings the key.

The study recommended a self-hosted SearXNG default and that was wrong for a shipping product:
it means either shipping Docker or shipping a feature that works for one person. The deciding
argument for a keyed API was not index quality but that `chunks_per_source` bounds the returned
text **server side**, so the prompt budget is a request parameter instead of something recovered
afterwards by trimming HTML. That deletes the extraction problem, which was the part of the
study with no cheap answer.

The free tier is a thousand searches a month with no card. Hydra ships no key of its own: no
free tier in this market survives being shared between everyone who downloads a build. The key
lives in the keychain, never in `conversations.json`.

**One thing to revisit.** LM Studio's search plugin defaults to DuckDuckGo, which needs no key at
all. A keyless default would remove the largest adoption barrier this feature has, and the
token-control argument above is about Tavily's advantages over raw SERP JSON rather than about
DuckDuckGo specifically.

## 3. Two designs, one per model

This is the part no study would have predicted.

### Gemma decides for itself

Gemma is offered the tool and left to judge. That is what its checkpoint was trained for, it is
what every other runtime does, and it works: asked something it knows, it answers without
searching and spends no credit; asked about something recent, it searches once, cites each claim
by number, and does not repeat itself.

The protocol is the template's own, transcribed rather than approximated:

```
declaration   <|tool>declaration:web_search{description:<|"|>...<|"|>,
              parameters:{properties:{...},required:[...],type:<|"|>OBJECT<|"|>}}<tool|>
call          <|tool_call>call:web_search{query:<|"|>...<|"|>}<tool_call|>
result        <|tool_response>response:web_search{value:...}<tool_response|>
```

Two properties matter. **Every marker is a single token**: `<|tool_call>` is 48 and nothing else
ever is, so the parser matches on identity and the straddling that forced Qwen's holdback, and
that M-071 was about, cannot occur. And **a result continues the model's own turn**: the
template opens no new turn after a call and emits no generation prompt, so the model simply
carries on.

Gemma calls from **inside** its reasoning trace. Runtimes that read only the content field lose
the call entirely and are advised to disable thinking to get it back; here the marker is matched
before any channel is considered, so where it arrives does not matter.

### Qwen is never asked

Qwen has the same dialect and is deliberately not offered it. Letting it decide meant letting it
deliberate, and deliberation is where it fails. Its turn is split instead:

1. The conversation is prefilled. `prefill` ends by checkpointing the recurrence.
2. A continuation asks for a search query, thinking closed, bounded to 64 tokens.
3. The recurrence rewinds to that checkpoint, so nothing is reprocessed and the query pass
   leaves no trace in the answer's context.
4. The results are fed and the answer is written, thinking closed.

The query pass has never degenerated in any run. It is twenty tokens and it does not think.

Removing the declaration also returned the 317 tokens it cost at the head of every prompt: a
searching Qwen turn prompts at 36 tokens where it used to prompt at 448.

## 4. The date, which was the whole difference

Neither Qwen's template nor Gemma's states the date. Harmony's does, which is why this only ever
showed up on the other two.

A model that does not know the year defends its training data against the pages in front of it.
Handed a search result whose address was literally `huggingface.co/Qwen/Qwen3.8-27B`, Qwen wrote
"there is no evidence that a model named Qwen3.8 27B exists" and cited that result in the next
sentence. Gemma, with no date, spent 1,400 tokens concluding the model could not exist and never
called the tool at all.

With the date, Gemma reads it, cross-checks a release date against a result, and searches:

> Qwen 3.8 27B was released recently (around mid-August 2026 according to result [2]) ...
> Since today is August 22, 2026, it is indeed very new

The note is rendered **only when the turn may read the web**, because it changes at midnight and
sits at the head of the prompt: rendered always, it would cost every conversation its cached
prefix once a day for nothing.

Two further things were needed, each found by a case that broke:

- **The trust instruction must sit after the results**, not only in the system turn. The same
  words a thousand tokens earlier did nothing.
- **It must permit synthesis.** The first wording said "report what the sources say", which is
  right for a lookup and makes the model refuse every comparison no single source answers
  outright.

## 5. What was measured and did not work

Recorded because the reasons are more useful than the conclusion.

| | outcome |
| --- | --- |
| bounding the presence penalty's window | null on its own, worst case got worse |
| Qwen's own published thinking recipe (`presence_penalty` 0) | worst configuration measured, 321-token loop |
| a frequency penalty at 0.5 | fewest loops, and the grammar breaks |
| thinking on for Qwen's answer, after the results are in | degenerates on all three seeds |
| `enable_thinking:false` for Gemma, the ecosystem's advice | unnecessary once it is told the date |

The engine was cleared as a cause, with evidence: the recurrent state is fp32 throughout, the
chunked recurrence is the step recurrence in a loop with no precision-sensitive chunk algebra,
greedy output is byte identical between prefill chunk sizes at 4,488 tokens, and prefill still
equals token by token at 512, an invariant the suite previously checked at ten.

Full numbers, including two withdrawn claims, are in `02-MEASUREMENTS.md` M-077.

## 6. Known limits

- **Qwen 3.6 is unreliable when it reasons at length.** Search routes around that rather than
  fixing it. A thinking turn that does **not** search can still repeat itself, and about one
  searching turn in sixteen still shows a visible repeat.
- **Gemma is verbose.** One searching answer ran 2,600 output tokens against 1,300 of prompt and
  results, filling a 4k context in a single turn. The reasoning is the larger half.
- **GPT-OSS has no search.** Its Harmony parser also has two known defects found while reading
  it: the recipient in `to=functions.name` is discarded, and a `commentary json` channel name
  falls through to `.final`, which would print a call's arguments into the answer.
- **The evaluation is thin.** Eight cases at two seeds, one model at a time.

## 7. Tools

- `hydra search "query"` prices a block in tokens through a real tokenizer, with a breakdown.
  `--save` records a response and `--from` renders a saved one, spending no credits.
- `hydra chat --search` runs the whole turn headlessly, printing the query, the search and the
  feed. `--search-from` replays a recorded response: no network, no query pass, so two prompts
  can be compared on a fixed input. That exists because two identical queries a minute apart
  returned 990 and 996 tokens, which is enough to move a generation onto another trajectory and
  which invalidated several comparisons before it was noticed.
- `tools/search-eval.sh` runs eight cases across a spread of question shapes, including a
  deliberately fictitious entity, and refuses to report until its detector passes a self-test.
- `tools/degeneration.py` scores repetition as a magnitude, and proves itself on known-bad and
  known-good text before any batch is believed.
