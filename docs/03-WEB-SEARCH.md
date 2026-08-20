# Web search: a feasibility study

**Written 2026-08-20, against Hydra 0.5.1.** A brief for whoever implements this, written so a
fresh session can start work without rediscovering any of it.

The conclusion first, because it decides everything else: **web search is feasible, and the hard
constraint is not the network, the API, or the tool protocol. It is that this machine reads text
at about 40 tokens a second.** Every design decision below follows from that one number.

---

## 1. The constraint, measured

Gemma 4 26B-A4B Q4, prefill, on the M4:

| prompt | time |
| ---: | ---: |
| 60 tokens | 1.5 s |
| 700 tokens | 18 s |
| 1,143 tokens | 28.6 s |
| 3,421 tokens | 90.5 s |

Qwen 3.6 is about 30 % quicker, 53 tok/s against 40, and the shape is the same.

**A single extracted web page is 2,000 to 4,000 tokens, so reading one costs 50 to 100 seconds.**
Three pages is three to five minutes before the model writes a word. That is not a slow feature;
it is an unusable one.

### Why it is this slow, and why that will not change

A 913-token prompt makes Gemma **read 27.5 GiB from SSD**, against a model that is 15 GiB on
disk. A long prompt routes to nearly every expert in every layer, so the bounded slot cache stops
helping and prefill streams more than the whole model past the GPU. That is not a defect to fix;
it is the direct consequence of the thing Hydra exists to do.

Where the 21 s of a 913-token prefill goes, measured (M-076):

| | Gemma | Qwen |
| --- | ---: | ---: |
| attention, router, dense branch | 9.8 s | 10.0 s |
| the expert GEMMs | 6.7 s | 4.3 s |
| reading experts from SSD | 4.6 s | 2.9 s |

No single pathology, and both models have the same shape. M-061 and M-062 already took one pass
at this and won 43 %. Expect no easy further multiple.

### How the providers you are comparing against do it

They do not have this problem. A hosted endpoint prefills on datacentre accelerators with the
whole model resident and enormous batch parallelism, at thousands to tens of thousands of tokens
a second. The same 3,421 tokens that cost Hydra 90 seconds cost them well under one.

**So do not copy their design.** The usual approach, fetch the top three or five results in full
and paste them into the prompt, assumes reading is free. Here reading is the entire cost, and a
design that minimises tokens fed to the model beats a design that maximises information every
time.

---

## 2. What that implies: snippet-first

The viable shape is a ladder, cheapest rung first, and the model climbs it only when it must.

| rung | tokens | cost | when |
| --- | ---: | ---: | --- |
| search result snippets, 8 to 10 of them | ~700 | **18 s** | always |
| one page, extracted and trimmed | ~2,500 | 65 s | the model asks for it by URL |
| three pages | ~8,000 | 200 s | effectively never |

Ten snippets at 18 seconds is a usable feature. It is roughly what an image already costs, and
users accept that. Everything above the first rung must be a deliberate, visible, second step.

Three consequences for the design:

- **Snippets are the product, not a preview.** Most questions that need the web need a fact, a
  date, a name or a link, and a good snippet carries it. Build for answering from snippets and
  treat full-page reads as the exception.
- **Prefer a search API that returns clean, short, ranked text.** Raw SERP JSON spends its budget
  on markup and navigation chrome. This is the one place where paying for a better-shaped
  response converts directly into seconds saved.
- **Trim hard before the model sees anything.** Strip navigation, cookie banners, footers,
  scripts. Every 400 tokens removed is 10 seconds returned.

---

## 3. What already exists in the codebase

This is the good news: most of the seams are already there, and one model was trained for it.

### The three models all support tool calling natively

Each checkpoint's `chat_template.jinja` already defines a tool protocol. **They are three
completely different protocols**, and each needs its own renderer and parser:

| model | shape |
| --- | --- |
| Qwen 3.6 | XML-ish: `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>` |
| Gemma 4 | its own `declaration:` block, results wrapped in `<\|tool_response>` … `<tool_response\|>` |
| GPT-OSS | Harmony channels: a call goes to `commentary` addressed `to=functions.name` |

**GPT-OSS ships a built-in `browser` tool namespace** that OpenAI trained it against, alongside
`python`. If one model is the place to start, it is this one: the model already knows the shape
of a browsing tool and Hydra's Harmony implementation already documents `commentary` as the
tool-call channel, though it does not yet parse it.

### The architecture has the right seam already

`ConversationFormat` is per model and already carries both halves of what a tool protocol needs:

```swift
protocol ConversationFormat {
    func render(turns: [ChatTurn], settings: PromptSettings) -> String
    func makeParser(tokenizer: BPETokenizer, settings: PromptSettings) -> any ConversationParser
}
```

So tool support is three implementations behind one existing interface, exactly as the three
prompt formats and the two vision towers already are. Nothing structural has to be invented.

### What has to change

1. **`PromptEvent` gains a case.** Today it is `.answer`, `.reasoning`, `.stopped`. It needs
   `.toolCall(name:arguments:)`. Every parser gains the ability to emit it; every consumer gains
   a branch. The decode loop in `InferenceEngine.generate` is the only real consumer.

2. **The generate loop becomes a loop.** Today it prefills once and decodes until stopped. It
   becomes: decode until `.toolCall` or `.stopped`; on a call, run the search, append the result
   as a turn, prefill the new tokens, continue. Bounded by a maximum number of rounds.

3. **`Message.Role` gains a tool role**, or messages gain an attachment kind for tool results, so
   a search result is stored in the conversation and re-renders on the next turn. Today the enum
   is `case user, assistant`. Follow how images were added: an **optional** field on the existing
   type, so conversations saved before it decode unchanged.

4. **A search client.** `URLSession` is already used by the installer, and there is a
   `StreamingHTTPClient` with range support. The app is unsandboxed, so no entitlement is needed.

### The cache reuse works in your favour

This is worth knowing before you design the loop. Hydra already keeps the prefix a turn shares
with the previous one and prefills only what follows. A tool result is **appended** to the
conversation, so the second pass through the model pays only for the new tokens, not for the
whole prompt again. A tool loop is therefore not a multiple of the prefill cost; it is the sum of
each round's new text.

**But note the exception**: a conversation carrying an image reprefills from nothing every turn,
because the reuse machinery reasons in flat token arrays and an image is not one. A search inside
an image conversation therefore pays full price each round. Either fix that first or accept it.

---

## 4. The backend

You already run a SearXNG instance, which resolves the question that would otherwise be the
awkward part of this study. It is worth stating why, and what the alternatives cost, because the
shipped default has to work for someone who does not run one.

| option | free allowance as of 2026-08 | notes |
| --- | --- | --- |
| **your SearXNG** | unlimited, private | needs `format=json` enabled in `settings.yml`; JSON is off by default |
| Exa | 20,000 requests a month | neural, ranks by meaning; clean snippets |
| Serper | 2,500 searches once | raw SERP, cheapest per query, more junk tokens |
| Tavily | 1,000 credits a month | search and extraction in one call, agent-shaped output |
| Brave | ~1,000 a month via a $5 credit | **the free tier was withdrawn in February 2026** |
| Google CSE | 100 a day | |

Two things follow.

**Self-hosted SearXNG is the right default for this project and a bad default for the product.**
It is free, private, keyless, and it fits Hydra's premise exactly: nothing leaves the machine
except the query itself. But public instances rate-limit and block under sustained use and return
inconsistent JSON, and requiring a user to run Docker is not a shipping story. Design the client
around a small protocol with a SearXNG implementation first, and leave room for a keyed provider.

**The privacy tension is real and should be stated in the UI, not buried.** Hydra's entire pitch
is that the model runs on your machine and your conversation does not leave it. A web search
sends your query to a third party. That is a defensible trade the user should make knowingly:
search off by default, an explicit control, and the endpoint visible.

---

## 5. Extraction

Getting from a URL to clean text is its own problem and it is where the token budget is won.

- **`NSAttributedString(html:)` exists on macOS and is a trap.** It must run on the main thread,
  it is slow, and it retains layout text. It will fight the design rather than serve it.
- The realistic options are a small hand-written HTML-to-text pass over the semantic content, or
  an extraction library ported or bridged. Readability-style heuristics, take the densest text
  block, drop `nav`, `header`, `footer`, `aside`, `script`, `style`, get most of the win.
- **Budget the output in tokens, not characters**, and truncate to fit a stated ceiling. A hard
  cap of, say, 2,000 tokens a page turns an unbounded cost into a known one.
- Tavily and Exa return extracted text as part of the search response, which removes this whole
  section if you use them. That is most of what you are paying for.

---

## 6. A suggested order of work

Each step is separately useful and separately testable, which matters because this project's
failures have consistently been at the seams between components rather than inside them.

1. **A `WebSearch` protocol and a SearXNG client**, with a `hydra search "query"` CLI command.
   No model involved. Verifies the endpoint, the JSON shape, and the snippet budget in tokens.
2. **Tool rendering and parsing for one model**, GPT-OSS first because it was trained with a
   browser tool and its channel format already has a place for the call. Add `.toolCall` to
   `PromptEvent`. Test the parser against captured token streams, and make sure the test streams a
   call split across token boundaries: the markers straddle tokens and that is where the emoji bug
   lived.
3. **The tool loop in the engine**, bounded to two or three rounds, with the search result
   appended as a turn. Assert that a conversation without a tool call is byte-identical to today,
   the same way the multimodal prefill asserts it.
4. **The UI**: a toggle, the endpoint, a visible "searching…" state, and the sources listed under
   the answer. The token cost of the results should be visible before they are fed, exactly as an
   image's cost is shown on its chip.
5. **Then, and only then, full-page fetch**, as a second tool the model must ask for by URL.

## 7. Things that will bite

Collected because each one has an analogue that already bit this project once.

- **Markers split across tokens.** The tool-call syntax will straddle token boundaries. Qwen's
  parser accumulated a `String` and corrupted every emoji for exactly this reason (M-071). The
  parsers accumulate **bytes** and decode at a character boundary; keep it that way.
- **A model that will not stop calling.** Bound the rounds and make the bound visible.
- **The stop-token budget.** `reservedStopTokens` exists per format and will need revisiting once
  a tool call can appear; a truncated call is worse than no call.
- **Injection from page content.** Text fetched from the web is data, not instruction. A page that
  says "ignore previous instructions" must not be obeyed. This is a genuine security property and
  the tool result should be clearly delimited in the prompt.
- **Test the shipped path.** Four times this year a suite passed while exercising a code path
  production does not take: the wrong attention kernel, the wrong quantization group width, the
  narrow-group projector path, the numbered-only list detector. Whatever fixture you build, check
  it reaches the kernel, the format and the branch that actually ship.
- **Measure the instrument before the result.** Six measurements this session were wrong about
  their own method rather than about the world. Before trusting a benchmark, run the known-bad
  case through it and confirm it reports the failure.

## 8. The honest verdict

**Feasible, worth doing, and only in the snippet-first shape.**

The protocol work is straightforward and the architecture already has the seams. The backend is
solved for you personally and needs a small abstraction to be solved for everyone else. The real
design work is the token budget, and the answer there is not clever: send the model less.

What would make it genuinely good rather than merely working is the thing this hardware makes
hard, reading several sources in full and reasoning across them. That costs minutes here and
milliseconds in a datacentre, and no amount of kernel work closes a gap that large. Ship the rung
that fits, be honest in the interface about what the next rung costs, and let the user choose to
pay it.
