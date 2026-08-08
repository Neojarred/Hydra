import Foundation
import Testing

@testable import HydraCore

/// Gemma 4's expert blob, checked against the published checkpoint's real dimensions.
///
/// These numbers decide how much is read per token and how many slots fit in memory, so they
/// are asserted rather than trusted: `moe_intermediate_size 704`, `hidden_size 2816`,
/// `num_experts 128`, `num_hidden_layers 30`, all from `google/gemma-4-26B-A4B`.
@Suite("Gemma 4 expert blob")
struct GemmaExpertBlobTests {

    private let hidden = 2816
    private let moeIntermediate = 704
    private let experts = 128
    private let layers = 30

    private var blob: GemmaExpertBlobLayout {
        GemmaExpertBlobLayout(hiddenSize: hidden, moeIntermediateSize: moeIntermediate)
    }

    @Test("The sub-tensors match the checkpoint's shapes")
    func matchesTheCheckpoint() {
        // experts.gate_up_proj is (128, 2 × 704, 2816) — per expert, 1408 × 2816 BF16.
        #expect(blob.gateUp.byteCount == 1408 * 2816 * 2)
        // experts.down_proj is (128, 2816, 704).
        #expect(blob.down.byteCount == 2816 * 704 * 2)
        #expect(blob.sourceBytes == 11_894_784)
    }

    /// The alignment invariant the whole slot cache rests on. A blob that did not start on a
    /// page boundary would make every `pread` touch one extra page at each end, across
    /// 240 reads per token.
    @Test("The stride is page-aligned and covers the payload")
    func strideIsPageAligned() {
        #expect(blob.strideBytes % ExpertBlobLayout.pageAlignment == 0)
        #expect(blob.strideBytes >= blob.payloadBytes)
        #expect(blob.payloadBytes >= blob.sourceBytes)
        // Sub-tensors are individually aligned, so the shaders can use wide vector loads.
        for slot in blob.slots {
            #expect(slot.offset % ExpertBlobLayout.tensorAlignment == 0)
        }
    }

    /// D-021 accepted a per-token read of 2.86 GB as the price of running Gemma as published.
    /// If this figure moves, the throughput expectation recorded there moves with it.
    @Test("The per-token read is what D-021 accepted")
    func perTokenReadMatchesTheDecision() {
        let activePerLayer = 8  // top_k_experts
        let bytesPerToken = activePerLayer * layers * blob.sourceBytes
        #expect(bytesPerToken > 2_800_000_000 && bytesPerToken < 2_900_000_000)

        // And the pool that has to fit on disk.
        let pool = layers * experts * blob.sourceBytes
        #expect(pool > 45_000_000_000 && pool < 46_500_000_000)
    }

    /// A Gemma expert holds a quarter of GPT-OSS's parameters and is nearly the same size on
    /// disk — the whole cost of BF16 over MXFP4, stated as a test so it cannot be forgotten.
    @Test("BF16 makes a smaller expert nearly as large as an MXFP4 one")
    func bf16CostsWhatItCosts() {
        let gemmaValues = 1408 * 2816 + 2816 * 704
        let gptOssValues = 2 * 2880 * 2880 + 2880 * 2880
        #expect(gemmaValues * 4 < gptOssValues, "a Gemma expert should be far smaller")
        #expect(Double(blob.sourceBytes) / Double(GptOssConfig.b20.expertBlobBytes) > 0.85)
    }
}
