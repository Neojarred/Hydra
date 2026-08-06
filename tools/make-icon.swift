// Builds AppIcon.icns from a square illustration.
//
// A generated image arrives as a full square: the artwork sits on an opaque background, and
// the squircle is only decoration painted on top. As-is, macOS shows it as a dark tile with
// visible corners, out of step with the Dock's other icons.
//
// This script isolates the artwork, restores the expected proportions — a macOS icon's live
// area occupies 824 points of 1024, the rest being the margin the system reserves for the
// shadow — and cuts out the squircle by making the outside transparent.
//
// Usage: swift tools/make-icon.swift source.png Resources/AppIcon.icns

import AppKit
import CoreGraphics
import ImageIO
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("✘ " + message + "\n").utf8))
    exit(1)
}

let arguments = CommandLine.arguments
guard arguments.count == 3 else { fail("usage: make-icon.swift <source.png> <output.icns>") }
let sourceURL = URL(fileURLWithPath: arguments[1])
let outputURL = URL(fileURLWithPath: arguments[2])

guard let source = NSImage(contentsOf: sourceURL),
      let sourceCG = source.cgImage(forProposedRect: nil, context: nil, hints: nil)
else { fail("unreadable illustration: \(sourceURL.path)") }

// MARK: - Reading the pixels

let width = sourceCG.width
let height = sourceCG.height
var pixels = [UInt8](repeating: 0, count: width * height * 4)
guard let readContext = CGContext(
    data: &pixels, width: width, height: height, bitsPerComponent: 8,
    bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
else { fail("could not create a read context") }
readContext.draw(sourceCG, in: CGRect(x: 0, y: 0, width: width, height: height))

func luminance(x: Int, y: Int) -> Int {
    let offset = (y * width + x) * 4
    return (Int(pixels[offset]) * 299 + Int(pixels[offset + 1]) * 587
            + Int(pixels[offset + 2]) * 114) / 1000
}

// MARK: - Detecting the squircle

// The background surrounds the artwork; we sample it in a corner, then look for the box of
// pixels that depart from it. That avoids hard-coding a crop specific to one image, which
// would be wrong for the next.
let background = luminance(x: 2, y: 2)
let threshold = 6

var minX = width, minY = height, maxX = -1, maxY = -1
for y in 0..<height {
    for x in 0..<width where abs(luminance(x: x, y: y) - background) > threshold {
        if x < minX { minX = x }
        if x > maxX { maxX = x }
        if y < minY { minY = y }
        if y > maxY { maxY = y }
    }
}
guard maxX > minX, maxY > minY else { fail("no artwork detected — a uniform image?") }

// A square box centred on the artwork: a non-square crop would distort the icon.
let side = max(maxX - minX, maxY - minY) + 1
let centerX = (minX + maxX) / 2
let centerY = (minY + maxY) / 2
let crop = CGRect(
    x: max(0, centerX - side / 2), y: max(0, centerY - side / 2),
    width: min(side, width), height: min(side, height))
guard let cropped = sourceCG.cropping(to: crop) else { fail("cropping failed") }
print("  artwork detected: \(Int(crop.width))×\(Int(crop.height)) at "
      + "(\(Int(crop.minX)), \(Int(crop.minY)))")

// MARK: - Masked rendering

/// Draws the illustration into a square canvas, at a macOS icon's live size, clipping to the
/// squircle.
func render(size: Int) -> CGImage? {
    guard let context = CGContext(
        data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return nil }

    context.interpolationQuality = .high
    context.clear(CGRect(x: 0, y: 0, width: size, height: size))

    // 824/1024 of live area, and a corner radius of 0.2237 — the proportions of the macOS
    // icon template since Big Sur.
    let inset = CGFloat(size) * (1024 - 824) / 2 / 1024
    let box = CGRect(x: inset, y: inset,
                     width: CGFloat(size) - 2 * inset, height: CGFloat(size) - 2 * inset)
    let radius = box.width * 0.2237

    context.addPath(CGPath(roundedRect: box, cornerWidth: radius, cornerHeight: radius,
                           transform: nil))
    context.clip()
    context.draw(cropped, in: box)
    return context.makeImage()
}

// MARK: - Writing the iconset

let iconset = FileManager.default.temporaryDirectory
    .appendingPathComponent("Hydra-\(UUID().uuidString).iconset")
try? FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)

// Each size is rendered from the original rather than downscaled from the previous one:
// cascaded resampling muddies the small sizes, the ones seen most.
let variants: [(name: String, size: Int)] = [
    ("icon_16x16", 16), ("icon_16x16@2x", 32),
    ("icon_32x32", 32), ("icon_32x32@2x", 64),
    ("icon_128x128", 128), ("icon_128x128@2x", 256),
    ("icon_256x256", 256), ("icon_256x256@2x", 512),
    ("icon_512x512", 512), ("icon_512x512@2x", 1024),
]

for variant in variants {
    guard let image = render(size: variant.size) else { fail("rendering \(variant.size) failed") }
    let url = iconset.appendingPathComponent(variant.name + ".png")
    guard let destination = CGImageDestinationCreateWithURL(
        url as CFURL, "public.png" as CFString, 1, nil)
    else { fail("writing \(variant.name) failed") }
    CGImageDestinationAddImage(destination, image, nil)
    CGImageDestinationFinalize(destination)
}

try? FileManager.default.createDirectory(
    at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

let iconutil = Process()
iconutil.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
iconutil.arguments = ["-c", "icns", iconset.path, "-o", outputURL.path]
try iconutil.run()
iconutil.waitUntilExit()
guard iconutil.terminationStatus == 0 else { fail("iconutil failed") }

try? FileManager.default.removeItem(at: iconset)
print("✔ \(outputURL.path)")
