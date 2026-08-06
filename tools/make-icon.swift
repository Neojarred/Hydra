// Fabrique AppIcon.icns à partir d'une illustration carrée.
//
// Une image générée arrive en carré plein : le dessin est posé sur un fond opaque, et le
// squircle n'est qu'un décor peint dessus. Telle quelle, macOS l'affiche comme une tuile
// sombre aux coins visibles, décalée des autres icônes du Dock.
//
// Ce script isole le dessin, le remet aux proportions attendues — la zone utile d'une
// icône macOS occupe 824 points sur 1024, le reste étant la marge que le système réserve
// à l'ombre — et découpe le squircle en rendant l'extérieur transparent.
//
// Usage : swift tools/make-icon.swift source.png Resources/AppIcon.icns

import AppKit
import CoreGraphics
import ImageIO
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("✘ " + message + "\n").utf8))
    exit(1)
}

let arguments = CommandLine.arguments
guard arguments.count == 3 else { fail("usage : make-icon.swift <source.png> <sortie.icns>") }
let sourceURL = URL(fileURLWithPath: arguments[1])
let outputURL = URL(fileURLWithPath: arguments[2])

guard let source = NSImage(contentsOf: sourceURL),
      let sourceCG = source.cgImage(forProposedRect: nil, context: nil, hints: nil)
else { fail("illustration illisible : \(sourceURL.path)") }

// MARK: - Lecture des pixels

let width = sourceCG.width
let height = sourceCG.height
var pixels = [UInt8](repeating: 0, count: width * height * 4)
guard let readContext = CGContext(
    data: &pixels, width: width, height: height, bitsPerComponent: 8,
    bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
else { fail("contexte de lecture impossible") }
readContext.draw(sourceCG, in: CGRect(x: 0, y: 0, width: width, height: height))

func luminance(x: Int, y: Int) -> Int {
    let offset = (y * width + x) * 4
    return (Int(pixels[offset]) * 299 + Int(pixels[offset + 1]) * 587
            + Int(pixels[offset + 2]) * 114) / 1000
}

// MARK: - Détection du squircle

// Le fond entoure le dessin ; on l'échantillonne dans un coin, puis on cherche la boîte
// des pixels qui s'en écartent. Cela évite de coder en dur un recadrage propre à une
// image, qui serait faux à la prochaine.
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
guard maxX > minX, maxY > minY else { fail("aucun dessin détecté — image uniforme ?") }

// Boîte carrée centrée sur le dessin : un recadrage non carré déformerait l'icône.
let side = max(maxX - minX, maxY - minY) + 1
let centerX = (minX + maxX) / 2
let centerY = (minY + maxY) / 2
let crop = CGRect(
    x: max(0, centerX - side / 2), y: max(0, centerY - side / 2),
    width: min(side, width), height: min(side, height))
guard let cropped = sourceCG.cropping(to: crop) else { fail("recadrage impossible") }
print("  dessin détecté : \(Int(crop.width))×\(Int(crop.height)) à "
      + "(\(Int(crop.minX)), \(Int(crop.minY)))")

// MARK: - Rendu masqué

/// Dessine l'illustration dans un canevas carré, à la taille utile d'une icône macOS,
/// en découpant le squircle.
func render(size: Int) -> CGImage? {
    guard let context = CGContext(
        data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return nil }

    context.interpolationQuality = .high
    context.clear(CGRect(x: 0, y: 0, width: size, height: size))

    // 824/1024 de zone utile, et un rayon de coin de 0,2237 — les proportions du gabarit
    // d'icône macOS depuis Big Sur.
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

// MARK: - Écriture de l'iconset

let iconset = FileManager.default.temporaryDirectory
    .appendingPathComponent("Hydra-\(UUID().uuidString).iconset")
try? FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)

// Chaque taille est rendue depuis l'original plutôt que réduite depuis la précédente :
// un rééchantillonnage en cascade empâte les petits formats, ceux qu'on voit le plus.
let variants: [(name: String, size: Int)] = [
    ("icon_16x16", 16), ("icon_16x16@2x", 32),
    ("icon_32x32", 32), ("icon_32x32@2x", 64),
    ("icon_128x128", 128), ("icon_128x128@2x", 256),
    ("icon_256x256", 256), ("icon_256x256@2x", 512),
    ("icon_512x512", 512), ("icon_512x512@2x", 1024),
]

for variant in variants {
    guard let image = render(size: variant.size) else { fail("rendu \(variant.size) impossible") }
    let url = iconset.appendingPathComponent(variant.name + ".png")
    guard let destination = CGImageDestinationCreateWithURL(
        url as CFURL, "public.png" as CFString, 1, nil)
    else { fail("écriture \(variant.name) impossible") }
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
guard iconutil.terminationStatus == 0 else { fail("iconutil a échoué") }

try? FileManager.default.removeItem(at: iconset)
print("✔ \(outputURL.path)")
